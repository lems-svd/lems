import gc

import torch
from torch import nn

from ._dataclasses import UnefficientFactorizedMatrix
from ._interface import BaseFactorization
from ._interface import FactorizedMatrix
from ._interface import Hookstuff
from ._interface import get_valid_layers
from ._interface import whitening, safe_whitened_svd
from tqdm import tqdm


class KFAC_SVD_Hook(Hookstuff):
    """
    Forward/backward hook pair for KFAC-SVD factorisation.

    Parameters
    ----------
    fast : bool
        When *True* the backward hook uses ``torch.einsum`` and accumulates
        covariances directly on the GPU (faster, uses more VRAM).
        When *False* (default) it uses pinned-memory double-buffered CPU
        accumulation (slower, lower peak VRAM).
    """
    def __init__(self, model, name_omit, dump_shape=False,
                 white_list: list = [], name_prefix="",
                 vision=False, fast=False):
        super().__init__(model=model, name_omit=name_omit,
                         dump_shape=dump_shape, white_list=white_list,
                         name_prefix=name_prefix)
        self.vision = vision
        self.fast = fast

    def _hook_fn(self, layer_name, last_feat=False):
        def get_scaling_mat(module, input, output):
            x = self._reshape_input(input[0].detach().clone())
            if self._maybe_record_shape(layer_name, x, module):
                return
            if last_feat:
                if "head" in layer_name:
                    self.model.last_feat = x
                return
            self.activation_cache[layer_name] = x
        return get_scaling_mat

    # ----- backward hook dispatch -----

    def _bw_hook_fn(self, layer_name):
        if self.fast:
            return self._bw_hook_fn_fast(layer_name)
        return self._bw_hook_fn_standard(layer_name)

    # ----- standard: pinned-memory CPU accumulation -----

    def _bw_hook_fn_standard(self, layer_name):
        def get_scaling_mat_grad(module, ginput, goutput, last_feat=False):
            if not self.layer_trigger:
                self.layer_trigger = layer_name
            gout = self._reshape_input(goutput[0].detach().clone().float())

            x = self.activation_cache[layer_name].detach().float()
            seq_len = x.shape[1]
            cutoff = 0 if self.vision else int(seq_len * 0.125)
            gout_middle = gout[:, cutoff:seq_len - cutoff, :]

            gout_middle = self.clip_token_norms(gout_middle)

            batch_column_scaling = x.transpose(1, 2) @ x
            batch_row_scaling = gout_middle.transpose(1, 2) @ gout_middle

            row_scaling = torch.mean(batch_row_scaling, dim=0).float()
            column_scaling = torch.mean(batch_column_scaling, dim=0).float()

            self._accumulate_scaling(layer_name, row_scaling, column_scaling)

            del module, goutput, ginput, gout, batch_row_scaling, batch_column_scaling, x
        return get_scaling_mat_grad

    # ----- fast: GPU-only einsum accumulation -----

    def _bw_hook_fn_fast(self, layer_name):
        def get_scaling_mat_grad(module, ginput, goutput, last_feat=False):
            if not self.layer_trigger:
                self.layer_trigger = layer_name

            gout = self._reshape_input(goutput[0].detach().float())

            x = self.activation_cache[layer_name].detach().float()
            seq_len = x.shape[1]
            batch_size = x.shape[0]

            cutoff = 0 if self.vision else int(seq_len * 0.125)
            gout_middle = gout[:, cutoff:seq_len - cutoff, :]

            gout_middle = self.clip_token_norms(gout_middle)

            # Efficient covariance via einsum (avoids [B, D, D] allocation)
            row_scaling = torch.einsum('bni,bnj->ij', gout_middle, gout_middle) / batch_size
            column_scaling = torch.einsum('bni,bnj->ij', x, x) / batch_size

            # GPU accumulation (no synchronisation locks, no CPU transfers)
            if layer_name not in self.row_scale:
                self.row_scale[layer_name] = row_scaling.clone()
                self.column_scale[layer_name] = column_scaling.clone()
            else:
                self.row_scale[layer_name] += row_scaling
                self.column_scale[layer_name] += column_scaling

            del module, goutput, ginput, gout, x, gout_middle
        return get_scaling_mat_grad


class KFAC_SVDFactorization(BaseFactorization):
    """
    KFAC-SVD factorisation with optional *fast* mode and optional truncation.

    Parameters
    ----------
    fast : bool
        When *True*, uses GPU-only accumulation and ``torch.einsum`` in the
        backward hook (the former ``kfac_svd_fast`` variant).
    truncate : bool
        When *True* (default), the SVD result is truncated to ``eq_rank``
        and returned as a :class:`FactorizedMatrix`.  When *False*, all
        singular components are kept and returned as an
        :class:`UnefficientFactorizedMatrix` (the former
        ``kfac_svd_no_truncate`` variant).
    """
    def __init__(self, vision, fast=False, truncate=True, *args, **kwargs):
        super().__init__(vision=vision, *args, **kwargs)
        self.fast = fast
        self.truncate = truncate
        self.column_scaling_dict = {}
        self.row_scaling_dict = {}

    @property
    def post_search_calibration(self):
        if not self.truncate:
            # no-truncate mode: default to True for LLM, False for vision
            if self._do_post_calibration == "default":
                return not self.vision
            return self._do_post_calibration
        # truncate mode: inherit the base default (False)
        return False if self._do_post_calibration == "default" else self._do_post_calibration

    def _compute_scaling(self, model, hook_module, name_prefix, calib_data, name_omit,
                         mixup_fn=None, white_list=[], tqdm_message="Gathering "):
        dev = self.dev

        extractor = KFAC_SVD_Hook(
            hook_module, name_omit, False,
            name_prefix=name_prefix, vision=self.vision,
            white_list=white_list, fast=self.fast,
        )
        extractor.attach_hooks()
        extractor.attach_bw_hooks()

        self._enable_weight_grads(model, hook_module, name_omit, white_list)
        self._run_backward_calib(model, calib_data, mixup_fn, dev, desc=tqdm_message)

        if self.fast:
            # fast mode: accumulated on GPU, bulk-transfer to CPU now
            self.column_scaling_dict.update(
                {k: v.cpu() for k, v in extractor.column_scale.items()})
            self.row_scaling_dict.update(
                {k: v.cpu() for k, v in extractor.row_scale.items()})
        else:
            # standard mode: finalise pinned double-buffer
            for layer_name in extractor.row_scale:
                torch.cuda.synchronize()
                extractor.row_scale[layer_name] += extractor.buf_1[layer_name]
                extractor.column_scale[layer_name] += extractor.buf_2[layer_name]
            self.column_scaling_dict.update(extractor.column_scale)
            self.row_scaling_dict.update(extractor.row_scale)

        extractor.clear_hooks()
        del extractor.activation_cache, extractor.buf_1, extractor.buf_2
        del extractor.row_scale, extractor.column_scale, extractor
        gc.collect()

    def _factorize_matrix(self, matrix, eq_rank, rank, name, dev, verbose=False):
        if rank == 0:
            rank = eq_rank
        elif rank > eq_rank:
            print(f"Warning: {name} rank is larger than equivalent rank!")
            return

        print("Factorizing matrix:", name) if verbose else None
        dev = torch.device(torch.cuda.current_device())

        col_alpha = 0.1
        row_alpha = 0.7 if self.vision else 0.3
        column_scale_diag, column_scale_diag_inv = whitening(dev, self.column_scaling_dict, name, alpha=col_alpha)
        row_scale_diag, row_scale_diag_inv = whitening(dev, self.row_scaling_dict, name, alpha=row_alpha)

        dtype_final = matrix.dtype

        if self.truncate:
            result_cls = FactorizedMatrix
            svd_rank = rank
        else:
            result_cls = UnefficientFactorizedMatrix
            svd_rank = min(matrix.shape[0], matrix.shape[1])

        mat_l, mat_r, s = safe_whitened_svd(
            matrix, row_scale_diag, row_scale_diag_inv,
            column_scale_diag, column_scale_diag_inv, svd_rank, name,
        )

        self.row_scaling_dict[name] = self.row_scaling_dict[name].to('cpu', non_blocking=True)
        self.column_scaling_dict[name] = self.column_scaling_dict[name].to('cpu', non_blocking=True)
        return result_cls(
            mat_l=mat_l.cpu().to(dtype_final),
            mat_r=mat_r.cpu().to(dtype_final),
            eq_rank=eq_rank,
            active_rank=rank,
            singular_values=s,
        )
