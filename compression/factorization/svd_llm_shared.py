"""
Shared-basis variant of the SVD-LLM factorization.

Reuses ``SVD_LLM_Hook`` and per-layer logic from :mod:`.svd_llm` and extends
the shared ``BaseFactorization`` from :mod:`._interface_sharing` with a
group-SVD implementation for shared right factors.
"""

import gc
import torch

from ._interface import whitening, FactorizedMatrix
from ._interface_sharing import BaseFactorization, SharedFactorizedGroup
from .svd_llm import SVD_LLM_Hook  # identical hook – no need to duplicate


class SVD_LLM_SHAREDFactorization(BaseFactorization):
    def __init__(self, vision=False, whitening_method="cholesky", whitening_reg_increment=1e-6, *args, **kwargs):
        super().__init__(vision=vision, *args, **kwargs)
        self.whitening_method = whitening_method
        self.whitening_reg_increment = whitening_reg_increment
        print(f"increment = {whitening_reg_increment}")
    @property
    def post_search_calibration(self):
        return False if self._do_post_calibration == "default" else self._do_post_calibration

    # ------------------------------------------------------------------
    #  Scaling collection (delegates to SVD_LLM_Hook, identical to svd_llm)
    # ------------------------------------------------------------------

    def _compute_scaling(self, model, hook_module, name_prefix, calib_data,
                         name_omit, mixup_fn=None, white_list=[], tqdm_message="Gathering "):
        from tqdm import tqdm
        torch.cuda.empty_cache()
        model.config.use_cache = False
        model = model.eval().to(self.dev)
        extractor = SVD_LLM_Hook(
            model=hook_module, name_omit=name_omit, dump_shape=False,
            name_prefix=name_prefix, white_list=white_list,
        )
        extractor.attach_hooks()
        if self.vision:
            with torch.no_grad():
                for data, target in tqdm(calib_data, desc=tqdm_message + "(Activations for SVD-LLM (shared))"):
                    model_inps, targets = mixup_fn(data, target) if mixup_fn is not None else (data, target)
                    model_inps = model_inps.to(self.dev)
                    model(model_inps)
                    del model_inps, targets
                extractor.dump_shape = True
                dummy_input = torch.randn(20, 3, 224, 224).to(self.dev)
                model(dummy_input)
                for key, value in extractor.input_shape.items():
                    self.input_shapes[key] = value
                del dummy_input
        else:
            with torch.no_grad():
                for batch in tqdm(calib_data, desc=tqdm_message + "(Activations for SVD-LLM (shared))"):
                    batch = {k: v.to(self.dev) for k, v in batch.items()}
                    model(**batch)
                    del batch

        extractor.clear_hooks()
        for key, value in extractor.profile.items():
            self.scaling_dict[key] = value
        del extractor
        torch.cuda.empty_cache()
        gc.collect()

    # ------------------------------------------------------------------
    #  Per-layer factorization (same maths as svd_llm)
    # ------------------------------------------------------------------

    def _factorize_matrix(self, matrix, eq_rank, rank, name, dev, verbose=False):
        raw_profile = self.scaling_dict
        scale_diag, scale_diag_inv = whitening(dev, raw_profile, name, method=self.whitening_method, increment=self.whitening_reg_increment)

        if rank == 0:
            rank = eq_rank
        elif rank > eq_rank:
            print(f"Warning: {name} rank is larger than equivalent rank!")
            return

        dtype = matrix.dtype
        mat_scaled = torch.matmul(matrix.float().to(dev), scale_diag)
        u, s, vh = torch.linalg.svd(mat_scaled, full_matrices=False)

        mat_l = (u * s.unsqueeze(0))[:, :rank].cpu().to(dtype)
        mat_r = torch.matmul(vh, scale_diag_inv)[:rank, :].cpu().to(dtype)

        torch.cuda.empty_cache()
        gc.collect()

        return FactorizedMatrix(
            mat_l=mat_l, mat_r=mat_r,
            eq_rank=eq_rank, active_rank=rank, singular_values=s.cpu(),
        )

    # ------------------------------------------------------------------
    #  Shared-group factorization (unique to the shared variant)
    # ------------------------------------------------------------------

    def _shared_scale_name(self, full_name: str) -> str:
        return full_name

    def _factorize_shared_group(self, matrices, names, eq_rank, rank, dev, verbose=False):
        self._dprint(
            f"_factorize_shared_group() | members={len(names)} | eq_rank={eq_rank}"
        )

        scale_names = [self._shared_scale_name(n) for n in names]
        group_scale = sum(self.scaling_dict[n] for n in scale_names)

        tmp_profile = {"__tmp__": group_scale}
        scale_diag, scale_diag_inv = whitening(dev, tmp_profile, "__tmp__", method=self.whitening_method, increment=self.whitening_reg_increment)

        scale_diag = scale_diag.double()
        scale_diag_inv = scale_diag_inv.double()

        dtype = matrices[0].dtype
        out_dims = [mat.shape[0] for mat in matrices]

        scaled_blocks = [mat.double().to(dev) @ scale_diag for mat in matrices]
        mat_scaled_group = torch.cat(scaled_blocks, dim=0)

        u, s, vh = torch.linalg.svd(mat_scaled_group, full_matrices=False)

        mat_l_cat = u[:, :eq_rank].cpu().to(dtype)
        mat_r = ((s[:eq_rank].unsqueeze(1) * vh[:eq_rank, :]) @ scale_diag_inv).cpu().to(dtype)

        mat_ls = [m.to(dtype) for m in mat_l_cat.split(out_dims, dim=0)]

        torch.cuda.empty_cache()
        gc.collect()

        return SharedFactorizedGroup(
            mat_ls=mat_ls, mat_r=mat_r,
            eq_rank=eq_rank, active_rank=rank, layer_names=tuple(names),
        )
