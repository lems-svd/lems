"""
Unified SVD-LLM factorization.

Supports the following modes via constructor flags:

- ``whitening_method="cholesky"`` (default) or ``"svd"`` (former ``svd_llmv2``).
- ``memory_efficient=True`` for CPU-offloaded hooks and factorization (former ``svd_llm_large``).
- ``truncate=True`` (default) or ``False`` to keep all singular components (former ``svd_llm_no_truncate``).
"""

import gc
import torch
import torch.nn as nn
from tqdm import tqdm

from ._dataclasses import UnefficientFactorizedMatrix
from ._interface import (
    BaseFactorization,
    FactorizedMatrix,
    Hookstuff,
    get_valid_layers,
    whitening,
)


# ---------------------------------------------------------------------------
#  Hook
# ---------------------------------------------------------------------------

class SVD_LLM_Hook(Hookstuff):
    """Forward hook that accumulates ``X^T X`` covariance matrices.

    Parameters
    ----------
    memory_efficient : bool
        When *True*, the outer-product sum is moved to CPU immediately
        (lower peak VRAM, slower due to PCIe traffic).
    """

    def __init__(self, *args, memory_efficient: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_efficient = memory_efficient

    def _hook_fn(self, layer_name, last_feat=False):
        def get_scaling_mat(module, input, output):
            x = self._reshape_input(input[0].detach().float())
            if self._maybe_record_shape(layer_name, x, module):
                return
            if last_feat:
                if "head" in layer_name:
                    self.model.last_feat = x.clone()
                return

            out_prod = torch.matmul(x.transpose(1, 2), x)
            outpro_sum = out_prod.sum(dim=0)
            if self.memory_efficient:
                outpro_sum = outpro_sum.cpu()

            if layer_name not in self.profile:
                self.profile[layer_name] = outpro_sum
            else:
                self.profile[layer_name] += outpro_sum

            del x, out_prod, outpro_sum, output
            torch.cuda.empty_cache()

        return get_scaling_mat


# ---------------------------------------------------------------------------
#  Factorization
# ---------------------------------------------------------------------------

class SVD_LLMFactorization(BaseFactorization):
    """
    Unified SVD-LLM factorization.

    Parameters
    ----------
    whitening_method : ``"cholesky"`` | ``"svd"``
        Whitening strategy.  ``"svd"`` is the former SVD-LLMv2 variant.
    memory_efficient : bool
        When *True*, hooks offload covariance sums to CPU, model is
        round-tripped to CPU during factorization, and SVD can run on
        CPU (the former ``svd_llm_large`` variant for 70B+ models).
    truncate : bool
        When *False*, returns :class:`UnefficientFactorizedMatrix`
        keeping all singular components (the former
        ``svd_llm_no_truncate`` variant).
    """

    def __init__(self, vision=False, whitening_method="cholesky",
                 memory_efficient=False, truncate=True, whitening_reg_increment=1e-6, *args, **kwargs):
        super().__init__(vision=vision, *args, **kwargs)
        self.whitening_method = whitening_method
        self.memory_efficient = memory_efficient
        self.truncate = truncate
        self.whitening_reg_increment = whitening_reg_increment

    @property
    def post_search_calibration(self):
        if self.memory_efficient or not self.truncate:
            # Large / no-truncate modes default to no recalibration
            return False if self._do_post_calibration == "default" else self._do_post_calibration
        return super().post_search_calibration

    def get_cache_name(self) -> str:
        decomp_name = super().get_cache_name()
        if self.whitening_method != "cholesky":
            decomp_name += f"_whitening_{self.whitening_method}"
        return decomp_name

    # ------------------------------------------------------------------
    #  Scaling (hook-based covariance collection)
    # ------------------------------------------------------------------

    def _compute_scaling(self, model, hook_module, name_prefix, calib_data,
                         name_omit, mixup_fn=None, white_list=[],
                         tqdm_message="Gathering "):
        torch.cuda.empty_cache()
        model = model.eval().to(self.dev)

        extractor = SVD_LLM_Hook(
            model=hook_module, name_omit=name_omit, dump_shape=False,
            name_prefix=name_prefix, white_list=white_list,
            memory_efficient=self.memory_efficient,
        )
        extractor.attach_hooks()

        if self.memory_efficient:
            # Memory-efficient calibration with per-batch GC
            if self.vision:
                with torch.inference_mode():
                    for data, target in tqdm(calib_data, desc=tqdm_message + "(Activations for SVD-LLM)"):
                        model_inps, _ = mixup_fn(data, target) if mixup_fn is not None else (data, target)
                        model(model_inps.to(self.dev))
                        del model_inps
            else:
                with torch.no_grad():
                    for batch in tqdm(calib_data, desc=tqdm_message + "(Activations for SVD-LLM)"):
                        batch = {k: v.to(self.dev) for k, v in batch.items()}
                        model(**batch, use_cache=False)
                        del batch
                        gc.collect()
                        torch.cuda.empty_cache()
        else:
            # Standard calibration
            self._run_forward_calib(model, calib_data, mixup_fn, self.dev,
                                    desc=tqdm_message + "(Activations for SVD-LLM)")

        if self.vision:
            extractor.dump_shape = True
            dummy_input = torch.randn(20, 3, 224, 224).to(self.dev)
            model(dummy_input)
            for key, value in extractor.input_shape.items():
                self.input_shapes[key] = value
            del dummy_input

        extractor.clear_hooks()
        for key, value in extractor.profile.items():
            self.scaling_dict[key] = value
        extractor.profile.clear()
        del extractor

        torch.cuda.empty_cache()
        gc.collect()

    # ------------------------------------------------------------------
    #  Memory-efficient overrides (model <-> CPU round-trips)
    # ------------------------------------------------------------------

    def factorize_model(self, uncom_model, rank_dict, name_omit, verbose=True, apply_fact=True):
        if self.memory_efficient:
            uncom_model = uncom_model.cpu()
            torch.cuda.empty_cache()
            gc.collect()
        super().factorize_model(
            uncom_model=uncom_model, rank_dict=rank_dict,
            name_omit=name_omit, verbose=verbose, apply_fact=apply_fact,
        )
        if self.memory_efficient:
            uncom_model = uncom_model.to(self.dev)

    def _get_scale_and_factorize_module(
        self, model, hook_module, name_prefix, calib_data,
        name_omit, mixup_fn=None, white_list=[], tqdm_message="Gathering ",
    ):
        if not self.memory_efficient:
            return super()._get_scale_and_factorize_module(
                model=model, hook_module=hook_module, name_prefix=name_prefix,
                calib_data=calib_data, name_omit=name_omit, mixup_fn=mixup_fn,
                white_list=white_list, tqdm_message=tqdm_message,
            )
        # Memory-efficient path: CPU round-trip between scaling and factorization
        # Ignore this if you don't compress very large models.
        self._compute_scaling(
            model=model, hook_module=hook_module, name_prefix=name_prefix,
            calib_data=calib_data, name_omit=name_omit, mixup_fn=mixup_fn,
            white_list=white_list, tqdm_message=tqdm_message + "scalings...",
        )
        torch.cuda.empty_cache()
        model = model.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        if self.use_local_cache:
            copied_modules = get_valid_layers(hook_module, name_omit, white_list=white_list)
            for name, module_sub in tqdm(copied_modules, desc=tqdm_message + "factorizations..."):
                name = f"{name_prefix}{name}"
                rank, ratio, cntinue = self._get_active_rank(module_sub.weight.shape, name)
                if cntinue:
                    continue
                det_weight = module_sub.weight.clone().detach()
                _ = self.factorize_matrix(
                    matrix=det_weight, rank=rank, ratio=ratio, name=name, verbose=False,
                )
                det_weight = None
                del det_weight

        model = model.to(self.dev)
        torch.cuda.empty_cache()
        gc.collect()

    # ------------------------------------------------------------------
    #  Matrix factorization
    # ------------------------------------------------------------------

    def _factorize_matrix(self, matrix, eq_rank, rank, name, dev, verbose=False):
        raw_profile = self.scaling_dict

        # Memory-efficient mode: do SVD on CPU
        fact_dev = torch.device("cpu") if (self.memory_efficient and getattr(self, "compute_memory_efficient", False)) else dev

        use_double = not self.vision
        scale_diag, scale_diag_inv = whitening(
            fact_dev, raw_profile, name,
            method=self.whitening_method,
            increment=self.whitening_reg_increment,
            double_precision=use_double,
        )

        if rank == 0:
            rank = eq_rank
        elif rank > eq_rank:
            print(f"Warning: {name} rank is larger than equivalent rank!")
            return

        if self.truncate:
            result_cls = FactorizedMatrix
            svd_rank = rank
        else:
            result_cls = UnefficientFactorizedMatrix
            svd_rank = min(matrix.shape[0], matrix.shape[1])

        dtype = matrix.dtype
        mat_scaled = torch.matmul(matrix.float().to(fact_dev), scale_diag)

        u, s, vh = torch.linalg.svd(mat_scaled, full_matrices=False)
        s_val = torch.sqrt(s)
        mat_l = (u * s_val.unsqueeze(0))[:, :svd_rank]
        mat_r = (s_val.unsqueeze(1) * torch.matmul(vh, scale_diag_inv))[:svd_rank, :]

        mat_l = mat_l.cpu().to(dtype)
        mat_r = mat_r.cpu().to(dtype)

        del mat_scaled, scale_diag, scale_diag_inv
        gc.collect()
        torch.cuda.empty_cache()

        return result_cls(
            mat_l=mat_l,
            mat_r=mat_r,
            eq_rank=eq_rank,
            active_rank=rank,
            singular_values=s,
        )
