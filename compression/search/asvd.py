from torch import nn

from ..factorization._interface import BaseFactorization
from ._sensitivity_base import SensitivityBasedSearch
import torch


class ASVDSearch(SensitivityBasedSearch):
    """
    ASVD-style threshold search via binary search over a sorted sensitivity list.

    Reference::

        @misc{yuan2025asvd,
            title={{ASVD}: Activation-aware Singular Value Decomposition for Compressing Large Language Models},
            author={Zhihang Yuan and Yuzhang Shang and Yue Song and Dawei Yang and Qiang Wu and Yan Yan and Guangyu Sun},
            year={2025},
            url={https://openreview.net/forum?id=HyPofygOCT}
        }
    """

    def __init__(
        self,
        ratio_target=0.5,
        sensitivity_loss="ce",
        measurements_points="asvd_default",
        target_metric="params",
        min_ratio=0.0,
        max_ratio=1.0,
        *args,
        **kwargs,
    ):
        super().__init__(
            ratio_target=ratio_target,
            sensitivity_loss=sensitivity_loss,
            measurements_points=measurements_points,
            *args,
            **kwargs,
        )
        self.target_metric = target_metric
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    @property
    def requires_decomposed_model_for_search(self):
        return True

    def initialize_search(
        self, lrd_method: BaseFactorization, model: nn.Module, spec_tensor=None
    ):
        self.lrd_method = lrd_method
        layer_sensitivity, _ = self._get_layer_sensitivity(model, spec_tensor)
        self.sensitivity_dict = layer_sensitivity

    # ------------------------------------------------------------------
    #  Sensitivity list construction (overridden by ASVD+ for bias)
    # ------------------------------------------------------------------

    def _build_sensitivity_list(self, model: nn.Module):
        """Build ``(layer_name, param_ratio, score)`` triples from the
        sensitivity dict, filtering by ``min_ratio`` / ``max_ratio``."""
        sensitivity_list = []
        for layername, v in self.sensitivity_dict.items():
            for param_ratio, score in v.items():
                if param_ratio >= self.max_ratio or param_ratio < self.min_ratio:
                    continue
                sensitivity_list.append((layername, param_ratio, score))
        return sensitivity_list

    # ------------------------------------------------------------------
    #  Core binary search
    # ------------------------------------------------------------------

    def _realized_ratio(self, layers_min_ratio, module_dict):
        tot_params = 0
        compress_params = 0
        for layername, param_ratio in layers_min_ratio.items():
            raw_linear = module_dict[layername]
            if self.target_metric == "flops":
                tot_params += self._flops_dict[layername]
                compress_params += self._flops_dict[layername] * param_ratio
            else:
                n_params = raw_linear.weight.numel()
                tot_params += n_params
                compress_params += n_params * param_ratio
        return compress_params / tot_params

    def single_search(self, model: nn.Module):
        """Binary search over the sorted sensitivity threshold."""
        module_dict = {name: module for name, module in model.named_modules()}

        if self.target_metric == "flops":
            self._flops_dict, _ = self.get_layer_wise_flops(model)

        default_param_ratio = 1.0

        sensitivity_list = self._build_sensitivity_list(model)
        if len(sensitivity_list) == 0:
            return {
                layername: default_param_ratio
                for layername in self.sensitivity_dict.keys()
            }

        # Higher score = more harmful → sort descending
        sorted_sensitive_list = sorted(sensitivity_list, key=lambda x: -x[2])

        low = 0
        high = len(sorted_sensitive_list) - 1
        best_idx = 0

        while low <= high:
            mid = (low + high) // 2
            layers_min_ratio = {
                layername: default_param_ratio
                for layername in self.sensitivity_dict.keys()
            }
            for layername, param_ratio, _ in sorted_sensitive_list[mid:]:
                layers_min_ratio[layername] = min(
                    layers_min_ratio[layername], param_ratio
                )
            now_ratio = self._realized_ratio(layers_min_ratio, module_dict)

            if now_ratio <= self.ratio_target:
                best_idx = mid
                low = mid + 1
            else:
                high = mid - 1

        print("=== Searching done, decomposing layers... ===")
        layers_min_ratio = {
            layername: default_param_ratio
            for layername in self.sensitivity_dict.keys()
        }
        for layername, param_ratio, _ in sorted_sensitive_list[best_idx:]:
            layers_min_ratio[layername] = min(
                layers_min_ratio[layername], param_ratio
            )

        cumulative_error = 0.0
        for layername, param_ratio in layers_min_ratio.items():
            if param_ratio != default_param_ratio:
                cumulative_error += self.sensitivity_dict[layername][param_ratio]
        print(f"cumulative error: {cumulative_error}")

        return layers_min_ratio

    def search(self, model: nn.Module):
        return self.single_search(model)