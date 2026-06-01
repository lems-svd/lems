"""
Shared-basis variant of the LEMS search.

Extends :class:`.lems.LEMSSearch` to support shared-basis factorization by:
- calling ``lrd_method._build_shared_groups()`` during initialization,
- tracking per-layer ``rank_list`` in ``prepare_data()``, and
- passing ``shared_rank_groups`` to the ILP solvers so that grouped layers
  are forced to the same compression ratio.

All ILP solver logic, ``ILPSettings``, and grid-search / Optuna machinery are
imported from :mod:`.lems` — no duplication.
"""

from torch import nn

from ..factorization._interface_sharing import BaseFactorization, get_valid_layers
from ..factorization._interface import get_eq_rank, _find_decoder_layers

from .lems import (
    LEMSSearch,
    ILPSettings,
    ilp_search_gurobi,
    ilp_search_cbc,
    compress_model as _lems_compress_model,
    restore_model,
    get_depth_multiplier,
)

import copy
import gc
import torch


class LEMS_SHAREDSearch(LEMSSearch):
    """
    Shared-basis variant of the LEMS search.

    Overrides only the methods that need shared-group awareness:
    ``initialize_search``, ``prepare_data``, ``single_search``.
    Everything else (``grid_search``, ``get_layer_wise_flops``, ``search``)
    is inherited unchanged.
    """

    def initialize_search(self, lrd_method: BaseFactorization, model: nn.Module, spec_tensor=None):
        self.lrd_method = lrd_method
        self.lrd_method._build_shared_groups(model, self.name_omit)
        self.ilp_settings = ILPSettings(
            layer_wise_monotone=False,
            block_wise_monotone=False,
            in_block_monotone=False,
            in_block_monotone_qkv=False,
            solver="gurobi",
        )
        layer_sensitivity, size_dict = self._get_layer_sensitivity(model, spec_tensor)
        self.sensitivity_dict = layer_sensitivity
        self.size_dict = size_dict
        self.shape_dict = self.get_layer_shape_dict(model)

    def prepare_data(self, size_dict, layer_sensitivity, compression_target, layers_per_block):
        is_vision = self.lrd_method.vision
        data = []
        layer_name_list = list(layer_sensitivity.keys())
        compression_list = []
        rank_list = []
        active_layer_sizes = []

        upper_bound_offset = 1.0
        if self.lrd_method.vision or compression_target < 0.5:
            lower_bound = 0.1
        else:
            lower_bound = 0.3
        upper_bound = compression_target + upper_bound_offset

        for i, (layer_name, sensitivity_data) in enumerate(layer_sensitivity.items()):
            n, m = self.shape_dict[layer_name]
            eq_rank = get_eq_rank(n, m)

            if self.enforce_rank_multiples_of is not None:
                sensitivity_data = {
                    k: v for k, v in sensitivity_data.items()
                    if int(k * eq_rank) % self.enforce_rank_multiples_of == 0
                }

            current_block = i // layers_per_block
            total_blocks = max(len(layer_name_list) // layers_per_block, 1)
            if is_vision:
                depth_multiplier = 1.0
            else:
                depth_multiplier = get_depth_multiplier(
                    current_block=current_block,
                    total_blocks=total_blocks,
                    crosslayer_term=self.crosslayer_term,
                    alpha=self.alpha,
                    gamma=self.gamma,
                    is_vision=is_vision,
                )

            layer_data = []
            layer_compression_list = []
            layer_rank_choices = []

            # Dense option
            dense_cost = (n * m) if self.target_metric == "params" else size_dict[layer_name]
            layer_data.append((dense_cost, 0.0))
            layer_compression_list.append(1.0)
            layer_rank_choices.append(eq_rank)

            for compression, sensitivity in sensitivity_data.items():
                if compression < lower_bound or compression > upper_bound:
                    continue
                rank = self.lrd_method.get_search_candidate_rank(
                    layer_name=layer_name, shape_dict=self.shape_dict, compression=compression,
                )
                cand_size = self.lrd_method.get_candidate_size(
                    layer_name=layer_name, shape=(n, m), rank=rank, metric=self.target_metric,
                )
                layer_data.append((cand_size, sensitivity * depth_multiplier))
                layer_compression_list.append(compression)
                layer_rank_choices.append(rank)

            data.append(layer_data)
            compression_list.append(layer_compression_list)
            rank_list.append(layer_rank_choices)
            active_layer_sizes.append(n * m)

        total_parameters = sum(active_layer_sizes)
        compression_param_target = total_parameters * compression_target
        shared_rank_groups = self.lrd_method.get_shared_groups_for_search(layer_name_list)
        return data, layer_name_list, compression_list, rank_list, compression_param_target, shared_rank_groups

    def single_search(self, layers_per_block, default_param_ratio):
        data, layer_name_list, compression_list, rank_list, compression_param_target, shared_rank_groups = self.prepare_data(
            size_dict=self.size_dict,
            layer_sensitivity=self.sensitivity_dict,
            compression_target=self.ratio_target,
            layers_per_block=layers_per_block,
        )

        if self.ilp_settings.solver == "gurobi":
            compression_dict = ilp_search_gurobi(
                data=data,
                compression_list=compression_list,
                rank_list=rank_list,
                layer_name_list=layer_name_list,
                compression_param_target=compression_param_target,
                ilp_settings=self.ilp_settings,
                layers_per_block=layers_per_block,
                shared_rank_groups=shared_rank_groups,
            )
        else:
            compression_dict = ilp_search_cbc(
                data=data,
                compression_list=compression_list,
                rank_list=rank_list,
                layer_name_list=layer_name_list,
                compression_param_target=compression_param_target,
                ilp_settings=self.ilp_settings,
                layers_per_block=layers_per_block,
                shared_rank_groups=shared_rank_groups,
            )

        layers_min_ratio = {
            layername: default_param_ratio for layername in self.sensitivity_dict.keys()
        }
        for layername, param_ratio in compression_dict.items():
            layers_min_ratio[layername] = param_ratio

        return layers_min_ratio


def compress_model(module_dict, module_bkup_dict, rank_dict, lrd_method: BaseFactorization):
    """Delegate to the factorization method's shared-aware compress_module_dict."""
    lrd_method.compress_module_dict(module_dict, module_bkup_dict, rank_dict)
