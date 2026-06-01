"""
Shared-basis extension for the low-rank factorization interface.

This module builds on :mod:`._interface` by adding support for **shared-basis
factorization**, where groups of linear layers (e.g. Q/K/V projections in the
same decoder block) share a common right factor while keeping independent left
factors.

Backward compatibility
----------------------
All public names that used to live here (``BaseFactorization``,
``FactorizedMatrix``, ``get_valid_layers``, ``get_eq_rank``, etc.) are
re-exported so that existing ``from ._interface_sharing import X`` statements
continue to work.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import gc
import os
import pickle
import torch
from torch import nn
from tqdm import tqdm
from all_utils.model_io import get_model_name

# ---------------------------------------------------------------------------
#  Re-export everything that _interface already provides
# ---------------------------------------------------------------------------
from ._interface import (                      # noqa: F401 – re-exports
    whitening,
    FactorizedMatrix,
    is_linear_like_conv,
    get_valid_layers,
    get_eq_rank,
    _find_decoder_layers,
    BaseFactorization as _OrigBaseFactorization,
    SeqSVD,
    SeqSVDMemViT,
    Hookstuff,
    ShapeHook,
    plot_compression_rates,
)
from ._dataclasses import UnefficientFactorizedMatrix  # noqa: F401 – re-exported

# Keep the name ``BaseFactorization`` pointing at the *shared* subclass that
# this file defines (see below) so that ``svd_llm_shared.py`` and
# ``lems_shared.py`` can import it unchanged.  The original base is
# available as ``_OrigBaseFactorization`` when needed.


# ---------------------------------------------------------------------------
#  Sharing-specific helpers and data classes
# ---------------------------------------------------------------------------

def canonical_linear_part(name: str) -> str:
    short = name.split(".")[-1].replace("_proj", "")
    if short == "out":
        short = "o"
    return short


class NamedModuleSubset:
    """
    Lightweight view that exposes only selected named modules through
    ``named_modules()``.  This is enough for ``get_valid_layers(...)`` and
    hook registration.
    """
    def __init__(self, named_modules_list):
        self._named_modules_list = named_modules_list

    def named_modules(self):
        yield "", self
        for name, module in self._named_modules_list:
            yield name, module


@dataclass
class SharedFactorizedGroup:
    """
    One shared right factor and one left factor per grouped layer.
    Like ``FactorizedMatrix``, this stores the full decomposition once and
    exposes active-rank views dynamically.
    """
    mat_ls: List[torch.Tensor]               # one [out_i, eq_rank] tensor per layer
    mat_r: torch.Tensor                       # [eq_rank, in]
    eq_rank: int
    active_rank: int = 0
    layer_names: Tuple[str, ...] = ()

    def __post_init__(self):
        new_ls = []
        for mat_l in self.mat_ls:
            if mat_l.is_cuda:
                mat_l = mat_l.cpu()
            if mat_l.shape[1] > self.eq_rank:
                mat_l = mat_l[:, :self.eq_rank]
            new_ls.append(mat_l)
        self.mat_ls = new_ls

        if self.mat_r.is_cuda:
            self.mat_r = self.mat_r.cpu()
        if self.mat_r.shape[0] > self.eq_rank:
            self.mat_r = self.mat_r[:self.eq_rank, :]

    def left_factor(self, idx: int) -> torch.Tensor:
        return self.mat_ls[idx][:, :self.active_rank]

    @property
    def right_factor(self) -> torch.Tensor:
        return self.mat_r[:self.active_rank, :]


# ---------------------------------------------------------------------------
#  SharedBaseFactorization – extends BaseFactorization with shared-basis logic
# ---------------------------------------------------------------------------

class BaseFactorization(_OrigBaseFactorization):
    """
    Drop-in replacement for :class:`._interface.BaseFactorization` that adds
    optional shared-basis factorization.  When ``use_shared_basis=False``
    (the default), it behaves identically to the original base class.
    """

    def __init__(
        self,
        vision,
        calib_dataset_name,
        use_cache=True,
        blockwise_factorization=False,
        progressive_compression=False,
        do_post_calibration="default",
        calibration_ranks={},
        # Shared-basis config --------------------------------------------------
        use_shared_basis: bool = False,
        shared_group_size: int = 2,
        shared_part: list = None,
        private_part: list = None,
        **kwargs,
    ):
        super().__init__(
            vision=vision,
            calib_dataset_name=calib_dataset_name,
            use_cache=use_cache,
            blockwise_factorization=blockwise_factorization,
            progressive_compression=progressive_compression,
            do_post_calibration=do_post_calibration,
            calibration_ranks=calibration_ranks,
            # debug=True,
            **kwargs,
        )

        # Shared-basis attributes
        self.use_shared_basis = use_shared_basis
        self.shared_group_size = shared_group_size
        self.shared_part = set(shared_part or [])
        self.private_part = set(private_part or [])

        # Built dynamically from model structure
        self.shared_groups: Dict[str, List[str]] = {}
        self.layer_to_shared_group: Dict[str, str] = {}
        self.shared_group_anchor: Dict[str, str] = {}

    # ------------------------------------------------------------------
    #  Cache name includes shared-basis config
    # ------------------------------------------------------------------

    def get_cache_name(self) -> str:
        decomp_name = super().get_cache_name()

        if self.one_shot_factorization:
            decomp_name += "_oneshot"
        else:
            decomp_name += "_blockwise"

        if self.use_shared_basis:
            shared_tag = "-".join(sorted(self.shared_part)) if self.shared_part else "none"
            private_tag = "-".join(sorted(self.private_part)) if self.private_part else "none"
            decomp_name += f"_shared_g{self.shared_group_size}_{shared_tag}_{private_tag}"
        else:
            decomp_name += "_noshared"

        return decomp_name

    # ------------------------------------------------------------------
    #  Shared-group building
    # ------------------------------------------------------------------

    def _build_shared_groups(self, model: nn.Module, name_omit):
        self.shared_groups = {}
        self.layer_to_shared_group = {}
        self.shared_group_anchor = {}

        self._dprint(
            f"Building shared groups | use_shared_basis={self.use_shared_basis} "
            f"| shared_group_size={self.shared_group_size} "
            f"| shared_part={sorted(self.shared_part)}"
        )

        if not self.use_shared_basis or self.shared_group_size <= 1 or not self.shared_part:
            self._dprint("Shared groups disabled or no valid shared_part configured.")
            return

        layers, mod_name = _find_decoder_layers(model)
        self._dprint(f"Decoder container found: {mod_name} | num_layers={len(layers)}")

        per_part = defaultdict(list)
        for l_idx, layer in enumerate(layers):
            name_prefix = f"{mod_name}.{l_idx}."
            for name, module_sub in get_valid_layers(layer, name_omit, white_list=[]):
                full_name = f"{name_prefix}{name}"
                part = canonical_linear_part(name)
                if part in self.shared_part:
                    per_part[part].append(full_name)

        for part, names in per_part.items():
            self._dprint(f"Shared-candidate part='{part}' | found_layers={len(names)}")
            gid = 0
            for start in range(0, len(names), self.shared_group_size):
                members = names[start:start + self.shared_group_size]
                if len(members) < 2:
                    continue
                group_id = f"{part}:{gid}"
                gid += 1
                self.shared_groups[group_id] = members
                self.shared_group_anchor[group_id] = members[0]
                for n in members:
                    self.layer_to_shared_group[n] = group_id
                self._dprint(f"Created shared group '{group_id}' | size={len(members)}")

        self._dprint(f"Total shared groups built: {len(self.shared_groups)}")

    # ------------------------------------------------------------------
    #  Shared-group rank / size helpers
    # ------------------------------------------------------------------

    def _get_shared_group_storage_rank_from_shapes(self, shapes: List[Tuple[int, int]]) -> int:
        if len(shapes) == 0:
            raise ValueError("Expected at least one shape.")
        in_dims = [s[1] for s in shapes]
        out_dims = [s[0] for s in shapes]
        if len(set(in_dims)) != 1:
            raise ValueError(f"Shared input-side basis requires matching in_features, got {in_dims}")
        shared_in = in_dims[0]
        total_out = sum(out_dims)
        return int(shared_in * total_out / (shared_in + total_out))

    def _get_shared_group_storage_rank(self, matrices: List[torch.Tensor]) -> int:
        return self._get_shared_group_storage_rank_from_shapes(
            [tuple(mat.shape) for mat in matrices]
        )

    def _make_shared_group_cache_key(self, group_id: str) -> str:
        return f"__shared__::{group_id}"

    def get_search_candidate_rank(self, layer_name, shape_dict, compression):
        n, m = shape_dict[layer_name]
        if (not self.use_shared_basis) or (layer_name not in self.layer_to_shared_group):
            return int(get_eq_rank(n, m) * compression)

        group_id = self.layer_to_shared_group[layer_name]
        members = [n_ for n_ in self.shared_groups[group_id] if n_ in shape_dict]
        in_dims = [shape_dict[n_][1] for n_ in members]
        out_dims = [shape_dict[n_][0] for n_ in members]
        if len(set(in_dims)) != 1:
            raise ValueError(f"Shared group {group_id} mismatch in_features: {dict(zip(members, in_dims))}")
        shared_in = in_dims[0]
        dense_total = sum(o * shared_in for o in out_dims)
        stored_den = shared_in + sum(out_dims)
        return int(compression * dense_total / stored_den)

    def get_candidate_size(self, layer_name, shape, rank, metric="params"):
        n, m = shape
        if metric == "flops":
            return rank * (n + m)
        if not self.use_shared_basis or layer_name not in self.layer_to_shared_group:
            return rank * (n + m)
        group_id = self.layer_to_shared_group[layer_name]
        anchor = self.shared_group_anchor[group_id]
        shared_cost = rank * m
        coeff_cost = rank * n
        return (shared_cost + coeff_cost) if layer_name == anchor else coeff_cost

    def get_shared_groups_for_search(self, layer_name_list: List[str]) -> List[List[int]]:
        name_to_idx = {n: i for i, n in enumerate(layer_name_list)}
        out = []
        for _, members in self.shared_groups.items():
            idxs = [name_to_idx[n] for n in members if n in name_to_idx]
            if len(idxs) >= 2:
                out.append(idxs)
        return out

    def get_unique_model_num_parameters(self, model: nn.Module, trainable_only=False, verbose=False) -> int:
        total = 0
        seen_param_ids = set()
        for name, param in model.named_parameters(recurse=True, remove_duplicate=False):
            if param is None or (trainable_only and not param.requires_grad):
                continue
            pid = id(param)
            if pid in seen_param_ids:
                continue
            seen_param_ids.add(pid)
            total += param.numel()
        return total

    # ------------------------------------------------------------------
    #  Shared-group rank resolution
    # ------------------------------------------------------------------

    def _resolve_shared_group_rank(self, member_names, module_dict, rank_dict):
        values = [rank_dict[n] for n in member_names]
        if all(isinstance(v, int) for v in values):
            if len(set(values)) != 1:
                raise ValueError(f"Shared group has mismatched int ranks: {dict(zip(member_names, values))}")
            v = values[0]
            if v == 0:
                shapes = [tuple(module_dict[n].weight.shape) for n in member_names]
                return self._get_shared_group_storage_rank_from_shapes(shapes)
            return v
        if all(isinstance(v, float) for v in values):
            if len(set(values)) != 1:
                raise ValueError(f"Shared group has mismatched ratios: {dict(zip(member_names, values))}")
            c = values[0]
            if c <= 0.0:
                return 0
            if c >= 1.0:
                shapes = [tuple(module_dict[n].weight.shape) for n in member_names]
                return self._get_shared_group_storage_rank_from_shapes(shapes)
            in_dims = [module_dict[n].weight.shape[1] for n in member_names]
            out_dims = [module_dict[n].weight.shape[0] for n in member_names]
            if len(set(in_dims)) != 1:
                raise ValueError(f"Shared input-side basis requires same in_features, got {in_dims}")
            shared_in = in_dims[0]
            dense_total = sum(o * shared_in for o in out_dims)
            stored_den = shared_in + sum(out_dims)
            return int(c * dense_total / stored_den)
        raise ValueError(f"Shared group values must be all int or all float: {dict(zip(member_names, values))}")

    def _resolve_requested_rank(self, shape, key, default_ratio=None):
        rank, ratio, cntinue = self._get_active_rank(
            shape=shape, key=key,
            default_ratio=default_ratio if default_ratio is not None else 1.0,
        )
        if cntinue:
            return None
        n, m = shape
        eq_rank = get_eq_rank(n, m)
        if rank == 0:
            return eq_rank
        elif rank != -1:
            return rank
        elif ratio != -1:
            return int(eq_rank * ratio)
        return None

    # ------------------------------------------------------------------
    #  Factorize shared groups
    # ------------------------------------------------------------------

    def factorize_shared_group(
        self, matrices, names, rank, verbose=True,
    ) -> SharedFactorizedGroup:
        if len(names) < 2:
            raise ValueError("factorize_shared_group requires at least two layers.")
        group_id = self.layer_to_shared_group.get(names[0], None)
        if group_id is None:
            raise ValueError(f"{names[0]} is not mapped to any shared group.")
        group_key = self._make_shared_group_cache_key(group_id)

        if self.use_local_cache and group_key in self.factorized_layers_cache:
            fact_group = self.factorized_layers_cache[group_key]
        else:
            storage_rank = self._get_shared_group_storage_rank(matrices)
            fact_group = self._factorize_shared_group(
                matrices=matrices, names=names,
                eq_rank=storage_rank, rank=storage_rank,
                dev=self.dev, verbose=verbose,
            )
            if self.use_local_cache:
                self.factorized_layers_cache[group_key] = fact_group

        if rank > fact_group.eq_rank:
            raise ValueError(
                f"Requested rank {rank} exceeds cached eq_rank {fact_group.eq_rank} for group {group_id}."
            )
        fact_group.active_rank = rank
        return fact_group

    def _factorize_shared_group(self, matrices, names, eq_rank, rank, dev, verbose=True):
        raise NotImplementedError("Subclasses should implement shared-group factorization.")

    # ------------------------------------------------------------------
    #  Block-chunking helper
    # ------------------------------------------------------------------

    def _get_block_chunk_size(self) -> int:
        if not self.one_shot_factorization:
            return max(1, int(self.shared_group_size)) if self.use_shared_basis else 1
        return 1

    # ------------------------------------------------------------------
    #  Override: factorization_computations
    # ------------------------------------------------------------------

    def factorization_computations(self, model, name_omit, calib_data, mixup_fn, white_list=[]):
        self._build_shared_groups(model, name_omit)
        super().factorization_computations(model, name_omit, calib_data, mixup_fn, white_list=white_list)

    # ------------------------------------------------------------------
    #  Override: block-wise factorization (chunk support)
    # ------------------------------------------------------------------

    def _get_scale_and_factorize_block_wise(self, model, name_omit, calib_data, mixup_fn, white_list):
        layers, mod_name = _find_decoder_layers(model)
        num_layers = len(layers)
        block_chunk_size = self._get_block_chunk_size()

        for chunk_start in range(0, num_layers, block_chunk_size):
            chunk_end = min(chunk_start + block_chunk_size, num_layers)

            chunk_named_modules = []
            for global_block_idx in range(chunk_start, chunk_end):
                block = layers[global_block_idx]
                for local_name, module_sub in get_valid_layers(block, name_omit, white_list=white_list):
                    full_name = f"{mod_name}.{global_block_idx}.{local_name}"
                    chunk_named_modules.append((full_name, module_sub))

            hook_subset = NamedModuleSubset(chunk_named_modules)

            self._get_scale_and_factorize_module(
                model=model, hook_module=hook_subset,
                calib_data=calib_data, name_omit=name_omit,
                name_prefix="", white_list=white_list, mixup_fn=mixup_fn,
                tqdm_message=f"Blocks {chunk_start+1}-{chunk_end}/{num_layers}: Gathering ",
            )

            if self.progressive_compression:
                self._progressively_compress_chunk(chunk_named_modules)

            if self.compute_memory_efficient:
                for full_name, _ in chunk_named_modules:
                    self._factorize_cleanup(full_name)
                torch.cuda.empty_cache()
                gc.collect()

    # ------------------------------------------------------------------
    #  Override: per-module factorization (+ shared-group precompute)
    # ------------------------------------------------------------------

    def _get_scale_and_factorize_module(
        self, model, hook_module, name_prefix, calib_data,
        name_omit, mixup_fn=None, white_list=[], tqdm_message="Gathering ",
    ):
        self._compute_scaling(
            model=model, hook_module=hook_module, name_prefix=name_prefix,
            calib_data=calib_data, name_omit=name_omit, mixup_fn=mixup_fn,
            white_list=white_list, tqdm_message=tqdm_message + "scalings...",
        )
        torch.cuda.empty_cache()

        if not self.use_local_cache:
            return

        copied_modules = get_valid_layers(hook_module, name_omit, white_list=white_list)
        full_named_modules = [
            (f"{name_prefix}{name}", module_sub) for name, module_sub in copied_modules
        ]

        available_named_modules = {}
        for full_name, module_sub in tqdm(full_named_modules, desc=tqdm_message + "factorizations..."):
            available_named_modules[full_name] = module_sub
            rank, ratio, cntinue = self._get_active_rank(module_sub.weight.shape, full_name)
            if cntinue:
                continue
            det_weight = module_sub.weight.detach().clone()
            _ = self.factorize_matrix(matrix=det_weight, rank=rank, ratio=ratio, name=full_name, verbose=False)
            del det_weight
            torch.cuda.empty_cache()

        if self.use_shared_basis:
            self._precompute_shared_groups_for_available_modules(available_named_modules)

        torch.cuda.empty_cache()
        gc.collect()

    def _precompute_shared_groups_for_available_modules(self, available_named_modules, verbose=False):
        if not self.use_shared_basis:
            return
        for group_id, names in self.shared_groups.items():
            group_key = self._make_shared_group_cache_key(group_id)
            if group_key in self.factorized_layers_cache:
                continue
            if not all(n in available_named_modules for n in names):
                continue
            matrices = [available_named_modules[n].weight.detach().clone() for n in names]
            storage_rank = self._get_shared_group_storage_rank(matrices)
            fact_group = self._factorize_shared_group(
                matrices=matrices, names=names,
                eq_rank=storage_rank, rank=storage_rank,
                dev=self.dev, verbose=verbose,
            )
            self.factorized_layers_cache[group_key] = fact_group
            for mat in matrices:
                del mat
            del matrices
            torch.cuda.empty_cache()
        gc.collect()

    # ------------------------------------------------------------------
    #  Progressive compression (shared-group aware)
    # ------------------------------------------------------------------

    def _progressively_compress_chunk(self, chunk_named_modules):
        module_dict = {name: mod for name, mod in chunk_named_modules}
        handled = set()

        # 1) Shared groups first
        if self.use_shared_basis:
            for group_id, member_names in self.shared_groups.items():
                present = [n for n in member_names if n in module_dict]
                if len(present) < 2:
                    continue
                group_rank_dict = {}
                for n in present:
                    group_rank_dict[n] = (
                        self.calibration_ranks[n]
                        if n in self.calibration_ranks
                        else float(self.static_progressive_compression_ratio)
                    )
                target_rank = self._resolve_shared_group_rank(present, module_dict, group_rank_dict)
                eq_cap = self._get_shared_group_storage_rank_from_shapes(
                    [tuple(module_dict[n].weight.shape) for n in present]
                )
                if target_rank >= eq_cap:
                    handled.update(present)
                    continue
                matrices = [module_dict[n].weight.detach().clone() for n in present]
                fact_group = self.factorize_shared_group(matrices=matrices, names=present, rank=target_rank, verbose=False)
                shared_r = fact_group.right_factor.to(self.dev)
                for idx, n in enumerate(present):
                    mat_l = fact_group.left_factor(idx).to(self.dev)
                    module_dict[n].weight.data.copy_(mat_l @ shared_r)
                    handled.add(n)
                for mat in matrices:
                    del mat
                del matrices
                torch.cuda.empty_cache()

        # 2) Remaining layers individually
        for name, module_sub in chunk_named_modules:
            if name in handled:
                continue
            target_rank = self._resolve_requested_rank(
                shape=module_sub.weight.shape, key=name,
                default_ratio=self.static_progressive_compression_ratio,
            )
            if target_rank is None:
                continue
            eq_rank = get_eq_rank(*module_sub.weight.shape)
            if target_rank >= eq_rank:
                continue
            factorized_matrix = self.factorize_matrix(
                matrix=module_sub.weight, name=name, rank=target_rank, ratio=-1, verbose=False,
            )
            module_sub.weight.data.copy_(
                factorized_matrix.mat_l.to(self.dev) @ factorized_matrix.mat_r.to(self.dev)
            )
        torch.cuda.empty_cache()
        gc.collect()

    # ------------------------------------------------------------------
    #  Override: factorize_model (shared-group aware)
    # ------------------------------------------------------------------

    def factorize_model(self, uncom_model, rank_dict, name_omit, verbose=True, apply_fact=True) -> dict:
        print("\nApplying factorization")
        dev = torch.device(torch.cuda.current_device())
        with torch.cuda.device(dev):
            torch.cuda.empty_cache()

        model = uncom_model.eval().cpu()
        self._build_shared_groups(model, name_omit)

        copied_modules = get_valid_layers(model, name_omit, white_list=[])
        module_dict = {name: mod for name, mod in copied_modules}

        print(rank_dict) if verbose else None
        plot_compression_rates(rank_dict) if verbose else None

        handled = set()

        def set_module_by_name(root_module, full_name, new_module):
            base, localname = root_module, full_name
            while "." in localname:
                prefix, localname = localname.split(".", 1)
                base = getattr(base, prefix)
            setattr(base, localname, new_module)

        # 1) Shared-basis groups
        if self.use_shared_basis:
            for group_id, member_names in self.shared_groups.items():
                present = [n for n in member_names if n in module_dict and n in rank_dict]
                if len(present) < 2:
                    continue

                target_rank = self._resolve_shared_group_rank(present, module_dict, rank_dict)
                eq_cap = self._get_shared_group_storage_rank_from_shapes(
                    [tuple(module_dict[n].weight.shape) for n in present]
                )
                if target_rank >= eq_cap:
                    handled.update(present)
                    continue

                matrices = [module_dict[n].weight.detach().clone() for n in present]
                fact_group = self.factorize_shared_group(matrices=matrices, names=present, rank=target_rank, verbose=verbose)

                if not apply_fact:
                    handled.update(present)
                    continue

                first_mod = module_dict[present[0]]
                dtype = first_mod.weight.dtype
                shared_dev = first_mod.weight.device

                shared_module_l = nn.Linear(first_mod.in_features, fact_group.active_rank, bias=False, dtype=dtype).to(shared_dev)
                shared_module_l.weight.data.copy_(fact_group.right_factor.to(shared_dev))

                for idx, name in enumerate(present):
                    original_module = module_dict[name]
                    module_r = nn.Linear(fact_group.active_rank, original_module.out_features, bias=False, dtype=dtype).to(shared_dev)
                    module_r.weight.data.copy_(fact_group.left_factor(idx).to(shared_dev))
                    replacement = SeqSVD(shared_module_l, module_r, original_module.bias if hasattr(original_module, "bias") else None).to(shared_dev)
                    set_module_by_name(model, name, replacement)
                    handled.add(name)
                    print(f"Applying shared low rank on {name:^10}, rank {rank_dict[name]}, group {group_id}") if verbose else None

                for mat in matrices:
                    del mat
                del matrices
                torch.cuda.empty_cache()
                gc.collect()

        # 2) Per-layer SVD for remaining layers
        for name, module_sub in tqdm(copied_modules):
            if name in handled or name not in rank_dict:
                continue
            if rank_dict[name] == -1 or rank_dict[name] == 1.0:
                continue

            factorized = self.factorize_matrix(
                matrix=module_sub.weight,
                rank=rank_dict[name] if isinstance(rank_dict[name], int) else -1,
                ratio=rank_dict[name] if isinstance(rank_dict[name], float) else -1,
                name=name, verbose=verbose,
            )
            if factorized is None:
                continue
            if not apply_fact:
                continue

            svd_replacement = self.create_factorized_sequential(factorized_matrix=factorized, original_module=module_sub)
            print(f"Applying low rank on {name:^10}, rank {rank_dict[name]}") if verbose else None
            set_module_by_name(model, name, svd_replacement)

            with torch.cuda.device(dev):
                torch.cuda.empty_cache()
            gc.collect()

        unique_num_params = self.get_unique_model_num_parameters(model, verbose=False)
        print(f"\nNumber of UNIQUE parameters in factorized model: {unique_num_params}")

    # ------------------------------------------------------------------
    #  compress_module_dict (used by grid_search / optuna)
    # ------------------------------------------------------------------

    def compress_module_dict(self, module_dict, module_bkup_dict, rank_dict):
        dev = torch.device(torch.cuda.current_device())
        handled = set()

        for group_id, member_names in self.shared_groups.items():
            present = [n for n in member_names if n in module_dict and n in module_bkup_dict and n in rank_dict]
            if len(present) < 2:
                continue
            rank = self._resolve_shared_group_rank(present, module_bkup_dict, rank_dict)
            eq_cap = self._get_shared_group_storage_rank_from_shapes(
                [tuple(module_bkup_dict[n].weight.shape) for n in present]
            )
            if rank >= eq_cap:
                for n in present:
                    module_dict[n].weight.data.copy_(module_bkup_dict[n].weight.data)
                    handled.add(n)
                continue

            matrices = [module_bkup_dict[n].weight.data for n in present]
            fact_group = self.factorize_shared_group(matrices=matrices, names=present, rank=rank, verbose=False)
            shared_r = fact_group.right_factor.to(dev)
            for idx, n in enumerate(present):
                mat_l = fact_group.left_factor(idx).to(dev)
                module_dict[n].weight.data.copy_(mat_l @ shared_r)
                handled.add(n)

        for name, module_sub in module_dict.items():
            if name in handled:
                continue
            if rank_dict[name] == 1.0:
                module_sub.weight.data.copy_(module_bkup_dict[name].weight.data)
            else:
                factorized_matrix = self.factorize_matrix(
                    module_bkup_dict[name].weight, ratio=rank_dict[name], name=name, verbose=False,
                )
                tensor_to_copy = factorized_matrix.mat_l.to(dev) @ factorized_matrix.mat_r.to(dev)
                module_sub.weight.data.copy_(tensor_to_copy)
