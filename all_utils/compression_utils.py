"""
Shared utilities for the compression pipeline.

This module hosts functions and classes that are used across both the
``factorization`` and ``search`` sub-packages — model inspection helpers,
pure-math utilities, and generic hook classes.

``get_valid_layers`` is the filter-aware layer iterator used throughout the
compression pipeline.  For a simpler recursive variant that finds all
Linear/Conv2d modules, see ``llm_utils.model_utils.find_layers``.
"""

import torch
from torch import nn
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
#  Pure-math helpers
# ---------------------------------------------------------------------------


def get_eq_rank(n, m):
    """Equivalent rank: *n·m / (n + m)*."""
    return int(n * m / (n + m))


# ---------------------------------------------------------------------------
#  Model inspection
# ---------------------------------------------------------------------------


def is_linear_like_conv(layer):
    """Return True for 1×1 group-1 Conv2d with ≥10 output channels."""
    return (
        isinstance(layer, nn.Conv2d)
        and layer.kernel_size == (1, 1)
        and layer.groups == 1
        and layer.out_channels >= 10
    )


def get_valid_layers(model: nn.Module, name_omit, white_list=[]):
    """Return ``(name, module)`` tuples for compressible Linear layers.

    Filters by *name_omit* (exclusion) and *white_list* (inclusion) patterns
    and requires ``out_features >= 10``.
    """
    valid_layers = [
        (name, module_sub)
        for name, module_sub in model.named_modules()
        if all(omit not in name for omit in name_omit)
        and (not white_list or any(n in name for n in white_list))
        if isinstance(module_sub, nn.Linear) and module_sub.out_features >= 10
    ]
    return valid_layers


def _find_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """
    Dynamically finds the module list containing the decoder layers of a transformer model.
    This is a heuristic-based approach that should work for most modern LLMs.
    """
    for name, module in model.named_modules():
        # The decoder layers are usually in a ModuleList.
        if isinstance(module, nn.ModuleList):
            # Check if the children of this ModuleList are the decoder blocks.
            # A common heuristic is that decoder blocks have 'self_attn' and 'mlp' attributes.
            if (
                len(module) > 0 and
                (
                    hasattr(module[0], 'self_attn') or
                    hasattr(module[0], 'mixer')
                )
            ):
                print(f"Found decoder layers at path: {name}")
                return module, name

    banned = ("qkv", "mlp", "attn", "fc1", "fc2", "patch_embed")
    banned = tuple(s.lower() for s in banned)

    def is_banned(name: str) -> bool:
        lname = name.lower()
        return any(b in lname for b in banned)

    best_path, best_mod, best_depth = None, None, -1
    stack = [([], model)]  # DFS stack of (path, module)

    while stack:
        path, mod = stack.pop()

        # Candidate: this module itself is a Sequential and its own name isn't banned
        own_name = path[-1] if path else ""  # name within parent
        if isinstance(mod, nn.Sequential) and not is_banned(own_name) and len(mod) > 2:
            depth = len(path)
            if depth > best_depth:
                best_path, best_mod, best_depth = path, mod, depth

        # Recurse into children unless the child's *name* is banned
        for name, child in mod.named_children():
            if not is_banned(name):
                stack.append((path + [name], child))
    if best_mod is not None:
        print(f"Found decoder layers at path: {'.'.join(best_path + [best_mod._get_name()])} with depth {best_depth}")
        return best_mod, ".".join(best_path + [best_mod._get_name()])
    raise ValueError("Could not find any decoder layers module list or nn.Sequential in the model.")


# ---------------------------------------------------------------------------
#  Safe whitened SVD
# ---------------------------------------------------------------------------


def safe_whitened_svd(matrix, row_scale_diag, row_scale_diag_inv,
                      column_scale_diag, column_scale_diag_inv,
                      rank, name):
    """SVD of a whitened weight matrix with non-finite fallback.

    Computes ``row_scale_diag @ matrix @ column_scale_diag``, runs SVD, and
    distributes ``sqrt(S)`` symmetrically between the left and right factors
    while undoing the whitening transform.

    If the SVD fails (typically because a scaling matrix contains non-finite
    values), the offending scaling is replaced by the identity and the SVD is
    retried.

    Returns
    -------
    mat_l : Tensor   – ``row_scale_diag_inv @ U @ diag(sqrt(s))``  truncated to *rank*
    mat_r : Tensor   – ``diag(sqrt(s)) @ Vh @ column_scale_diag_inv``  truncated to *rank*
    s     : Tensor   – full singular-value vector
    """
    temp_dtype = row_scale_diag.dtype
    dev = row_scale_diag.device
    mat_scaled = row_scale_diag @ matrix.to(dev).to(temp_dtype) @ column_scale_diag

    try:
        u, s, vh = torch.linalg.svd(mat_scaled, full_matrices=False)
    except Exception:
        if not torch.all(torch.isfinite(row_scale_diag)) or not torch.all(torch.isfinite(row_scale_diag_inv)):
            print(f"⚠️  Warning: Row scaling for layer {name} is non-finite. Replacing with identity.")
            row_scale_diag = torch.eye(row_scale_diag.shape[0], device=dev, dtype=temp_dtype)
            row_scale_diag_inv = torch.eye(row_scale_diag_inv.shape[0], device=dev, dtype=temp_dtype)
        if not torch.all(torch.isfinite(column_scale_diag)) or not torch.all(torch.isfinite(column_scale_diag_inv)):
            print(f"⚠️  Warning: Column scaling for layer {name} is non-finite. Replacing with identity.")
            column_scale_diag = torch.eye(column_scale_diag.shape[0], device=dev, dtype=temp_dtype)
            column_scale_diag_inv = torch.eye(column_scale_diag_inv.shape[0], device=dev, dtype=temp_dtype)
        mat_scaled = row_scale_diag @ matrix.to(dev).to(temp_dtype) @ column_scale_diag
        u, s, vh = torch.linalg.svd(mat_scaled, full_matrices=False)

    s_val = torch.sqrt(s)
    mat_l = row_scale_diag_inv @ (u * s_val.unsqueeze(0))[:, :rank]
    mat_r = (s_val.unsqueeze(1) * torch.matmul(vh, column_scale_diag_inv))[:rank, :]
    return mat_l, mat_r, s


# ---------------------------------------------------------------------------
#  Visualization
# ---------------------------------------------------------------------------


def plot_compression_rates(rank_dict):
    """Bar-chart of per-layer compression ratios."""
    layer_names = list(rank_dict.keys())
    compression_rates = [rate for name, rate in rank_dict.items()]
    if not isinstance(compression_rates[0], float):
        return

    plt.figure(figsize=(10, 6))
    plt.bar(layer_names, compression_rates, color='skyblue')
    plt.xlabel('Layer Names')
    plt.ylabel('Compression Rate')
    plt.title('Compression Rates by Layer')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    plt.savefig('compression_rates.png', dpi=300)
