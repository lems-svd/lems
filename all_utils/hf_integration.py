"""HuggingFace integration for saving and loading compressed models.

Public helpers
--------------
* :func:`save_compressed` — write a HF-compatible model directory
  (``config.json``, ``model.safetensors``, tokenizer files, custom code,
  and an auto-generated model card).
* :func:`load_compressed` — convenience wrapper around
  ``AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)``.

The two custom-code files (``configuration_low_rank.py`` and
``modeling_low_rank.py``) are copied verbatim from the ``hf_export``
sub-package into the output directory so that users who do *not* have this
repository installed can still load the model with ``trust_remote_code``.
"""

import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

from compression.factorization._interface import SeqSVD
from .hf_export.configuration_low_rank import LowRankConfig


# -----------------------------------------------------------------------
#  Internal helpers
# -----------------------------------------------------------------------

def _extract_rank_dict(model):
    """Walk *model* and return ``{name: rank}`` for every SeqSVD module."""
    rank_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, SeqSVD):
            rank_dict[name] = module.mod_a.out_features
    return rank_dict


def _count_parameters(model):
    return sum(p.numel() for p in model.parameters())


# -----------------------------------------------------------------------
#  save_compressed
# -----------------------------------------------------------------------

def save_compressed(
    model,
    tokenizer,
    save_dir: str,
    *,
    base_model_id: str = None,
    compression_method: str = None,
    target_ratio: float = None,
    push_to_hub: bool = False,
    hub_repo_id: str = None,
    private: bool = False,
    ppl_results: dict = None,
):
    """Save a compressed model in HuggingFace-compatible format.

    After calling :func:`factorize_model` (which mutates the model in-place
    with :class:`SeqSVD` modules), call this function to persist the result
    as a standard HuggingFace model directory that can later be loaded with::

        AutoModelForCausalLM.from_pretrained(save_dir, trust_remote_code=True)

    Parameters
    ----------
    model : PreTrainedModel
        The **already-compressed** model (with SeqSVD modules in place).
    tokenizer : PreTrainedTokenizer
        The tokenizer.
    save_dir : str
        Local output directory.
    base_model_id : str, optional
        Original HuggingFace model id (e.g. ``"meta-llama/Llama-3-8B"``).
    compression_method : str, optional
        Name of the factorization method (``"kfac_svd"``, ``"svd_llm"``, …).
    target_ratio : float, optional
        The target parameter-ratio used during search (e.g. ``0.8``).
    push_to_hub : bool
        Upload the directory to HuggingFace Hub after saving.
    hub_repo_id : str, optional
        Hub repo id (e.g. ``"your-name/compressed-llama"``).
        Defaults to the directory name of *save_dir*.
    private : bool
        Whether the created Hub repo should be private.
    ppl_results : dict, optional
        ``{dataset_name: perplexity}`` dict for the model card.
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. Extract rank dict from the live model ----------------------------
    rank_dict = _extract_rank_dict(model)
    if not rank_dict:
        print("Warning: no SeqSVD modules found — saving the model as-is.")

    # 2. Build LowRankConfig + auto_map -----------------------------------
    base_model_id = base_model_id or getattr(
        model.config, "_name_or_path", "unknown"
    )
    config = LowRankConfig(
        base_config=model.config.to_dict(),
        base_model_type=model.config.model_type,
        base_model_id=base_model_id,
        rank_dict=rank_dict,
        compression_method=compression_method,
        target_ratio=target_ratio,
    )
    config.auto_map = {
        "AutoConfig": "configuration_low_rank.LowRankConfig",
        "AutoModelForCausalLM": "modeling_low_rank.LowRankCausalLM",
    }
    config.save_pretrained(save_dir)

    # 3. Save state dict (safetensors, with ``wrapped_model.`` prefix) ----
    state_dict = {
        f"wrapped_model.{k}": v.cpu().contiguous()
        for k, v in model.state_dict().items()
        if "dummy_weight" not in k
    }
    metadata = {
        "format": "pt",
        "compression_method": compression_method or "",
        "base_model_id": base_model_id or "",
    }
    save_file(state_dict, os.path.join(save_dir, "model.safetensors"),
              metadata=metadata)

    # 4. Tokenizer --------------------------------------------------------
    tokenizer.save_pretrained(save_dir)

    # 5. Generation config (if the model carries one) ---------------------
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.save_pretrained(save_dir)

    # 6. Copy self-contained custom-code files for trust_remote_code ------
    export_src = Path(__file__).resolve().parent / "hf_export"
    for fname in ("configuration_low_rank.py", "modeling_low_rank.py"):
        shutil.copy2(export_src / fname, Path(save_dir) / fname)

    # 7. Model card -------------------------------------------------------
    _generate_model_card(
        save_dir=save_dir,
        base_model_id=base_model_id,
        compression_method=compression_method,
        target_ratio=target_ratio,
        rank_dict=rank_dict,
        num_params=_count_parameters(model),
        ppl_results=ppl_results,
    )

    print(f"\u2705 Compressed model saved to {save_dir}")

    # 8. Optional Hub upload ----------------------------------------------
    if push_to_hub:
        _push(save_dir, hub_repo_id, private)


# -----------------------------------------------------------------------
#  load_compressed
# -----------------------------------------------------------------------

def load_compressed(model_path, *, device_map="auto", torch_dtype=None):
    """Load a compressed model from *model_path* (local dir or Hub repo).

    Returns ``(model, tokenizer)``."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


# -----------------------------------------------------------------------
#  Hub upload
# -----------------------------------------------------------------------

def _push(save_dir, hub_repo_id, private):
    from huggingface_hub import HfApi

    api = HfApi()
    repo_id = hub_repo_id or os.path.basename(os.path.normpath(save_dir))
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_folder(folder_path=save_dir, repo_id=repo_id)
    print(f"\u2705 Pushed to https://huggingface.co/{repo_id}")


# -----------------------------------------------------------------------
#  Model card generation
# -----------------------------------------------------------------------

def _generate_model_card(
    save_dir,
    base_model_id,
    compression_method,
    target_ratio,
    rank_dict,
    num_params,
    ppl_results=None,
):
    num_compressed = len(rank_dict)

    card = f"""\
---
library_name: transformers
tags:
- low-rank
- compressed
- {compression_method or "svd"}
base_model: {base_model_id or "unknown"}
---

# Low-Rank Compressed Model

This model was compressed using **{compression_method or "low-rank factorisation"}**
from [{base_model_id}](https://huggingface.co/{base_model_id}).

## Compression Details

| Metric | Value |
|--------|-------|
| Base Model | `{base_model_id}` |
| Method | `{compression_method}` |
| Target Ratio | {target_ratio or "N/A"} |
| Compressed Layers | {num_compressed} |
| Total Parameters | {num_params:,} |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "<this-repo-id>",
    trust_remote_code=True,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("<this-repo-id>")

inputs = tokenizer("Hello, ", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
"""

    if ppl_results:
        card += "\n## Evaluation Results\n\n"
        card += "| Dataset | Perplexity |\n|---------|------------|\n"
        for dataset, ppl in ppl_results.items():
            ppl_str = f"{ppl:.2f}" if isinstance(ppl, (int, float)) else str(ppl)
            card += f"| {dataset} | {ppl_str} |\n"

    card += """
## Rank Allocation

<details>
<summary>Per-layer ranks (click to expand)</summary>

| Layer | Rank |
|-------|------|
"""
    for name, rank in sorted(rank_dict.items()):
        card += f"| `{name}` | {rank} |\n"
    card += "\n</details>\n"

    with open(os.path.join(save_dir, "README.md"), "w") as f:
        f.write(card)
