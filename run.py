"""Hydra-based entry point for KFAC-SVD model compression.

Usage examples::

    # KFAC-SVD + LEMS on LLaMA-3-8B at 80 % parameter ratio
    python run.py model=llama3_8b compression_target=0.8

    # Named preset (provides model + method + tuned defaults)
    python run.py +preset=kfac_lems model=qwen3_8b compression_target=0.9

    # Override individual fields
    python run.py model=mistral_7b compression_target=0.5 svd=svd_llm search=uniform

    # Extended evaluation + HuggingFace export
    python run.py model=llama3_8b compression_target=0.8 eval=extended export=local
"""

# ── Bootstrap ────────────────────────────────────────────────────────
import os
if "CUDA_DEVICE" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_DEVICE"]
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from all_utils.circumvent_limiter import patch_package
patch_package()
import local_datasets.math_qa.register_mathqa_local  # noqa: F401

# ── Imports ──────────────────────────────────────────────────────────
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from compression.svd_core import ModelFactorizer
from all_utils.data_utils import get_calib_train_data
from all_utils.model_utils import get_model_from_huggingface
from all_utils.evaluater import ppl_eval, zero_shot_eval, generate_sample
from all_utils.excel_tracking import build_excel_dict, check_and_create_excel


# ── Helpers ──────────────────────────────────────────────────────────

def enforce_strict_determinism(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        import cupy as cp
        cp.random.seed(seed)
    except ImportError:
        pass
    torch.use_deterministic_algorithms(True, warn_only=False)

# ── Main ─────────────────────────────────────────────────────────────

@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    device = torch.device(cfg.device)
    enforce_strict_determinism(cfg.seed)

    # ── Load model ───────────────────────────────────────────────────
    print(f"Creating model: {cfg.model.name}")
    model, tokenizer = get_model_from_huggingface(
        model_id=cfg.model.name,
        seq_len=cfg.data.seq_len,
        grad_ckpt=cfg.model.gradient_ckpt,
        fp32=cfg.model.fp32,
        cache_dir=os.path.join(cfg.output_dir, "huggingface/llm") if not hasattr(cfg, "huggingface_cache_dir") else cfg.huggingface_cache_dir,
    )

    # ── Build method arg dicts directly from config ──────────────────
    # SVD / factorization args — keys match BaseFactorization.__init__
    svd_args = OmegaConf.to_container(cfg.svd, resolve=True)
    svd_method = svd_args.pop("method")
    svd_args["vision"] = False  # LLM pipeline — always False
    svd_args["workspace_dir"] = cfg.output_dir

    # Search args — keys match search class constructors
    search_args = OmegaConf.to_container(cfg.search, resolve=True)
    search_method_name = search_args.pop("method")
    # Inject fields that live outside the search config group
    search_args["ratio_target"] = cfg.compression_target
    search_args["sequence_length"] = cfg.data.seq_len
    search_args["use_cache"] = cfg.svd.use_cache
    search_args["workspace_dir"] = cfg.output_dir

    # ── Descriptive run directory ─────────────────────────────────────
    def _sanitize(s: str) -> str:
        return str(s).replace("/", "_").replace("-", "_").replace(".", "_")

    run_name = (
        f"{_sanitize(cfg.model.name)}"
        f"_{cfg.data.calib_dataset}_{cfg.seed}_{cfg.svd.do_post_calibration}"
        f"_{svd_method}_{search_method_name}_{cfg.compression_target}"
    )
    run_dir = os.path.join(cfg.output_dir, "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Persist the fully-resolved config alongside the run artifacts
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # ── Calibration & evaluation data ────────────────────────────────
    model = model.eval()
    calib_data = get_calib_train_data(
        cfg.data.calib_dataset, tokenizer,
        nsamples=cfg.data.calib_bs, seqlen=cfg.data.seq_len,
        seed=cfg.seed, mode=cfg.data.calib_data_mode,
        output_dir=os.path.join(cfg.output_dir, "cache", "datasets"),
    )
    eval_data = get_calib_train_data(
        cfg.data.calib_dataset, tokenizer,
        nsamples=cfg.data.search_samples, seqlen=cfg.data.seq_len,
        seed=cfg.seed + 1000, batch_size=1,
        mode=cfg.data.calib_data_mode,
    )

    # ── Checkpoint restore (optional) ────────────────────────────────
    chkp_path = os.path.join(run_dir, "checkpoint.pt")
    checkpoint_restored = False

    if cfg.re_model and os.path.exists(chkp_path):
        print(f"Loading model from checkpoint: {chkp_path}")
        checkpoint = torch.load(chkp_path, map_location="cpu")
        model = checkpoint["model"]
        checkpoint_restored = True

    # ── Compression ──────────────────────────────────────────────────
    time_taken = 0.0
    rank_dict = {}

    if svd_method != "baseline" and not checkpoint_restored:
        factorizor = ModelFactorizer(
            svd_method=svd_method,
            svd_method_args=svd_args,
            search_method=search_method_name,
            search_method_args=search_args,
        )
        calib_dataset_name = cfg.data.calib_dataset
        if cfg.data.calib_data_mode != "v1":
            calib_dataset_name += f"_{cfg.data.calib_data_mode}"

        time_taken, search_time, num_params, model, rank_dict = (
            factorizor.factorize_and_search(
                model=model,
                calib_data=calib_data,
                eval_data=eval_data,
                calib_dataset_name=calib_dataset_name,
                mixup_fn=None,
                name_omit=list(cfg.name_omit),
                blockwise_search=cfg.svd.blockwise_factorization,
            )
        )
        torch.save({"model": model, "tokenizer": tokenizer}, chkp_path)
    else:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Baseline model has {num_params / 1e6:.2f}M parameters.")

    # ── Evaluation ───────────────────────────────────────────────────
    model = model.to(device).eval()
    print(model)
    extended_eval_results = {}

    torch.use_deterministic_algorithms(True, warn_only=True)

    if cfg.eval.ppl_datasets:
        ppls = ppl_eval(
            model, tokenizer, datasets=list(cfg.eval.ppl_datasets),
            model_seq_len=cfg.data.seq_len, batch_size=1, device=device,
        )
        ppl_summary = " | ".join(f"{k}: {v}" for k, v in ppls.items())
        print(f"PPL — {ppl_summary}")

    if cfg.eval.tasks:
        extended_eval_results = zero_shot_eval(
            model, tokenizer, device=device, tasks=list(cfg.eval.tasks),
        )
        print(extended_eval_results)

        extended_eval_results["answer"] = generate_sample(model, tokenizer, device=str(device))

    # ── Excel tracking (optional) ────────────────────────────────────
    if cfg.excel_tracking:
        data_dict = build_excel_dict(cfg, time_taken, ppls, num_params, extended_eval_results)
        data_dict.update(extended_eval_results)
        data_dict["chkp_path"] = chkp_path if svd_method != "baseline" else "N/A"
        check_and_create_excel(data_dict, file_path=cfg.excel_tracking)

    # ── HuggingFace export (optional) ────────────────────────────────
    if cfg.export.save_dir:
        from all_utils.hf_integration import save_compressed
        save_compressed(
            model=model, tokenizer=tokenizer, save_dir=cfg.export.save_dir,
            base_model_id=cfg.model.name, compression_method=svd_method,
            target_ratio=cfg.compression_target,
            push_to_hub=cfg.export.push_to_hub,
            hub_repo_id=cfg.export.hub_repo_id, ppl_results=ppls,
        )


if __name__ == "__main__":
    main()
