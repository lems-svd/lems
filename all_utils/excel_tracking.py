import os
import pandas as pd
import numpy as np


def check_and_create_excel(data_dict, file_path="automated_test_results.xlsx"):
    if not os.path.exists(file_path):
        df = pd.DataFrame(columns=list(data_dict.keys()))
        df.to_excel(file_path, index=False)
        print(f"Created new Excel file: {file_path}")
    df = pd.read_excel(file_path)
    df = pd.concat([df, pd.DataFrame([data_dict])], ignore_index=True)
    df.to_excel(file_path, index=False)
    print(f"Added new row to existing Excel file: {file_path}")


def build_excel_dict(cfg, time_taken, ppls, num_params, extended_eval_results):
    """Build a flat dict suitable for one Excel row from a Hydra *cfg*."""
    method = cfg.search.method
    is_lems = method in ("lems", "lems_shared")
    return {
        "model": cfg.model.name,
        "svd_method": cfg.svd.method,
        "search_method": method,
        "sensitivity_loss": cfg.search.sensitivity_loss if hasattr(cfg.search, "sensitivity_loss") else "N/A",
        "crosslayer_term": cfg.search.crosslayer_term if hasattr(cfg.search, "crosslayer_term") else "N/A",
        "fp32": str(cfg.model.fp32),
        "compression_target": cfg.compression_target,
        "calib_bs": cfg.data.calib_bs,
        "seq_len": cfg.data.seq_len,
        "seed": cfg.seed,
        "time": str(np.datetime64("now")),
        "time_taken": time_taken,
        "wikitext_ppl": ppls.get("wikitext2", "N/A"),
        "ptb_ppl": ppls.get("ptb", "N/A"),
        "c4_ppl": ppls.get("c4", "N/A"),
        "calib_set": cfg.data.calib_dataset,
        "num_params": num_params,
        "halpha": cfg.search.halpha if hasattr(cfg.search, "halpha") else "N/A",
        "hgamma": cfg.search.hgamma if hasattr(cfg.search, "hgamma") else "N/A",
        "measurement_points": str(cfg.search.measurements_points) if hasattr(cfg.search, "measurements_points") else "N/A",
        "rank_multiple": cfg.search.enforce_rank_multiples_of if hasattr(cfg.search, "enforce_rank_multiples_of") else "N/A",
        "whitening_method": cfg.svd.whitening_method if hasattr(cfg.svd, "whitening_method") else "N/A",
        **{f"{t}_acc": extended_eval_results.get(t, "N/A") for t in (
            "boolq", "piqa", "openbookqa", "hellaswag",
            "arc_challenge", "arc_easy", "winogrande", "mathqa",
        )},
    }