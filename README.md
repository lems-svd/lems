<p align="center">
<img src="figures/logo.png" width="20%"> <br>
</p>

<div align="center">

# Advancing SVD-based LLM Compression via Layer-Wise Error Model Search
</div>

This is the official repository for the paper **[Advancing SVD-based LLM Compression via Layer-Wise Error Model Search](https://openreview.net/forum?id=IjIgNPFuCt)**.
*Moritz Thoma, Maximilian Groezinger, Maximilian Forstenhäusler, Emad Aghajanzadeh, Manoj-Rohit Vemparala, Christos Anagnostopoulos, Pierpaolo Mori, Nael Fasfous, Alexander Frickenstein, Daniel Mueller-Gritschneder, Ulf Schlichtmann*. **ICML 2026**


## 📖 Overview

Low-rank SVD-based compression offers a powerful strategy to reduce the computational costs of Large Language Models (LLMs). However, existing methods commonly encounter two recurring obstacles: (i) global rank allocation, where uncalibrated error proxies fail to account for complex error propagation, and (ii) decomposition quality, where Fisher-based estimators suffer from severe rank collapse.

In this work, we address these limitations by presenting **Layer-wise Error Modeling Search (LEMS)** and **KFAC-SVD**.

* **LEMS** advances rank allocation by introducing a layer-wise error surrogate that integrates both local and global layer importance alongside a propagation bias, allowing us to determine global rank configurations efficiently as an Integer Linear Program (ILP).
* **KFAC-SVD** improves decomposition quality by utilizing token-wise statistics, preventing the rank deficiency observed in prior Fisher-based SVD.

We demonstrate across Mistral, Qwen3, and Llama-3 families that KFAC-SVD achieves average perplexity improvements of 15%, while LEMS consistently outperforms existing search strategies, delivering significant zero-shot accuracy improvements of up to 4.7 p.p. that generalize to scales of 70B parameters.


## 🚀 Quick Start

### 1. Evaluate Existing Models

You can directly download and test our pre-compressed models from Hugging Face without running the compression pipeline locally.
<details>
  <summary><b>Models available on HuggingFace</b></summary>

| Model | Search Type | Ratio | Hugging Face Name |
| --- | --- | --- | --- |
| **Llama-3-8B** | LEMS | 0.9 | `MoritzMo123/kfac-svd_lems_llama-3-8b_0.9` |
|  | LEMS | 0.8 | `MoritzMo123/kfac-svd_lems_llama-3-8b_0.8` |
|  | LEMS | 0.7 | `MoritzMo123/kfac-svd_lems_llama-3-8b_0.7` |
|  | LEMS | 0.6 | `MoritzMo123/kfac-svd_lems_llama-3-8b_0.6` |
|  | Uniform | 0.9 | `MoritzMo123/kfac-svd_uniform_llama-3-8b_0.9` |
|  | Uniform | 0.8 | `MoritzMo123/kfac-svd_uniform_llama-3-8b_0.8` |
|  | Uniform | 0.7 | `MoritzMo123/kfac-svd_uniform_llama-3-8b_0.7` |
|  | Uniform | 0.6 | `MoritzMo123/kfac-svd_uniform_llama-3-8b_0.6` |
| **Mistral-7B** | LEMS | 0.9 | `MoritzMo123/kfac-svd_lems_mistral-7b_0.9` |
|  | LEMS | 0.8 | `MoritzMo123/kfac-svd_lems_mistral-7b_0.8` |
|  | LEMS | 0.7 | `MoritzMo123/kfac-svd_lems_mistral-7b_0.7` |
|  | LEMS | 0.6 | `MoritzMo123/kfac-svd_lems_mistral-7b_0.6` |
|  | Uniform | 0.9 | `MoritzMo123/kfac-svd_uniform_mistral-7b_0.9` |
|  | Uniform | 0.8 | `MoritzMo123/kfac-svd_uniform_mistral-7b_0.8` |
|  | Uniform | 0.7 | `MoritzMo123/kfac-svd_uniform_mistral-7b_0.7` |
|  | Uniform | 0.6 | `MoritzMo123/kfac-svd_uniform_mistral-7b_0.6` |
| **Qwen3-8B** | LEMS | 0.9 | `MoritzMo123/kfac-svd_lems_Qwen3-8B_0.9` |
|  | LEMS | 0.8 | `MoritzMo123/kfac-svd_lems_Qwen3-8B_0.8` |
|  | LEMS | 0.7 | `MoritzMo123/kfac-svd_lems_Qwen3-8B_0.7` |
|  | LEMS | 0.6 | `MoritzMo123/kfac-svd_lems_Qwen3-8B_0.6` |
|  | Uniform | 0.9 | `MoritzMo123/kfac-svd_uniform_Qwen3-8B_0.9` |
|  | Uniform | 0.8 | `MoritzMo123/kfac-svd_uniform_Qwen3-8B_0.8` |
|  | Uniform | 0.7 | `MoritzMo123/kfac-svd_uniform_Qwen3-8B_0.7` |
|  | Uniform | 0.6 | `MoritzMo123/kfac-svd_uniform_Qwen3-8B_0.6` |
</details>

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model_string="MoritzMo123/kfac-svd_lems_llama-3-8b_0.8"
model = AutoModelForCausalLM.from_pretrained(
    model_string,
    trust_remote_code=True,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_string)
```

### 2. Run Compression Yourself

To compress a model from scratch and recreate our core results, use the following template (replace `mistral_7b` with `llama3_8b` or `qwen3_8b` as needed).

> **Note:** For very large models (e.g., 70B), set the environment variable `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to avoid OOM. Use `eval=extended` for full zero-shot task sweeps or `eval=quick` for Wikitext-only evaluation.

```bash
python run.py model=mistral_7b svd=kfac_svd search=lems compression_target=0.8 eval=extended
```


## 📂 Project Structure & ⚙️ Config Settings

### Project Structure

```text
KFAC-SVD/
├── run.py                        # Main entry point (Hydra-based)
├── finetune_LLM.py               # Post-compression fine-tuning
├── configs/                      # Hydra configuration groups
│   ├── config.yaml               #   Root config (merges all groups)
│   ├── model/                    #   Model definitions (llama3_8b, mistral_7b, …)
│   ├── svd/                      #   Decomposition methods (kfac_svd, svd_llm, …)
│   │   └── _common_svd.yaml      #     Shared defaults inherited by all SVD configs
│   ├── search/                   #   Search / rank-allocation methods
│   ├── data/                     #   Calibration dataset settings
│   ├── eval/                     #   Evaluation profiles (quick / extended)
│   ├── export/                   #   HuggingFace export settings
│   └── preset/                   #   Named presets combining svd + search + tuned params
├── compression/
│   ├── svd_core.py               # Orchestrator: factorize → search → compress
│   ├── factorization/            # SVD / decomposition implementations
│   └── search/                   # Search / rank-allocation implementations
├── all_utils/                    # Shared utilities (data, evaluation, model I/O, …)
├── local_datasets/               # Local dataset loaders (C4, MathQA, …)
└── docker/                       # Dockerfile and compose files
```

### Configuration System

This project relies heavily on [Hydra](https://hydra.cc/) for configuration. The root configuration at `configs/config.yaml` manages six main groups (`model`, `svd`, `search`, `data`, `eval`, `export`).

You can override any parameter directly via the command line:

```bash
python run.py model=llama3_8b svd=kfac_svd search=lems compression_target=0.8 \
    search.enforce_rank_multiples_of=16 svd.do_post_calibration=False
```

You can also use **presets** to combine multiple configuration groups quickly:

```bash
python run.py +preset=kfac_lems model=qwen3_8b compression_target=0.9
```


## 💻 Environment Setup Process

### Docker Setup

We strongly encourage using the dockerized environment for reproduction to avoid dependency conflicts.

```bash
cd docker
docker build -t lems:torch2.2.1 .

# Mount your HF cache or dataset folder as needed
docker run --gpus 'all' -it --name="LLM_COMPRESS" -v ~/.cache/huggingface:/root/.cache/huggingface -v ./:/workspace lems:torch2.2.1
```

### C4 Dataset Evaluation

To evaluate using the C4 dataset, additional manual setup is necessary. Please refer to `local_datasets/c4/readme.md` to make the dataset available. **Note: Without this step, running with `eval=extended` will fail!**

### ILP Solver Licenses

All LEMS results in the paper were obtained using Gurobi as the ILP solver.

* **Gurobi (Requires License):** Free for educational institutions. Place your license in `gurobi_license.json` formatted as: `{"WLSACCESSID": "your id", "WLSSECRET": "your secret", "LICENSEID": your_id}`.
* **PuLP/CBC (No License Required):** We provide an experimental open-source alternative by passing `search.solver=cbc`. To simplify the problem for faster solving with CBC, increase `search.enforce_rank_multiples_of` to a higher number (e.g., 64). *Note: We do not guarantee CBC will perfectly match Gurobi results.*

## 🗜️ Compression

### Available Methods Overview

To compress a model, you can mix and match different search and decomposition (SVD) configurations. The provided code is highly flexible and works out of the box for most HuggingFace models.

### Search Methods (`search=`)

| Config | Method | Key Parameters |
| --- | --- | --- |
| `lems` | **LEMS (Ours)** | `solver` (gurobi/cbc), `crosslayer_term`, `halpha`, `hgamma`, `enforce_rank_multiples_of` |
| `uniform` | Uniform | `enforce_rank_multiples_of` |
| `asvd` | ASVD threshold | `sensitivity_loss`, `target_metric`, `min_ratio`, `max_ratio` |
| `asvd_plus` | ASVD+ (bias + Optuna) | Same as ASVD + `crosslayer_term`, `halpha`, `hgamma`, `optuna_trials` |
| `memvit` | MRCS (greedy) | `sensitivity_loss`, `target_metric`, `lower_bound`, `enforce_rank_multiples_of` |
| `memvit_plus` | MRCS+ (bias + Optuna) | Same as MRCS + `crosslayer_term`, `halpha`, `hgamma`, `optuna_trials` |
| `svd_llmv2` | SVD-LLMv2 | `sensitivity_loss` |
| `atp` | ATP | `beta` |
| `loadconfig` | Load from JSON | `layer_compression_json_path` |

### SVD Methods (`svd=`)

| Config | Method | Description |
| --- | --- | --- |
| `kfac_svd` | **KFAC-SVD (Ours)** | Token-wise Fisher-based whitened SVD |
| `svd_llm` | SVD-LLM | Cholesky-whitened SVD |
| `svd_llmv2` | SVD-LLM (SVD whitening) | SVD-based whitening variant |
| `svd_llm_large` | SVD-LLM (memory efficient) | CPU-offloaded whitening for 70B+ models |
| `svd` | Vanilla SVD | Plain truncated SVD (no activation awareness) |
| `asvd` | ASVD | Activation-scaled SVD |
| `fwsvd` | FWSVD | Fisher-weighted SVD |
| `gfwsvd` | GFWSVD | Gradient Fisher-weighted SVD |
| `dobi_svd` | DOBI-SVD | Double-sided bi-orthogonal SVD |


## ✨ Extra: Fine-Tuning

After compression, you have the option to fine-tune the compressed model to recover further performance using `finetune_LLM.py`:

```bash
python finetune_LLM.py \
    --model_path outputs/runs/<run_name>/checkpoint.pt \
    --tuning_strategy peft_lora \
    --num_epochs 1 \
    --learning_rate 2e-5
```

**Available Tuning Strategies:**

* `peft_lora`: Standard PEFT LoRA adapters.
* `tune_svd`: Unfreezes only the decomposed parameters.
* `custom_lora`: Native LoRA applied specifically to the decomposed layers.


## 📊 Evaluation & Results

For a complete breakdown of zero-shot accuracy, perplexity generalizability, and scaling laws, please refer to the main paper and our project page.

Below are the primary results comparing LEMS to baseline search and SVD methods. Lower Wiki (Perplexity) is better; Higher Acc (Accuracy) is better.

| Ratio | Search Method | **Mistral-7B** <br> Wiki / Acc | **Llama3-8B** <br> Wiki / Acc | **Qwen3-8B** <br> Wiki / Acc |
| :---: | :--- | :---: | :---: | :---: |
| **-** | **Baseline** | 5.25 / 63.95 | 6.14 / 63.34 | 9.71 / 62.03 |
| **0.8** | Uniform | 7.14 / 52.35 | 11.44 / 47.69 | 12.52 / 53.56 |
|  | ASVD | 7.20 / 47.81 | 12.92 / 45.95 | 15.70 / 47.72 |
|  | SVD-LLMv2 | 7.13 / 52.68 | 11.40 / 47.70 | 12.52 / 53.36 |
|  | MRCS | 7.11 / 51.75 | 11.99 / 46.63 | 14.52 / 52.04 |
|  | ATP | 7.14 / 52.35 | 11.07 / 51.56 | 12.52 / 53.56 |
|  | **LEMS (Ours)** | **5.98 / 57.67** | **8.16 / 55.99** | **10.38 / 59.28** |
| **0.6** | Uniform | 14.38 / 39.10 | 48.56 / 34.39 | 21.68 / 39.63 |
|  | ASVD | 16.78 / 35.56 | 75.21 / 33.98 | 29.22 / 36.09 |
|  | SVD-LLMv2 | 14.90 / 39.29 | 47.53 / 34.43 | 21.42 / 39.74 |
|  | MRCS | 14.21 / 37.44 | 67.95 / 33.41 | 38.72 / 36.42 |
|  | ATP | 13.59 / 40.24 | 25.14 / 39.95 | 19.47 / 40.20 |
|  | **LEMS (Ours)** | **10.58 / 45.60** | **17.86 / 43.09** | **15.70 / 45.51** |

| Ratio | Search Method | **Mistral-7B** <br> Wiki / Acc | **Llama3-8B** <br> Wiki / Acc | **Qwen3-8B** <br> Wiki / Acc |
| :---: | :--- | :---: | :---: | :---: |
| **-** | **Baseline** | 5.25 / 63.96 | 6.14 / 63.33 | 9.71 / 62.04 |
| **0.9** | FWSVD | 9.47 / 56.93 | 42.11 / 48.88 | 16.95 / 53.82 |
|  | ASVD | 9.14 / 57.30 | 65.09 / 47.89 | 20.36 / 51.19 |
|  | SVD-LLM | 6.46 / 56.81 | 10.14 / 52.27 | 12.52 / 55.96 |
|  | SVD-LLMv2 | 6.46 / 56.78 | 10.18 / 52.17 | 12.52 / 55.80 |
|  | DOBI-SVD | 7.11 / 55.17 | 11.22 / 52.98 | 13.46 / 55.48 |
|  | GFWSVD | 31.66 / 40.73 | 2569.75 / 33.40 | 171.49 / 46.66 |
|  | **KFAC-SVD (Ours)** | 6.22 / 56.92 | 8.84 / 54.45 | 11.51 / 57.07 |
|  | **+ LEMS (Full)** | **5.37 / 62.77** | **6.58 / 61.99** | **9.85 / 62.89** |
| **0.7** | FWSVD | 34.84 / 44.24 | 716.39 / 33.40 | 41.37 / 42.68 |
|  | ASVD | 28.16 / 46.09 | 10989.41 / 33.03 | 70.66 / 43.10 |
|  | SVD-LLM | 10.96 / 44.39 | 34.64 / 37.69 | 17.11 / 45.76 |
|  | SVD-LLMv2 | 10.96 / 44.35 | 34.98 / 37.70 | 17.15 / 45.62 |
|  | DOBI-SVD | 12.37 / 42.94 | 31.54 / 38.48 | 18.54 / 46.24 |
|  | GFWSVD | 2549.75 / 32.29 | 41798.89 / 31.77 | 21684.98 / 34.06 |
|  | **KFAC-SVD (Ours)** | 9.29 / 45.76 | 19.43 / 40.73 | 14.69 / 46.78 |
|  | **+ LEMS (Full)** | **7.42 / 51.52** | **11.07 / 49.39** | **11.81 / 53.80** |

---

## 🤝 Contribution

If our work or code helps your research, please consider citing our paper:

```bibtex
@inproceedings{thoma_advancing_2026,
  location = {Seoul, South Korea},
  title = {Advancing {SVD}-based {LLM} Compression via Layer-Wise Error Model Search},
  url = {https://openreview.net/forum?id=IjIgNPFuCt},
  abstract = {Low-rank {SVD}-based compression offers a powerful strategy to reduce the computational costs of Large language models ({LLMs}); however, existing methods commonly encounter two recurring obstacles: (i) global rank allocation, where uncalibrated error proxies fail to account for complex error propagation, and (ii) decomposition quality, where Fisher-based estimators suffer from severe rank collapse. In this work, we address these limitations by presenting Layer-wise Error Modeling Search ({LEMS}) and {KFAC}-{SVD}. {LEMS} advances rank allocation by introducing a layer-wise error surrogate that integrates both local and global layer importance alongside a propagation bias, allowing us to determine global rank configurations efficiently as an Integer Linear Program ({ILP}). Simultaneously, {KFAC}-{SVD} improves decomposition quality by utilizing token-wise statistics, preventing the rank deficiency observed in prior Fisher-based {SVD}. We demonstrate across Mistral, Qwen3, and Llama-3 families that {KFAC}-{SVD} achieves an average perplexity improvements of 15\%, while {LEMS} consistently outperforms existing search strategies, delivering significant zero-shot accuracy improvements of up to 4.7 p.p. that generalize to scales of 70B parameters. Code is made available in the Supplement.},
  eventtitle = {Forty-third International Conference on Machine Learning},
  author = {Thoma, Moritz and Groezinger, Maximilian and Forstenhäusler, Maximilian and Aghajanzadeh, Emad and Vemparala, Manoj Rohit and Anagnostopoulos, Christos and Mori, Pierpaolo and Fasfous, Nael and Frickenstein, Alexander and Mueller-Gritschneder, Daniel and Schlichtmann, Ulf},
  urldate = {2026-05-27},
  date = {2026-04-30},
  langid = {english},
}

```