# Advancing SVD-based LLM Compression via Layer-Wise Error Model Search

This is the official repository for the paper **Advancing SVD-based LLM Compression via Layer-Wise Error Model Search**.
*Moritz Thoma, Maximilian Groezinger, Maximilian Forstenhäusler, Emad Aghajanzadeh, Manoj-Rohit Vemparala, Christos Anagnostopoulos, Pierpaolo Mori, Nael Fasfous, Alexander Frickenstein, Daniel Mueller-Gritschneder, Ulf Schlichtmann*. **ICML 2026**

<details>
  <summary>
  <font size="+1">Abstract</font>
  </summary>
Low-rank SVD-based compression offers a powerful strategy to reduce the computational costs of Large Language Models (LLMs); however, existing methods commonly encounter two recurring obstacles: (i) global rank allocation, where uncalibrated error proxies fail to account for complex error propagation, and (ii) decomposition quality, where Fisher-based estimators suffer from severe rank collapse. In this work, we address these limitations by presenting Layer-wise Error Modeling Search (LEMS) and KFAC-SVD. LEMS advances rank allocation by introducing a layer-wise error surrogate that integrates both local and global layer importance alongside a propagation bias, allowing us to determine global rank configurations efficiently as an Integer Linear Program (ILP). Simultaneously, KFAC-SVD improves decomposition quality by utilizing token-wise statistics, preventing the rank deficiency observed in prior Fisher-based SVD. We demonstrate across Mistral, Qwen3, and Llama-3 families that KFAC-SVD achieves an average perplexity improvements of 15%, while LEMS consistently outperforms existing search strategies, delivering significant zero-shot accuracy improvements of up to 4.7 p.p. that generalize to scales of 70B parameters.
</details>

### Main Results of LEMS (Mistral, Llama-3, Qwen-3)

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

*Note: Lower Wiki (Perplexity) is better. Higher Acc (Accuracy) is better.*

### SVD Method Comparison (Uniform)

We further analyze the effectiveness of our **KFAC-SVD** decomposition against other activation- and Fisher-based SVD methods. The table below compares these methods under a fixed **Uniform** compression strategy, as well as the combination of KFAC-SVD and LEMS.

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

## Reproduce Results

To compress a model, you may pick and choose different compression and search methods from the range that is offered. We provide example commands to replicate the results of the paper in the seciton below.

### Setup & Environment
#### Docker Setup
We strongly encourage using the dockerized environment for reproduction.
```bash
cd docker
docker build -t lems:torch2.2.1 .
# Mount your HF cache or dataset folder as needed
docker run --gpus 'all' -it --name="LLM_COMPRESS" -v ~/.cache/huggingface:/root/.cache/huggingface -v ./:/workspace lems:torch2.2.1
```
In this environment, you will be able to run the commands specified below. 

####  C4 Evaluation
For evaluating the C4 dataset additional steps are necessary. Please refer to ```./local_datasets/c4/readme.md``` to manually make the dataset available. **Without this step, ```eval=extended``` will fail!

#### ILP Solver Licences
All LEMS results in the paper have been obtained using gurobi as the ILP solver. However, running it will require a license ([free for educational institutions](https://www.gurobi.com/academia/academic-program-and-licenses/)). If you have a license put it in ```gurobi_license.json``` with format ```{"WLSACCESSID": "your id", "WLSSECRET": "your secret", "LICENSEID": your_id}```. 

Alternatively, you can use the **experimental PuLP/CBC implementation** that **does not require a licence** by passing ```search.solver=cbc```. In preliminary testing of our implementation it seems to yield good results as well, although we do not guarantee that it will match our results. To simplify the problem and make it easier/fast to solve with CBC, we recommend lowering the number of variables by increasing ```search.enforce_rank_multiples_of``` to high numbers like 64.

### Rerun Comparisons
To recreate the numbers reported in the tables, run the commands below. The entry point is `run.py` which uses [Hydra](https://hydra.cc/) for configuration. For more detailed insights into the algorithms refer to any of the search approaches implementations located in `./compression/search/*` or SVD approaches located in `./compression/factorization/*`. All commands provided will compress to `0.8` (search) and `0.7` (SVD) compression, to change it to `0.6` or any other rate, just change `compression_target=`. The `eval=extended` config performs the extended sweep over zero-shot tasks, but will increase overall runtime. Use `eval=quick` for fast wikitext-only evaluation.
For running very large models like 70B variants, you should set the enironment variable `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to avoid OOM.
#### Reproduce Search Comparisons
The following commands reproduce the search comparison results using the exact same decomposition setup for all approaches. **Note:** The **ARS** baseline results were obtained using a [third-party repository](https://github.com/sidhantls/adaptive-rank-selection-svd) and are therefore not included in the direct execution commands below.
<details>
<summary>Mistral-7B</summary>
<ul>
<details>
<summary>LEMS (Ours)</summary>
<ul><pre><code>python run.py model=mistral_7b svd=kfac_svd search=lems compression_target=0.8 eval=extended
</code></pre></ul>
</details>
<details>

<summary>Uniform</summary>

<ul><pre><code>python run.py model=mistral_7b svd=kfac_svd search=uniform compression_target=0.8 eval=extended
</code></pre></ul>
</details>
<details>

<summary>ASVD</summary>
<ul><pre><code>python run.py model=mistral_7b svd=kfac_svd search=asvd compression_target=0.8 eval=extended</code></pre></ul>
</details>
<details>

<summary>SVD-LLMv2</summary>
<ul><pre><code>python run.py model=mistral_7b svd=kfac_svd search=svd_llmv2 compression_target=0.8 eval=extended</code></pre></ul>
</details>
<details>

<summary>MRCS</summary>
<ul><pre><code>python run.py model=mistral_7b svd=kfac_svd search=memvit compression_target=0.8 eval=extended
</code></pre></ul>
</details>
</ul>
</details>

<details>
<summary>Llama-3-8B</summary>
<ul>
<details>
<summary>LEMS (Ours)</summary>
<ul><pre><code>python run.py model=llama3_8b svd=kfac_svd search=lems compression_target=0.8 eval=extended
</code></pre></ul>
</details>
<details>

<summary>Uniform</summary>

<ul><pre><code>python run.py model=llama3_8b svd=kfac_svd search=uniform compression_target=0.8 eval=extended
</code></pre></ul>
</details>
<details>

<summary>ASVD</summary>
<ul><pre><code>python run.py model=llama3_8b svd=kfac_svd search=asvd compression_target=0.8 eval=extended</code></pre></ul>
</details>
<details>

<summary>SVD-LLMv2</summary>
<ul><pre><code>python run.py model=llama3_8b svd=kfac_svd search=svd_llmv2 compression_target=0.8 eval=extended</code></pre></ul>
</details>
<details>

<summary>MRCS</summary>
<ul><pre><code>python run.py model=llama3_8b svd=kfac_svd search=memvit compression_target=0.8 eval=extended
</code></pre></ul>
</details>
</ul>
</details>

<details>
<summary>Qwen3-8B</summary>
<ul>
<details>
<summary>LEMS (Ours)</summary>
<ul><pre><code>python run.py model=qwen3_8b svd=kfac_svd search=lems compression_target=0.8 eval=extended
</code></pre></ul>
</details>
<details>

<summary>Uniform</summary>

<ul><pre><code>python run.py model=qwen3_8b svd=kfac_svd search=uniform compression_target=0.8 eval=extended
</code></pre></ul>
</details>
<details>

<summary>ASVD</summary>
<ul><pre><code>python run.py model=qwen3_8b svd=kfac_svd search=asvd compression_target=0.8 eval=extended</code></pre></ul>
</details>
<details>

<summary>SVD-LLMv2</summary>
<ul><pre><code>python run.py model=qwen3_8b svd=kfac_svd search=svd_llmv2 compression_target=0.8 eval=extended</code></pre></ul>
</details>
<details>

<summary>MRCS</summary>
<ul><pre><code>python run.py model=qwen3_8b svd=kfac_svd search=memvit compression_target=0.8 eval=extended
</code></pre></ul>
</details>
</ul>
</details>

#### Reproduce SVD Comparisons

The following commands reproduce the SVD comparison results. The default commands use **Uniform** search to isolate the impact of the decomposition method.

<details>
<summary>Mistral-7B</summary>
<ul>
<details>
<summary>KFAC-SVD (Ours)</summary>
<ul>
<li><strong>Uniform:</strong></li>
<pre><code>python run.py model=mistral_7b svd=kfac_svd search=uniform compression_target=0.7 eval=extended
</code></pre>

<li><strong>+ LEMS:</strong></li>
<pre><code>python run.py model=mistral_7b svd=kfac_svd search=lems compression_target=0.7 eval=extended
</code></pre></ul>

</details>
<details>
<summary>Baselines (Uniform)</summary>
<ul>

<li><strong>FWSVD:</strong></li>
<pre><code>python run.py model=mistral_7b svd=fwsvd search=uniform compression_target=0.7 eval=extended
</code></pre>

<li><strong>ASVD:</strong></li>
<pre><code>python run.py model=mistral_7b svd=asvd search=uniform compression_target=0.7 eval=extended
</code></pre>

<li><strong>SVD-LLM:</strong></li>
<pre><code>python run.py model=mistral_7b svd=svd_llm search=uniform compression_target=0.7 eval=extended
</code></pre>

<li><strong>SVD-LLMv2:</strong></li>
<pre><code>python run.py model=mistral_7b svd=svd_llmv2 search=uniform compression_target=0.7 eval=extended
</code></pre>

<li><strong>DOBI-SVD:</strong></li>
<pre><code>python run.py model=mistral_7b svd=dobi_svd search=uniform compression_target=0.7 eval=extended
</code></pre>

<li><strong>GFWSVD:</strong></li>
<pre><code>python run.py model=mistral_7b svd=gfwsvd search=uniform compression_target=0.7 eval=extended
</code></pre>
</ul>
</details>
</ul>
</details>

<details>
<summary>Llama-3-8B</summary>
<ul>
<details>
<summary>KFAC-SVD (Ours)</summary>
<ul>
<li><strong>Uniform:</strong></li>
<pre><code>python run.py model=llama3_8b svd=kfac_svd search=uniform compression_target=0.7 eval=extended
</code></pre>

<li><strong>+ LEMS:</strong></li>
<pre><code>python run.py model=llama3_8b svd=kfac_svd search=lems compression_target=0.7 eval=extended
</code></pre></ul>

</details>
<details>
<summary>Baselines (Uniform)</summary>
<ul>

<li><strong>FWSVD:</strong></li>
<pre><code>python run.py model=llama3_8b svd=fwsvd search=uniform compression_target=0.7 eval=extended
</code></pre>

<li><strong>ASVD:</strong></li>
<pre><code>python run.py model=llama3_8b svd=asvd search=uniform compression_target=0.7 eval=extended
</code></pre>

<li><strong>SVD-LLM:</strong></li>
<pre><code>python run.py model=llama3_8b svd=svd_llm search=uniform compression_target=0.7 eval=extended
</code></pre>

<li><strong>SVD-LLMv2:</strong></li>
<pre><code>python run.py model=llama3_8b svd=svd_llmv2 search=uniform compression_target=0.7 eval=extended
</code></pre>

<li><strong>DOBI-SVD:</strong></li>
<pre><code>python run.py model=llama3_8b svd=dobi_svd search=uniform compression_target=0.7 eval=extended
</code></pre>

<li><strong>GFWSVD:</strong></li>
<pre><code>python run.py model=llama3_8b svd=gfwsvd search=uniform compression_target=0.7 eval=extended
</code></pre>
</ul>
</details>
</ul>
</details>

<details>
<summary>Qwen3-8B</summary>
<ul>
<details>
<summary>KFAC-SVD (Ours)</summary>
<ul>
<li><strong>Uniform:</strong></li>
<pre><code>python run.py model=qwen3_8b svd=kfac_svd search=uniform compression_target=0.7 eval=extended
</code></pre>

<li><strong>+ LEMS:</strong></li>
<pre><code>python run.py model=qwen3_8b svd=kfac_svd search=lems compression_target=0.7 eval=extended
</code></pre></ul>

</details>
<details>
<summary>Baselines (Uniform)</summary>
<ul>

<li><strong>FWSVD:</strong></li>
<pre><code>python run.py model=qwen3_8b svd=fwsvd search=uniform compression_target=0.7 eval=extended
</code></pre>

<li><strong>ASVD:</strong></li>
<pre><code>python run.py model=qwen3_8b svd=asvd search=uniform compression_target=0.7 eval=extended
</code></pre>

<li><strong>SVD-LLM:</strong></li>
<pre><code>python run.py model=qwen3_8b svd=svd_llm search=uniform compression_target=0.7 eval=extended
</code></pre>

<li><strong>SVD-LLMv2:</strong></li>
<pre><code>python run.py model=qwen3_8b svd=svd_llmv2 search=uniform compression_target=0.7 eval=extended
</code></pre>

<li><strong>DOBI-SVD:</strong></li>
<pre><code>python run.py model=qwen3_8b svd=dobi_svd search=uniform compression_target=0.7 eval=extended
</code></pre>

<li><strong>GFWSVD:</strong></li>
<pre><code>python run.py model=qwen3_8b svd=gfwsvd search=uniform compression_target=0.7 eval=extended
</code></pre>
</ul>
</details>
</ul>
</details>

#### Export & Load Compressed Models
Export a compressed model to a HuggingFace-compatible format:
```
python run.py model=llama3_8b svd=kfac_svd search=lems compression_target=0.8 \
    export=hf export.save_path=./compressed-llama-3-8b \
    export.push_to_hub=true export.hub_repo_id=your-name/compressed-llama-3-8b
```
This way you can directly download and use our models:
```
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "your-name/compressed-llama-3-8b",
    trust_remote_code=True,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("your-name/compressed-llama-3-8b")
```

#### Execution for other Models
The provided code is highly flexible when it comes to different model architectures and should work out of the box for most HuggingFace models. You may use the commands provided above as a template for executing experiments. Note that most decomposition and search approaches are compatible with each other and can be combined freely.

---

## Project Structure

```
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

## Configuration System

The project uses [Hydra](https://hydra.cc/) for configuration management. The root config at `configs/config.yaml` merges six config groups:

| Group | Directory | Purpose |
|---|---|---|
| `model` | `configs/model/` | HuggingFace model ID, precision, gradient checkpointing |
| `svd` | `configs/svd/` | Decomposition method and its parameters |
| `search` | `configs/search/` | Rank-allocation strategy and its parameters |
| `data` | `configs/data/` | Calibration dataset, sequence length, sample counts |
| `eval` | `configs/eval/` | Evaluation tasks (`quick` = WikiText PPL only; `extended` = + zero-shot) |
| `export` | `configs/export/` | HuggingFace export settings (local save, Hub push) |

**Override any parameter** on the command line:
```bash
python run.py model=llama3_8b svd=kfac_svd search=lems compression_target=0.8 \
    search.enforce_rank_multiples_of=16 svd.do_post_calibration=False
```

**Presets** combine multiple config groups into a single shorthand:
```bash
python run.py +preset=kfac_lems model=qwen3_8b compression_target=0.9
```

All SVD configs inherit shared defaults from `configs/svd/_common_svd.yaml` (`use_cache`, `progressive_compression`, `blockwise_factorization`, `do_post_calibration`) and only specify method-specific overrides.

## Available Methods

### Decomposition Methods (`svd=`)

| Config | Method | Description |
|---|---|---|
| `kfac_svd` | **KFAC-SVD** (Ours) | Token-wise Fisher-based whitened SVD |
| `svd_llm` | SVD-LLM | Cholesky-whitened SVD |
| `svd_llmv2` | SVD-LLM (SVD whitening) | SVD-based whitening variant |
| `svd_llm_large` | SVD-LLM (memory efficient) | CPU-offloaded whitening for 70B+ models |
| `svd` | Vanilla SVD | Plain truncated SVD (no activation awareness) |
| `asvd` | ASVD | Activation-scaled SVD |
| `fwsvd` | FWSVD | Fisher-weighted SVD |
| `gfwsvd` | GFWSVD | Gradient Fisher-weighted SVD |
| `dobi_svd` | DOBI-SVD | Double-sided bi-orthogonal SVD |

### Search Methods (`search=`)

| Config | Method | Key Parameters |
|---|---|---|
| `lems` | **LEMS** (Ours) | `solver` (gurobi/cbc), `crosslayer_term`, `halpha`, `hgamma`, `enforce_rank_multiples_of` |
| `uniform` | Uniform | `enforce_rank_multiples_of` |
| `asvd` | ASVD threshold | `sensitivity_loss`, `target_metric`, `min_ratio`, `max_ratio` |
| `asvd_plus` | ASVD+ (bias + Optuna) | Same as ASVD + `crosslayer_term`, `halpha`, `hgamma`, `optuna_trials` |
| `memvit` | MRCS (greedy) | `sensitivity_loss`, `target_metric`, `lower_bound`, `enforce_rank_multiples_of` |
| `memvit_plus` | MRCS+ (bias + Optuna) | Same as MRCS + `crosslayer_term`, `halpha`, `hgamma`, `optuna_trials` |
| `svd_llmv2` | SVD-LLMv2 | `sensitivity_loss` |
| `atp` | ATP | `beta` |
| `loadconfig` | Load from JSON | `layer_compression_json_path` |

## Fine-tuning

After compression, you can fine-tune the compressed model using `finetune_LLM.py`:
```bash
python finetune_LLM.py \
    --model_path outputs/runs/<run_name>/checkpoint.pt \
    --tuning_strategy peft_lora \
    --num_epochs 1 \
    --learning_rate 2e-5
```
Three tuning strategies are available: `peft_lora` (PEFT LoRA adapters), `tune_svd` (unfreeze only decomposed parameters), and `custom_lora` (native LoRA on decomposed layers).

# BibTeX
If our work or code help your work, please cite our paper:
```
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
