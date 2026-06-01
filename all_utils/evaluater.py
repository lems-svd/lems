import torch
import numpy as np
from tqdm import tqdm
import time
import itertools
from .data_utils import get_test_data
import traceback
import os
import gc



@torch.no_grad()
def ppl_eval(model, tokenizer, datasets=['wikitext2', 'ptb', 'c4'], model_seq_len=2048, batch_size=32, device="cuda", after_prune=True):
    model.to(device)
    model.eval()
    ppls = {}
    for dataset in datasets:
        try:
            test_loader = get_test_data(dataset, tokenizer, seq_len=model_seq_len, batch_size=batch_size)
            nlls = []
            for batch in tqdm(test_loader):
                batch = batch.to(device)
                output = model(batch, use_cache=False)
                lm_logits = output.logits
                if torch.isfinite(lm_logits).all():
                    shift_logits = lm_logits[:, :-1, :].contiguous()
                    shift_labels = batch[:, 1:].contiguous()

                    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                    loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    nlls.append(loss)
            ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
        except Exception as e:
            traceback.print_exc()
            print(f"Error evaluating PPL on {dataset}: {e}")
            ppl = float('inf')
        ppls[dataset] = ppl
    if after_prune:
        print("PPL after pruning: {}".format(ppls))
    else:
        print("PPL before pruning: {}".format(ppls))
    print("Weight Memory: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))
    return ppls

def ppl_eval_window(model, tokenizer, datasets=['wikitext2', 'ptb', 'c4'], model_seq_len=2048, batch_size=32, device="cuda", stride=1024):
    nlls = [] # Negative log-likelihoods

    max_length = model_seq_len #-1  # Model's maximum context length

    
    print(f"Max context length: {max_length}")
    print(f"Calculating PPL with stride: {stride}")

    encodings = get_test_data("wikitext2", tokenizer, seq_len=model_seq_len, batch_size = batch_size, do_process_data=False)
    seq_len = len(encodings)
    print(f"Sequence length: {seq_len} tokens")
    # Move encodings to the target device
    input_ids_ = encodings.to(device).unsqueeze(0)  # Add batch dimension if necessary 

    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = input_ids_[:, begin_loc:end_loc].to(device)
        # bos_tokens_tensor = torch.tensor([tokenizer.bos_token_id]).to(input_ids.device).unsqueeze(0)
        # input_ids = torch.cat([bos_tokens_tensor, input_ids], dim=1)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        # Accumulate the total negative log-likelihood and the total number of tokens
        num_valid_tokens = (target_ids != -100).sum().item()  # number of valid tokens in target_ids
        batch_size = target_ids.size(0)
        num_loss_tokens = num_valid_tokens - batch_size  # subtract batch_size due to internal label shift
        nll_sum += neg_log_likelihood * num_loss_tokens
        n_tokens += num_loss_tokens

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    avg_nll = nll_sum / n_tokens  # average negative log-likelihood per token
    ppl = torch.exp(avg_nll)
    print(f"PPL after pruning: {ppl.item()}")

# only call this function when for 65b or more model    
@torch.no_grad()
def ppl_eval_large(model, tokenizer, datasets=['wikitext2', 'ptb', 'c4'], seq_len=2048, batch_size=32, device="cuda"):
    import  torch.nn as nn
    class LlamaRMSNorm(nn.Module):
        def __init__(self, hidden_size=model.config.hidden_size, eps=model.config.rms_norm_eps):
            """
            LlamaRMSNorm is equivalent to T5LayerNorm
            """
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.variance_epsilon = eps

        def forward(self, hidden_states):
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            return self.weight * hidden_states.to(input_dtype)
    norm = LlamaRMSNorm().half().cuda()
    lm_head = model.lm_head.cuda()
    model.eval()
    ppls = {}
    layers = model.model.layers
    for dataset in datasets:
        test_loader = get_test_data(dataset, tokenizer, seq_len=seq_len, batch_size = batch_size)
        nlls = []
        for batch in tqdm(test_loader):
            model.model.embed_tokens = model.model.embed_tokens.cuda()
            model.model.norm = model.model.norm.cuda()
            layers[0] = layers[0].cuda()

            dtype = next(iter(model.parameters())).dtype
            inps = torch.zeros(
                (batch.shape[0], model.seqlen, model.config.hidden_size), dtype=dtype, device="cuda"
            )
            cache = {'i': 0, 'attention_mask': None, "position_ids": None}
            class Catcher(nn.Module):
                def __init__(self, module):
                    super().__init__()
                    self.module = module
                def forward(self, inp, **kwargs):
                    inps[cache['i']] = inp
                    cache['i'] += 1
                    if cache['attention_mask'] is None:
                        cache['attention_mask'] = kwargs['attention_mask']
                        cache['position_ids'] = kwargs['position_ids']
                    else:
                        cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask']), dim=0)
                        cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids']), dim=0)
                    raise ValueError
            layers[0] = Catcher(layers[0])
            for j in range(batch.shape[0]):
                try:
                    model(batch[j].unsqueeze(0).cuda())
                except ValueError:
                    pass
            layers[0] = layers[0].module
            layers[0] = layers[0].cpu()
            model.model.embed_tokens = model.model.embed_tokens.cpu()
            model.model.norm = model.model.norm.cpu()
            torch.cuda.empty_cache()
            attention_masks = cache['attention_mask']
            position_ids = cache['position_ids']
            for i in range(len(layers)):
                layer = layers[i].cuda()
                outs = layer(inps, attention_mask=attention_masks, position_ids=position_ids)[0]
                layers[i] = layer.cpu()
                inps = outs
                torch.cuda.empty_cache()
            hidden_states = norm(outs)
            lm_logits = lm_head(hidden_states)
            if torch.isfinite(lm_logits).all():
                shift_logits = lm_logits[:, :-1, :].contiguous()
                shift_labels = batch[:, 1:].contiguous().cuda()
                
                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1))
                nlls.append(loss)
            else:
                print("warning: nan or inf in lm_logits")
        ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
        ppls[dataset] = ppl
    print("PPL after pruning: {}".format(ppls))
    print("Weight Memory: {} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))

@torch.no_grad()
def eff_eval(model, tokenizer, dataset='wikitext2', original_len=4, generated_len=128, batch_size=1, device="cuda", use_cache=True):
    # dobi/svd llm evaluation
    model.eval()
    throughput = 0
    token_num = 0
    end_memory = 0
    num_batches_to_fetch = 10
    test_loader = get_test_data(dataset, tokenizer, seq_len=original_len, batch_size = batch_size)
    weight_memory = torch.cuda.memory_allocated()
    for batch_idx, batch_data in enumerate(itertools.islice(test_loader, num_batches_to_fetch)):
        batch = batch_data.to(device)
        token_num += batch.shape[0] * (generated_len + original_len)
        print(f"batch num, {batch.shape[0]} gen len {generated_len}, token num {token_num}")
        torch.cuda.empty_cache()
        start_memory = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.synchronize()
        start_time = time.time()
        generation_output = model.generate(
                input_ids=batch,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,#do_sample=True,
                use_cache=use_cache,
                top_k=50,
                max_new_tokens=generated_len, #max_length = original_len+generated_len,
                top_p=0.95,
                temperature=1,
        )
        torch.cuda.synchronize()
        end_time = time.time()
        end_memory = max(torch.cuda.max_memory_allocated(0), end_memory)
        #if torch.isfinite(generation_output[0]).all():  # check if the generation is successful since fp16 may cause nan
        throughput += end_time - start_time
        print("time: {}".format(end_time - start_time))
    print("Total Memory: {} GB".format(end_memory/(1024 ** 3)))
    print("Weight Memory: {} GB".format(weight_memory/(1024 ** 3)))
    print("Activation Memory: {} GB".format((end_memory - start_memory)/(1024 ** 3)))
    print("Throughput: {} tokens/sec".format(token_num / throughput))

@torch.no_grad()
def zero_shot_eval(model, tokenizer, tasks=["piqa", "openbookqa", "hellaswag", "arc_challenge", "arc_easy", "winogrande", "boolq"], device="cuda"):
    model.to(device)
    model.eval()
    processed_results = {}
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM

        hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size='auto')
        results = lm_eval.simple_evaluate(hflm, tasks=tasks, batch_size='auto', num_fewshot=0)['results']
        for result in results:
            print(result, f"{results[result]['acc,none']*100:.2f}%")
            processed_results[result] = results[result]['acc,none']
        print("average acc:", sum([results[result]['acc,none'] for result in results])/len(results))
    except Exception as e:
        traceback.print_exc()
        if "math_qa_custom" in tasks:
            print("math_qa_custom may not work if not launched from the main repo directory.")
            print("Attempting to evaluate other tasks excluding math_qa_custom...")
            try:
                import lm_eval
                from lm_eval.models.huggingface import HFLM
                tasks = [t for t in tasks if t != "math_qa_custom"]
                hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size='auto')
                results = lm_eval.simple_evaluate(hflm, tasks=tasks, batch_size='auto', num_fewshot=0)['results']
                for result in results:
                    print(result, f"{results[result]['acc,none']*100:.2f}%")
                    processed_results[result] = results[result]['acc,none']
                print("average acc:", sum([results[result]['acc,none'] for result in results])/len(results))
            except Exception as e_inner:
                print("Error during fallback evaluation:", e_inner)
        print("lm-eval-harness evaluation encountered an error:", e)
    return processed_results


def generate_sample(model, tokenizer, prompt="What is the responsibility of an AI assistant?",
                    device="cuda", max_new_tokens=256):
    """Generate a short text sample as a sanity check. Returns the decoded text or ``"none"`` on failure."""
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        gen_ids = model.generate(
            **inputs, max_length=inputs.input_ids.shape[1] + max_new_tokens,
            do_sample=True, eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            top_k=50, top_p=0.95, temperature=0.97, no_repeat_ngram_size=2
        )
        answer = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
        print(f"Answer: {answer}")
        return answer
    except Exception:
        traceback.print_exc()
        print("Answer generation failed. Skipping.")
        return "none"

@torch.no_grad()
def throughput_eval(model, device="cuda"):
    def get_theoretical_model_size_gb(model):
        """
        Calculates the theoretical size of model weights and buffers in GB.
        Useful as a failsafe to verify if GPU memory is holding extra garbage.
        
        Args:
            model: The PyTorch model to analyze.
            
        Returns:
            float: Estimated size in GB
        """
        total_bytes = 0
        # Count parameters (weights/biases)
        for param in model.parameters():
            total_bytes += param.numel() * param.element_size()
        
        # Count buffers (e.g., BatchNorm running stats, position embeddings)
        for buffer in model.buffers():
            total_bytes += buffer.numel() * buffer.element_size()
            
        return total_bytes / (1024**3)
    model.eval()
    model = model.to(device)
    
    # Headers for the table
    headers = ["Scenario", "Time (ms)", "Tokens/s", "Mem (Peak)", "Mem (Act)"]
    print(f"{headers[0]:<20} | {headers[1]:^10} | {headers[2]:^12} | {headers[3]:^10} | {headers[4]:^10}")
    print("-" * 75)

    scenarios = [
        ("Latency (Decode)", 1, 1),      # Single user typing
        ("Throughput (Decode)", 64, 1),  # 64 users typing at once
        ("Prefill (Prompt)", 1, 2048),   # Processing a long document
    ]

    for name, bs, seq in scenarios:
        # 1. Clean up and Baseline
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats(device)
        
        # Capture memory of weights only (before inputs are created)
        # This is unreliable
        weight_memory = torch.cuda.memory_allocated(device)
        
        # 2. Prepare Inputs
        inputs = torch.randint(0, 32000, (bs, seq), device=device)
        
        # 3. Warmup
        # We run this to compile kernels/allocate buffers, but we don't time it
        for _ in range(5): 
            with torch.no_grad(): 
                _ = model(inputs)
        torch.cuda.synchronize()
        
        # Reset peak stats again to capture ONLY the benchmark run peak
        # (This ignores memory spikes that might have happened during initialization/warmup)
        torch.cuda.reset_peak_memory_stats(device)

        # 4. Benchmark Timing
        iterations = 50
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(inputs)
        end_event.record()
        torch.cuda.synchronize()

        # 5. Capture Metrics
        # Peak memory during the actual benchmark loop
        peak_memory = torch.cuda.max_memory_allocated(device)
        
        # Calculations
        total_time_ms = start_event.elapsed_time(end_event)
        avg_time_ms = total_time_ms / iterations
        tok_per_sec = (bs * seq) / (avg_time_ms / 1000)

        # Conversions to GB
        GB = 1024 ** 3
        peak_memory_gb = peak_memory / GB
        weight_memory_gb_measured = weight_memory / GB

        weight_memory_gb_count = get_theoretical_model_size_gb(model)
        # Activation (+ Input) Memory = Peak - Static Weights
        activation_memory_gb = (peak_memory/ GB - weight_memory_gb_count) 
        
        print(f"{name:<20} | {avg_time_ms:>10.2f} | {tok_per_sec:>12.0f} | {peak_memory_gb:>8.2f}GB | {activation_memory_gb:>8.2f}GB")

    del inputs, _
    print("-" * 75)
    print(f"Model based Weight Memory: {weight_memory_gb_count:.2f} GB")
    print(f"GPU based Weight Memory: {weight_memory_gb_measured:.2f} GB")
    print(f"Computed activation memory: {peak_memory_gb - weight_memory_gb_count:.2f} GB")


def benchmark_inference(model, tokenizer, prompt=None, max_new_tokens=50, device="cpu"):
    """
    Benchmarks the inference speed (TTFT and TPS) with and without KV Caching.
    """
    model = model.to(device)
    model.eval()
    prompt = prompt if prompt else "The quick brown fox jumps over the lazy dog to"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    print(f"--- Benchmarking on {device.upper()} ---")
    print(f"Input text: '{prompt}'")
    print(f"Generating {max_new_tokens} new tokens...\n")

    # Helper to track timing
    def run_generation(use_cache):
        # warmup
        with torch.no_grad():
            _ = model.generate(input_ids, max_new_tokens=10, do_sample=False)
        # Reset specific to this run
        curr_input_ids = input_ids.clone()
        past_key_values = None
        token_times = []
        
        # Warmup (optional, helps stabilize GPU clocks if using CUDA)
        if device == "cuda":
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        # --- GENERATION LOOP ---
        for i in range(max_new_tokens):
            step_start = time.perf_counter()
            torch.compiler.cudagraph_mark_step_begin()

            with torch.no_grad():
                if use_cache:
                    # OPTIMIZED: Use Cache
                    if i == 0:
                        # First step: Process the whole prompt
                        outputs = model(curr_input_ids, use_cache=True)
                    else:
                        # Subsequent steps: Process ONLY the last token
                        # We pass the cache (past_key_values) from the previous step
                        last_token = curr_input_ids[:, -1:]
                        # last_token = curr_input_ids[:, -1:].clone()
                        outputs = model(last_token, past_key_values=past_key_values, use_cache=True)
                    
                    # Update cache for next step
                    past_key_values = outputs.past_key_values
                    
                else:
                    # NAIVE: No Cache
                    # We pass the ENTIRE sequence every time
                    # We strictly set use_cache=False so the model doesn't return or use keys
                    outputs = model(curr_input_ids, use_cache=False)
            
            # Greedy decoding: pick the token with highest probability
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            
            # Append new token to sequence
            curr_input_ids = torch.cat([curr_input_ids, next_token], dim=1)
            
            # Sync GPU for accurate timing
            if device == "cuda":
                torch.cuda.synchronize()
            
            step_end = time.perf_counter()
            token_times.append(step_end - step_start)

        total_time = time.perf_counter() - start_time
        decoded_text = tokenizer.decode(curr_input_ids[0], skip_special_tokens=True)
        
        return token_times, total_time, decoded_text

    # 1. RUN WITH CACHE
    print("Running WITH KV Cache...")
    times_cache, total_cache, text_cache = run_generation(use_cache=True)
    
    # 2. RUN WITHOUT CACHE
    print("Running WITHOUT KV Cache...")
    times_no_cache, total_no_cache, text_no_cache = run_generation(use_cache=False)

    # --- REPORTING ---
    def print_stats(name, times, total):
        ttft = times[0] * 1000  # ms
        # Exclude first token for TPS calculation to measure generation speed purely
        avg_step_time = np.mean(times[1:]) 
        tps = 1 / avg_step_time
        
        print(f"\nResults for {name}:")
        print(f"  Time To First Token (TTFT): {ttft:.2f} ms")
        print(f"  Average Gen Speed (TPS):    {tps:.2f} tokens/sec")
        print(f"  Total Time:                 {total:.2f} sec")
        print(f"  Slowdown per token:         {(times[-1] - times[1])*1000:.2f} ms increase (approx)")

    print_stats("WITH CACHE", times_cache, total_cache)
    print_stats("WITHOUT CACHE", times_no_cache, total_no_cache)
    
    # Speedup calculation
    speedup = total_no_cache / total_cache
    print(f"\n>>> Overall Speedup: {speedup:.2f}x")
    print("-" * 30)

def benchmark_inference_generate(model, tokenizer, prompt=None, max_new_tokens=50, device="cpu"):
    """
    Benchmarks the inference speed (TTFT and TPS) with and without KV Caching.
    """
    model = model.to(device)
    model.eval()
    prompt = prompt if prompt else "The quick brown fox jumps over the lazy dog to"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    print(f"--- Benchmarking on {device.upper()} ---")
    print(f"Input text: '{prompt}'")
    print(f"Generating {max_new_tokens} new tokens...\n")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.generation import BaseStreamer

    class PerformanceStreamer(BaseStreamer):
        def __init__(self):
            self.token_times = []
            self.start_time = None
            self.end_time = None

        def put(self, value):
            """Called by .generate() every time new tokens are produced."""
            current_time = time.perf_counter()
            
            # If this is the very first time we receive tokens, start the clock
            # Note: The first 'put' often contains the entire input prompt + 1st generated token
            if self.start_time is None:
                self.start_time = current_time
                # We record the time for the "first token" (which includes prefill)
                self.token_times.append(current_time)
            else:
                # Record time delta for subsequent tokens
                self.token_times.append(current_time)

        def end(self):
            """Called when generation finishes."""
            self.end_time = time.perf_counter()

        def get_metrics(self):
            # Calculate deltas between timestamps
            deltas = np.diff(self.token_times)
            
            # 1. TTFT (Time To First Token)
            # In a streamer, the timer starts when .generate() is called (outside this class),
            # but we can approximate the prefill time by looking at the start timestamp 
            # relative to when the function was invoked.
            # *However, accurate TTFT requires external start time.* # 2. TPS (Tokens Per Second) - Exclude the prefill step
            if len(deltas) > 0:
                avg_step_time = np.mean(deltas)
                tps = 1 / avg_step_time
            else:
                tps = 0
                
            return tps, deltas

    # --- RUN ---

    inputs = tokenizer("The quick brown fox jumps over the lazy dog", return_tensors="pt").to(model.device)

    # Instantiate our custom streamer
    perf_streamer = PerformanceStreamer()

    print("Generating...")
    start_wall_clock = time.perf_counter()

    # We pass the streamer to the generate function
    _ = model.generate(
        **inputs, 
        max_new_tokens=50, 
        streamer=perf_streamer,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=False,
        do_sample=False
    )

    # Calculate results
    tps, deltas = perf_streamer.get_metrics()
    ttft = (perf_streamer.token_times[0] - start_wall_clock) * 1000 # ms

    print(f"\nResults via model.generate():")
    print(f"  TTFT (approx): {ttft:.2f} ms")
    print(f"  TPS (Decode):  {tps:.2f} tokens/sec")

def benchmark_batched(model, tokenizer, prompt=None, batch_size=8, use_static_cache=True, max_new_tokens=50, device="cpu"):
    """
    Benchmarks the inference speed (TTFT and TPS) with and without KV Caching.
    """
    model = model.to(device)
    model = torch.compile(model, mode="reduce-overhead")
    model.eval()
    prompt = prompt if prompt else "The quick brown fox jumps over the lazy dog to"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    model.eval()

    from transformers import StaticCache

    # 2. Compile Strategy
    # if use_static_cache:
    #     print("-> Compiler: 'reduce-overhead' (CUDA Graphs)")
    #     # Static Cache allows aggressive graph capture
    #     model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    # else:
    #     print("-> Compiler: 'max-autotune' (Kernel Fusion)")
    #     # Dynamic Cache requires dynamic shape support
    #     model = torch.compile(model, mode="max-autotune", dynamic=True)

    # 3. Prepare Batch
    # Encode single prompt
    single_input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    # Repeat it to match batch_size (Shape: [batch_size, seq_len])
    input_ids = single_input_ids.repeat(batch_size, 1)
    
    seq_len = input_ids.shape[1]
    print(f"-> Input Shape: {input_ids.shape}")

    # --- GENERATION LOOP ---
    def run_generation():
        curr_input_ids = input_ids.clone()
        token_times = []
        
        # A. Init Cache
        if use_static_cache:
            past_key_values = StaticCache(
                config=model.config,
                max_batch_size=batch_size, # CRITICAL: Cache must fit the batch
                max_cache_len=seq_len + max_new_tokens + 1,
                device=device,
                dtype=model.dtype
            )
            cache_position = torch.arange(seq_len, device=device)
        else:
            past_key_values = None 
            cache_position = None
        
        # warmup
        outputs = model(
                        curr_input_ids, 
                        past_key_values=past_key_values, 
                        use_cache=True,
                        cache_position=cache_position
                    )

        if device == "cuda": torch.cuda.synchronize()
        start_time = time.perf_counter()

        for i in range(max_new_tokens):
            step_start = time.perf_counter()
            
            # CUDA Graph Marker
            if use_static_cache:
                torch.compiler.cudagraph_mark_step_begin()

            with torch.no_grad():
                # --- PREFILL ---
                if i == 0:
                    outputs = model(
                        curr_input_ids, 
                        past_key_values=past_key_values, 
                        use_cache=True,
                        cache_position=cache_position
                    )
                # --- DECODING ---
                else:
                    last_token = curr_input_ids[:, -1:]
                    
                    if use_static_cache:
                        current_cache_pos = torch.tensor([seq_len + i - 1], device=device)
                        outputs = model(
                            last_token, 
                            past_key_values=past_key_values, 
                            use_cache=True, 
                            cache_position=current_cache_pos
                        )
                    else:
                        outputs = model(
                            last_token, 
                            past_key_values=past_key_values, 
                            use_cache=True
                        )

                if not use_static_cache:
                    past_key_values = outputs.past_key_values

            # Greedy Decode
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
            curr_input_ids = torch.cat([curr_input_ids, next_token], dim=1)

            if device == "cuda": torch.cuda.synchronize()
            step_end = time.perf_counter()
            token_times.append(step_end - step_start)

        total_time = time.perf_counter() - start_time
        return token_times, total_time

    # 4. Warmup & Compile
    print("\nRunning Warmup...")
    try:
        run_generation()
    except Exception as e:
        print(f"Error in execution: {e}")
        return

    # 5. Benchmark
    print("Running Benchmark...")
    times, total_time = run_generation()

    # 6. Metrics
    ttft = times[0] * 1000
    # TPS = (Tokens per step * Number of steps) / Time
    # Note: Tokens per step = batch_size
    total_tokens_generated = max_new_tokens * batch_size
    # We calculate TPS based on the decoding phase only (excluding prefill)
    decoding_time = sum(times[1:])
    decoding_tokens = (max_new_tokens - 1) * batch_size
    
    tps = decoding_tokens / decoding_time

    print(f"\n>>> RESULTS (Batch Size: {batch_size}):")
    print(f"  TTFT: {ttft:.2f} ms")
    print(f"  TPS:  {tps:.2f} tokens/sec (Throughput)")
    print(f"  Latency per step: {decoding_time / (max_new_tokens - 1) * 1000:.2f} ms")