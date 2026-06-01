import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" if "CUDA_DEVICE" not in os.environ else os.environ["CUDA_DEVICE"]
import argparse
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    LlamaTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from all_utils.evaluater import *
from all_utils.data_utils import *
from all_utils.model_utils import *
from all_utils.excel_tracking import check_and_create_excel
from peft import PeftModel
from datetime import datetime

def load_for_evaluation(base_model_path, adapter_dir=None, fold_svd=False):
    print("Loading tokenizer...")
    # tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token

    if adapter_dir:
        # --- SCENARIO A: PEFT LORA ---
        print(f"Loading base SVD model from {base_model_path}...")
        model = torch.load(base_model_path)["model"]
        model.to(torch.bfloat16)

        if fold_svd:
            print("Folding SVD layers into standard nn.Linear layers before loading PEFT adapters...")
            model = fold_svd_to_linear(model)

        print(f"Loading PEFT adapters from {adapter_dir}...")
        model = PeftModel.from_pretrained(model, adapter_dir)
        
        # Optional but highly recommended for evaluation speed:
        print("Merging LoRA weights into base SVD matrices...")
        model = model.merge_and_unload() 

    else:
        # --- SCENARIO B: TUNE_SVD or CUSTOM_LORA ---
        # The file is the fully fine-tuned model saved as a .pt
        print(f"Loading fully fine-tuned model from {base_model_path}...")
        model = torch.load(base_model_path)
        model.to(torch.bfloat16)
        
        # Note: If you used custom_lora, the lora_A and lora_B paths are still active.
        # You can mathematically fold them into mod_a and mod_b if you want, 
        # but leaving them as-is works perfectly fine for evaluation.

    model.eval()
    return model #, tokenizer

def fold_svd_to_linear(model):
    """
    Walks through the model, finds SeqSVD or SeqSVDWithLoRA modules, 
    multiplies their internal matrices together, and replaces them 
    with a standard PyTorch nn.Linear layer.
    """
    print("Folding SVD and LoRA components back into standard nn.Linear layers...")
    folded_count = 0
    
    # We cast to a list of items to avoid modifying the dictionary while iterating
    for name, module in list(model.named_modules()):
        class_name = module.__class__.__name__
        
        if class_name in ["SeqSVD", "SeqSVDWithLoRA"]:
            
            # 1. Get the device and dtype to ensure safe matrix multiplication
            dev = module.mod_b.weight.device
            dtype = module.mod_b.weight.dtype
            
            # 2. Compute the base folded SVD weight: W_b @ W_a
            # mod_b.weight shape: (out_features, rank)
            # mod_a.weight shape: (rank, in_features)
            # Resulting shape: (out_features, in_features)
            with torch.no_grad():
                folded_weight = torch.matmul(module.mod_b.weight, module.mod_a.weight)
                
                # 3. Add Custom LoRA weights if they exist (Strategy 3)
                if class_name == "SeqSVDWithLoRA" and module.lora_r > 0:
                    lora_weight = torch.matmul(module.lora_B.weight, module.lora_A.weight)
                    lora_weight = lora_weight * module.scaling
                    folded_weight += lora_weight
            
            # 4. Create the replacement standard nn.Linear layer
            in_features = module.mod_a.in_features
            out_features = module.mod_b.out_features
            has_bias = module.bias is not None
            
            new_linear = nn.Linear(in_features, out_features, bias=has_bias, device=dev, dtype=dtype)
            new_linear.was_folded = True  # Custom attribute for tracking
            
            # 5. Copy the folded weights and bias into the new layer
            with torch.no_grad():
                new_linear.weight.copy_(folded_weight)
                if has_bias:
                    new_linear.bias.copy_(module.bias)
            
            # 6. Swap the module in the parent
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            
            if parent_name == "":
                setattr(model, child_name, new_linear)
            else:
                parent = model.get_submodule(parent_name)
                setattr(parent, child_name, new_linear)
                
            folded_count += 1

            # Free up memory explicitly
            del folded_weight
            torch.cuda.empty_cache()

    print(f"Successfully folded {folded_count} SVD/LoRA layers into standard nn.Linear layers.")
    return model

# =============================================================================
# 1. Custom SVD Module with Native LoRA (Strategy 3 Support)
# =============================================================================
class SeqSVDWithLoRA(nn.Module):
    def __init__(self, mod_a, mod_b, bias=None, lora_r=8, lora_alpha=16, lora_dropout=0.05):
        super().__init__()
        # Keep the original SVD matrices
        self.mod_a = mod_a
        self.mod_b = mod_b
        self.bias = bias
        
        # Native LoRA Implementation
        self.lora_r = lora_r
        if lora_r > 0:
            in_features = mod_a.in_features
            out_features = mod_b.out_features
            
            # Dropout layer matching PEFT
            self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0. else nn.Identity()
            self.lora_A = nn.Linear(in_features, lora_r, bias=False)
            self.lora_B = nn.Linear(lora_r, out_features, bias=False)
            self.scaling = lora_alpha / lora_r
            
            nn.init.zeros_(self.lora_B.weight)
            nn.init.normal_(self.lora_A.weight)

    def forward(self, x):
        base_out = self.mod_b(self.mod_a(x))
        if self.bias is not None:
            base_out += self.bias
            
        if self.lora_r > 0:
            # Apply dropout before passing to LoRA A
            lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
            return base_out + lora_out
            
        return base_out

# =============================================================================
# 2. Freezing & Preparation Logic
# =============================================================================
def inject_custom_lora(model, lora_r=8, lora_alpha=16, lora_dropout=0.05):
    """Walks the model tree and replaces standard SeqSVD with SeqSVDWithLoRA."""
    print("Injecting Custom LoRA modules into SeqSVD layers...")
    
    # Cast to dict to avoid dictionary-changed-during-iteration errors
    for name, module in dict(model.named_modules()).items():
        # Check if it is the original SeqSVD class loaded from your library
        if module.__class__.__name__ == "SeqSVD":
            
            # Instantiate our new upgraded class using the old submodules
            new_module = SeqSVDWithLoRA(
                mod_a=module.mod_a, 
                mod_b=module.mod_b, 
                bias=module.bias,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout
            )
            
            # Figure out the parent module to perform the swap
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            
            if parent_name == "":
                setattr(model, child_name, new_module)
            else:
                parent = model.get_submodule(parent_name)
                setattr(parent, child_name, new_module)
                
    return model

def prepare_model_for_tuning(model, strategy, lora_rank=8, lora_alpha=16, lora_dropout=0.05, extra_lora_targets=None, fold_svd=False, compressed_only=False):
    print(f"Preparing model using strategy: {strategy}")
    
    if strategy == "peft_lora":
        exact_target_modules = set()

        if fold_svd:
            print("Folding SVD layers into standard nn.Linear layers before applying PEFT LoRA...")
            model = fold_svd_to_linear(model)
        
        for name, module in model.named_modules():
            # We ONLY want to attach LoRA to actual Linear layers
            if isinstance(module, nn.Linear):
                
                # 1. Target the SVD submodules
                if "mod_a" in name or "mod_b" in name:
                    exact_target_modules.add(name)
                    
                # 2. Target any uncompressed layers requested by the user
                elif extra_lora_targets and any(t in name for t in extra_lora_targets):
                    if compressed_only:
                        if module.was_folded if hasattr(module, "was_folded") else False:
                            exact_target_modules.add(name)
                        else:
                            print(f"Skipping uncompressed layer {name} due to --compressed_only flag.")
                    else:
                        exact_target_modules.add(name)
        
        print(f"Dynamically found {len(exact_target_modules)} exact linear layers to target.")

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=list(exact_target_modules), # Pass the exact paths
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
    elif strategy == "tune_svd":
        for param in model.parameters():
            param.requires_grad = False
            
        for name, module in model.named_modules():
            if module.__class__.__name__ in ["SeqSVD", "SeqSVDMemViT"]:
                for param in module.parameters():
                    param.requires_grad = True
                    
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable:,} || Total: {total:,} || %: {100 * trainable / total:.2f}")

    elif strategy == "custom_lora":
        # Inject our new custom classes into the loaded model tree
        model = inject_custom_lora(model, lora_r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        
        for param in model.parameters():
            param.requires_grad = False
            
        for name, param in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                param.requires_grad = True
                
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable:,} || Total: {total:,} || %: {100 * trainable / total:.2f}")

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return model

# =============================================================================
# 3. Main Training Loop (Integrated with Streaming C4 & Evaluation)
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Fine-tune an SVD model on streaming C4.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved decomposed model (.pt)")
    parser.add_argument("--tokenizer_id", type=str, default="unsloth/llama-3-8b", help="Base model ID for tokenizer")
    parser.add_argument("--strategy", type=str, choices=["peft_lora", "peft_lora_lowlr", "tune_svd", "custom_lora"], default="tune_svd")
    parser.add_argument("--max_steps", type=int, default=10000, help="Total training steps")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank for custom_lora strategy")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha for custom_lora strategy")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout for custom_lora strategy")
    parser.add_argument(
        "--uncompressed_lora_targets", 
        type=str, 
        default="", 
        help="Comma-separated list of uncompressed layers to target with LoRA (e.g., 'q_proj,v_proj,down_proj')"
    )
    parser.add_argument("--fold_svd", action="store_true", help="Whether to fold SVD and LoRA weights back into standard linear layers before evaluation (recommended for best eval speed)")
    parser.add_argument("--extended_eval", action="store_true", help="Whether to run extended zero-shot evaluation after fine-tuning")
    parser.add_argument("--reload_model", action="store_true", help="Whether to load the model for evaluation from the adapter directory (PEFT scenario) instead of the base .pt file")
    parser.add_argument("--reload_model_override_path", type=str, default="", help="This allows loading the peft checkpoint from another model on the one specified here. Only works if --fold_svd is set")
    parser.add_argument("--excel_tracking", type=str, default="paper_llm_ablation_finetune_results.xlsx", help="Excel file to track results")
    parser.add_argument("--compressed_only", action="store_true", help="Whether to only use compressed layers for fine-tuning")
    args = parser.parse_args()

    # llama3.1 8b example command:
    # CUDA_VISIBLE_DEVICES="7" python finetune_LLM.py --model_path "/data/output/LLM_SVD/ICMLPaper/unsloth_llama_3_1_8b_wikitext2_233_default_svd_llm_elastic_0.5.pt" --strategy "peft_lora" --lora_rank 16 --lora_alpha 32 --max_steps 5000
    # --uncompressed_lora_targets "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

    # llama 2 7b example command:
    # CUDA_VISIBLE_DEVICES="7" python finetune_LLM.py --model_path "/data/output/LLM_SVD/ICMLPaper/unsloth_llama_2_7b_wikitext2_233_default_svd_llm_elastic_0.5.pt" --tokenizer_id="unsloth/llama-2-7b" --strategy "peft_lora" --lora_rank 16 --lora_alpha 32 --max_steps 5000
    # --uncompressed_lora_targets "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

    # 1. Load Tokenizer
    print(f"Loading tokenizer from {args.tokenizer_id}...")
    access_token = load_token() # Ensure token is loaded before tokenizer initialization
    if "llama-2" in args.tokenizer_id.lower():
        tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_id, trust_remote_code=True, token=access_token)
        tokenizer.pad_token = tokenizer.eos_token  # standard in causal language modeling
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id, use_fast=True, token=access_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Decomposed Model
    print(f"Loading SVD model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model = checkpoint['model']
    # Ensure model is in bfloat16 for Llama 3 compatibility/efficiency
    model.to(torch.bfloat16)
    
    # 3. Apply Fine-Tuning Strategy
    extra_targets = args.uncompressed_lora_targets.split(",") if args.uncompressed_lora_targets else None
    model = prepare_model_for_tuning(
        model,
        args.strategy,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        extra_lora_targets=extra_targets,
        fold_svd=args.fold_svd,
        compressed_only=args.compressed_only,
    )

    # 4. Load Streaming C4 Datasets (Train and Validation)
    print("Loading and preparing streaming C4 dataset...")
    
    # Train Split (Shuffled)
    c4_train = load_dataset("allenai/c4", "en", split="train", streaming=True, cache_dir="/data/output/file_cache/huggingface/datasets")
    c4_train = c4_train.shuffle(seed=42, buffer_size=10000)
    
    # Validation Split (Take 500 batches to keep eval fast)
    c4_val = load_dataset("allenai/c4", "en", split="validation", streaming=True, cache_dir="/data/output/file_cache/huggingface/datasets")
    c4_val = c4_val.take(500)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            max_length=1024,
            truncation=True,
        )

    # Apply tokenization mapping
    tokenized_c4_train = c4_train.map(
        tokenize_function,
        batched=True,
        remove_columns=["text", "timestamp", "url"]
    )
    
    tokenized_c4_val = c4_val.map(
        tokenize_function,
        batched=True,
        remove_columns=["text", "timestamp", "url"]
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # 5. Training Arguments
    learning_rate = 2e-5 if args.strategy == "peft_lora_lowlr" else 1e-4

    checkpoint_dir = os.path.dirname(os.path.abspath(args.model_path))
    output_dir = os.path.join(checkpoint_dir, "finetune")
    checkpoint_name = os.path.splitext(os.path.basename(args.model_path))[0]
    checkpoint_name += f"_{args.strategy}_{args.max_steps}steps_lr{learning_rate}_lora{args.lora_rank}r_alpha{args.lora_alpha}"
    if args.fold_svd:
        checkpoint_name += "_folded"
    output_dir = os.path.join(checkpoint_dir, "finetune", checkpoint_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory for this run: {output_dir}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=args.max_steps,              
        per_device_train_batch_size=4,
        warmup_steps=100,
        # gradient_accumulation_steps=4,         
        learning_rate=learning_rate, 
        logging_steps=100,
        
        # --- NEW: Evaluation Arguments ---
        eval_strategy="steps",           # Evaluate periodically
        eval_steps=500,                        # Run eval loop every 500 steps
        # ---------------------------------
        
        save_steps=2000,
        bf16=True,                             
        max_grad_norm=1.0,
        optim="adamw_torch",
        report_to="none",
        seed=42                                # Explicit global seed
    )

    if not args.reload_model:
        # 6. Initialize and Run Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_c4_train,
            eval_dataset=tokenized_c4_val,         # Pass the restricted validation stream
            data_collator=data_collator,
        )

        print(f"Starting fine-tuning on C4 for {args.max_steps} steps...")
        trainer.train()
    else:
        if args.strategy == "peft_lora":
            # In PEFT scenario, we want to load the model with adapters for evaluation
            model = load_for_evaluation(
                base_model_path=args.model_path if not args.reload_model_override_path else args.reload_model_override_path, 
                adapter_dir=os.path.join(output_dir, "final"),
                fold_svd=args.fold_svd
            )
        else:
            # If you used --strategy tune_svd OR custom_lora:
            # Pass ONLY the final saved .pt model
            model = load_for_evaluation(
                base_model_path=os.path.join(output_dir, "final", "model.pt"),
        )
    
    # 7. Save Final Model
    print(f"Saving fine-tuned model to {output_dir}/final...")
    if args.strategy == "peft_lora":
        model.save_pretrained(f"{output_dir}/final")
    else:
        torch.save(model, f"{output_dir}/final/model.pt")
    print("Training complete!")

    device = torch.device("cuda")

    if args.extended_eval:
        try:
            results = zero_shot_eval(model, tokenizer, device=device,
                           tasks=["piqa", "openbookqa", "hellaswag", "arc_challenge", "arc_easy", "winogrande", "boolq", "math_qa_custom"]
            )
            extended_eval_results = {
                "boolq": results["boolq"] if "boolq" in results else "N/A",
                "piqa": results["piqa"] if "piqa" in results else "N/A",
                "openbookqa": results["openbookqa"] if "openbookqa" in results else "N/A",
                "hellaswag": results["hellaswag"] if "hellaswag" in results else "N/A",
                "arc_challenge": results["arc_challenge"] if "arc_challenge" in results else "N/A",
                "arc_easy": results["arc_easy"] if "arc_easy" in results else "N/A",
                "winogrande": results["winogrande"] if "winogrande" in results else "N/A",
                "mathqa": results["math_qa_custom"] if "math_qa_custom" in results else "N/A",
            }
            print(extended_eval_results)
        except:
            print("loading lm_eval failed. Skipping extended evaluation.")
            extended_eval_results = {}
        try:
            prompt = "What is the responsibility of an AI assistant?"
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = inputs.to(device)
            generate_ids = model.generate(**inputs, max_length=len(inputs.input_ids) + 256, do_sample=True, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, top_k=50, top_p=0.95, temperature=0.97,no_repeat_ngram_size=2,)
            answer = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            extended_eval_results["answer"] = answer
            print("Answer to prompt '" + prompt + "': " + answer)
        except:
            print("Answer generation failed. Skipping.")
            extended_eval_results["answer"] = "none"
        ppls = ppl_eval(model, tokenizer, datasets=['wikitext2', 'ptb', 'c4'], model_seq_len=2048,
             batch_size=1, device=device)
        print(f"Eval done. Perplexity on wikitext2: {ppls['wikitext2']}, ptb: {ppls['ptb']}, c4: {ppls['c4']}")
    else:
        ppls = ppl_eval(model, tokenizer, datasets=['wikitext2'], model_seq_len=2048,
             batch_size=1, device=device)
        print(f"Eval done. Perplexity on wikitext2: {ppls['wikitext2']}")
    
    # 8. Write out results to Excel
    write_out_args = {
        "tokenizer_id": args.tokenizer_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": args.model_path,
        "strategy": args.strategy,
        "max_steps": args.max_steps,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "uncompressed_lora_targets": args.uncompressed_lora_targets,
        "fold_svd": args.fold_svd,
        "reload_model_for_eval": args.reload_model,
        "reload_model_override_path": args.reload_model_override_path,
        "ppl_wikitext2": ppls.get("wikitext2", "N/A"),
        "ppl_ptb": ppls.get("ptb", "N/A"),
        "ppl_c4": ppls.get("c4", "N/A"),
        **extended_eval_results
    }
    check_and_create_excel(data_dict=write_out_args, file_path=args.excel_tracking)

if __name__ == "__main__":
    main()