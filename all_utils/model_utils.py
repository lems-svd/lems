#coding:utf8
import os
import torch
import torch.nn as nn
import json

parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# bandaid fix
dev = torch.device("cuda")
def load_token():
    try:
        with open(os.path.join(parent_path, 'huggingface_token.json'), 'r') as f:
            token_data = json.load(f)
        return token_data.get('access_token')
    except:
        return None

def get_model_from_huggingface(model_id, seq_len, grad_ckpt, fp32=False, cache_dir=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
    access_token = load_token()
    if "llama-2" in model_id.lower():
        tokenizer = LlamaTokenizer.from_pretrained(model_id, device_map="cpu", trust_remote_code=True, cache_dir=cache_dir, token=access_token)
        tokenizer.pad_token = tokenizer.eos_token  # standard in causal language modeling
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="cpu", trust_remote_code=True, cache_dir=cache_dir, token=access_token)
        print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}, tokenizer type: {type(tokenizer)}")
    if fp32:
        dtype = torch.float32
    else:
        dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu", torch_dtype=dtype, trust_remote_code=True, cache_dir=cache_dir, token=access_token)
    model.seqlen = seq_len
    if grad_ckpt:
        print("Gradient checkpointing enabled.")
        # checkout torch docs about this https://docs.pytorch.org/docs/stable/checkpoint.html
        # git issue with suggestion to use it: https://github.com/huggingface/transformers/issues/21381
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model, tokenizer

def get_model_from_local(model_id):
    pruned_dict = torch.load(model_id, weights_only=False, map_location='cpu')
    tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']
    return model, tokenizer

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res