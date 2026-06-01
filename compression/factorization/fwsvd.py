import gc

import torch
from torch import nn
from tqdm import tqdm

from ._interface import BaseFactorization
from ._interface import FactorizedMatrix
from ._interface import Hookstuff
from ._interface import get_valid_layers


class FWSVD_Hook(Hookstuff):
    def _hook_fn(self, layer_name):
        def get_scaling_mat(module, input, output):
            x = input[0].detach().float()
            self.input_shape[layer_name] = list(x.shape)
            self.input_shape[layer_name].extend([module.out_features, 0])
            return

        return get_scaling_mat


class FWSVDFactorization(BaseFactorization):
    def __init__(self, alpha, vision, *args, **kwargs):
        super().__init__(vision=vision, *args, **kwargs)
        self.alpha = alpha
    
    def _compute_scaling(self, model, hook_module, name_prefix, calib_data, name_omit, mixup_fn=None, white_list=[], tqdm_message="Gathering "):
        torch.cuda.empty_cache()
        self._enable_weight_grads(model, hook_module, name_omit, white_list)
        copied_modules = get_valid_layers(hook_module, name_omit, white_list=white_list)
        
        model = model.to(self.dev)
        loss_fn = nn.CrossEntropyLoss()

        for batch in tqdm(calib_data, desc=tqdm_message + " fisher information"):
            # VISION MODELS
            if self.vision:
                data, target = batch
                model_inputs, target_mix = mixup_fn(data, target) if mixup_fn is not None else (data, target)
                model_inputs, target_mix = model_inputs.to(self.dev), target_mix.to(self.dev)
                out = model(model_inputs)
                loss = loss_fn(out, target_mix).mean()
            # LLM MODELS
            else:
                input_ids = batch["input_ids"].to(self.dev)
                out = model(input_ids=input_ids[:, :-1], labels=input_ids[:, 1:])
                loss = out.loss
            # BOTH
            loss.backward()
            for name, module in copied_modules:
                key = name_prefix + name
                if module.weight.grad is not None:
                    tmp = module.weight.grad.detach().pow(2).mean(0)
                    if key not in self.scaling_dict:
                        self.scaling_dict[key] = tmp
                    else:
                        self.scaling_dict[key] += tmp
            model.zero_grad()

        for key, val in self.scaling_dict.items():
            self.scaling_dict[key] = (val / len(calib_data)).sqrt()
        
        if self.vision:
            shapes_getter = FWSVD_Hook(
                model=hook_module,
                name_omit=name_omit,
                dump_shape=True,
                name_prefix=name_prefix,
                white_list=white_list
            )
            shapes_getter.attach_hooks()
            dummy_input = torch.randn(20, 3, 224, 224).to(self.dev)
            model(dummy_input)
            shapes_getter.clear_hooks()
            for key, value in shapes_getter.input_shape.items():
                self.input_shapes[key] = value
            del shapes_getter, dummy_input
        
        torch.cuda.empty_cache()
        gc.collect()
        return

    def _factorize_matrix(self, matrix, eq_rank, rank, name, dev, verbose=False):
        dev = torch.device(torch.cuda.current_device())
        raw_profile = self.scaling_dict[name]
        scale_diag = raw_profile**self.alpha + 1e-6

        if rank == 0:
            rank = eq_rank
        elif rank > eq_rank:
            print(f"Warning: {name} rank is larger than equivalent rank!")
            return

        mat_scaled = matrix.to(dev) * scale_diag.view(1, -1).to(dev)
        dtype = mat_scaled.dtype
        mat_scaled = mat_scaled.float()  # Ensure float for SVD

        u, s, vh = torch.svd_lowrank(mat_scaled, q=rank)
        s_val = torch.sqrt(torch.diag(s))  # half singular value
        vh = (vh / scale_diag.view(-1, 1)).t()

        s_val = torch.sqrt(torch.diag(s))  # half singular value
        mat_l = u @ s_val
        mat_l = mat_l[:, :rank].cpu().to(dtype)
        mat_r = s_val @ vh
        mat_r = mat_r[:rank, :].cpu().to(dtype)

        return FactorizedMatrix(
            mat_l=mat_l,  # Left singular vectors
            mat_r=mat_r,  # Right singular vectors
            eq_rank=eq_rank,  # Equivalent rank
            active_rank=rank,  # Active rank
            singular_values=s
        )
