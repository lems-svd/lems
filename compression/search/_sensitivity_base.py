from copy import deepcopy

import torch
from torch import nn
from all_utils.model_io import get_model_name

from ..factorization._interface import BaseFactorization, get_valid_layers, ShapeHook
from ._interface import BaseSearch
import torch.nn.functional as F
import numpy as np
from all_utils.distances import wasserstein_from_logits_3d_fast, jsd_from_logits_3d, bild_loss
import pickle
import os
from datetime import datetime
import gc

class LastFeatureHook:
    def __init__(self, model: nn.Module):
        self.model = model

        self.hooks = []
        self.cp_modules = reversed(
            [
                (name, module_sub)
                for name, module_sub in model.named_modules()
                # if all(omit not in name for omit in name_omit)
                if isinstance(module_sub, nn.Linear)
            ]
        ) # TODO: use centralized valid Linear functions.

    def _hook_fn(self, layer_name):
        def get_feature_extract_hook(module, input, output):
            if "head" in layer_name:
                x = input[0].detach().float()
                if x.dim() > 3:
                    x = x.reshape(x.shape[0], -1, x.shape[-1])
                elif x.dim() == 2:
                    x = x.unsqueeze(0)
                self.model.last_feat = x.clone()
                # self.model.last_feat = output   # if layer: blocks.-1.mlp.fc2

        return get_feature_extract_hook

    def _register_hooks_recursive(self):
        for name, layer in self.cp_modules:
            if layer.out_features < 10:
                continue  # for some head matrix, such as image-text match head

            hook = layer.register_forward_hook(self._hook_fn(name))
            self.hooks.append(hook)
            # if "head" in name: # continue
            return

    def attach_hooks(self):
        self._register_hooks_recursive()

    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()

class SensitivityBasedSearch(BaseSearch):
    def __init__(self, eval_data, mixup_fn, name_omit=[], ratio_target=0.5, sensitivity_loss="kl", measurements_points="0.1-0.9", sequence_length=256, use_cache=False, workspace_dir="./workspace", **kwargs):
        self.eval_data = tuple(data for data in eval_data)
        self.name_omit = name_omit
        self.sensitivity_dict = {}
        self.lrd_method = None
        self.ratio_target = ratio_target
        self.sensitivity_loss = sensitivity_loss
        self.sequence_length = sequence_length
        if "energy1" in sensitivity_loss:
            self.power_for_energy = 1
        else:
            self.power_for_energy = 2
        self.use_cache = use_cache
        self.sensitivity_cache_dir = os.path.join(workspace_dir, "cache", "sensitivity") + "/"
        measurements_points = measurements_points if type(measurements_points) is str else str(measurements_points)
        self.measurement_point_name = measurements_points

        if measurements_points == "0.1-0.9":
            self.measurements_points = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        elif measurements_points == "0.2-0.9":
            self.measurements_points = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        elif measurements_points == "0.1-0.9uneven":
            self.measurements_points = [0.1, 0.3, 0.5, 0.7, 0.9]
        elif measurements_points == "asvd_default":
            self.measurements_points = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        elif measurements_points == "asvd_gfwsvd":
            self.measurements_points = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        elif measurements_points == "gfwsvd":
            self.measurements_points = [i / 20.0 for i in range(1, 20)]
        elif measurements_points == "0.01":
            self.measurements_points = [0.01]
        elif measurements_points == "0.05":
            self.measurements_points = [0.05]
        elif measurements_points == "0.1":
            self.measurements_points = [0.1]
        elif measurements_points == "0.3":
            self.measurements_points = [0.3]
        elif measurements_points == "0.5":
            self.measurements_points = [0.5]
        elif measurements_points == "0.7":
            self.measurements_points = [0.7]
        else:
            raise ValueError(
                f"Unknown measurements points: {measurements_points}. "
                "Use '0.1', '0.3', '0.5', '0.7', '0.1-0.9', '0.2-0.9', '0.1-0.9uneven', 'gfwsvd' or 'asvd_default'."
            )
        print(f"Using measurement points {self.measurements_points} for the sensitivity search.")

    @property
    def requires_decomposed_model_for_search(self):
        return True

    def initialize_search(
        self, lrd_method: BaseFactorization, model: nn.Module, spec_tensor=None
    ):
        self.lrd_method = lrd_method
        layer_sensitivity, size_dict = self._get_layer_sensitivity(model, spec_tensor)
        self.size_dict = size_dict
        self.sensitivity_dict = layer_sensitivity

    def search(self, model: nn.Module):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def _precompute_original_outputs(self, model: nn.Module):
        dev = torch.device(torch.cuda.current_device())
        model = model.to(dev).eval()
        
        original_outputs = []
        
        hook_register_model = None
        if "mse" in self.sensitivity_loss:
            hook_register_model = LastFeatureHook(model)
            hook_register_model.attach_hooks()

        print("Pre-computing original model outputs for reference...")
        with torch.no_grad():
            for batch in self.eval_data:
                if self.lrd_method.vision:
                    samples, _ = batch
                    model_inputs = samples.to(dev)
                    outputs = model(model_inputs)
                    if self.sensitivity_loss == "mse":
                        original_outputs.append(model.last_feat.clone().detach().cpu())
                    else:
                        original_outputs.append(outputs.clone().detach().to(torch.bfloat16).cpu())
                else:
                    batch = {k: v.to(dev) for k, v in batch.items()}
                    outputs = model(**batch)
                    
                    if "mse" in self.sensitivity_loss:
                        original_outputs.append(model.last_feat.clone().detach().cpu())
                    else:
                        original_outputs.append(outputs.logits.clone().detach().to(torch.bfloat16).cpu())

        if hook_register_model:
            hook_register_model.clear_hooks()
            
        print("Attempting to move all reference outputs to GPU...")
        gpu_outputs = []
        try:
            for out in original_outputs:
                # gpu_outputs.append(out.to(dev))
                gpu_outputs.append(out)
            print("Success: All reference outputs are pinned to the GPU.")
            # Swap our CPU list out for the GPU list
            original_outputs = gpu_outputs 
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("VRAM limit reached: Keeping reference outputs on CPU and moving dynamically.")
                del gpu_outputs
                torch.cuda.empty_cache()
            else:
                raise e
        
        torch.cuda.empty_cache()
        print("Pre-computing original model outputs done.")
        print("Collating and pinning CPU memory for fast PCIe transfers...")
        
        # 1. Collate and pin the inputs
        self.pinned_eval_inputs = {
            k: torch.cat([b[k] for b in self.eval_data], dim=0).pin_memory()
            for k in self.eval_data[0].keys()
        }
        
        # 2. Collate and pin the references (already converted to bf16 earlier)
        self.pinned_original_outputs = torch.cat(original_outputs, dim=0).pin_memory()

        torch.cuda.empty_cache()
        print("Pre-computing original model outputs done.")
        
        # You don't even need to return original_outputs anymore if you use the class attributes
        return self.pinned_original_outputs

    def _eval_llm(self, cp_model, original_outputs):
        print(self.sensitivity_loss)
        dev = torch.device(torch.cuda.current_device())
        cp_model.eval()

        hook_register_model_copy = None
        if "mse" in self.sensitivity_loss:
            hook_register_model_copy = LastFeatureHook(cp_model)
            hook_register_model_copy.attach_hooks()

        total_loss_tensor = torch.tensor(0.0, device=dev)
        nlls = []
        
        total_samples = len(self.eval_data)
        chunk_size = total_samples // 8 
        success = False

        while chunk_size > 0 and not success:
            try:
                total_loss_tensor.zero_()
                nlls.clear()
                
                for i in range(0, total_samples, chunk_size):
                    chunk_data = self.eval_data[i : i + chunk_size]
                    chunk_refs = original_outputs[i : i + chunk_size]
                    
                    batched_inputs = {
                        k: v[i : i + chunk_size].to(dev, non_blocking=True)
                        for k, v in self.pinned_eval_inputs.items()
                    }
                    batched_refs = self.pinned_original_outputs[i : i + chunk_size].to(dev, non_blocking=True)

                    current_chunk_len = batched_refs.shape[0]

                    with torch.no_grad():
                        outputs_cp = cp_model(**batched_inputs)

                        if "mse" in self.sensitivity_loss:
                            L_fm = F.mse_loss(cp_model.last_feat, batched_refs)
                            loss = L_fm / torch.mean(batched_refs**2)
                        elif "kl" in self.sensitivity_loss:
                            probs_target = batched_refs.float() / 0.6
                            probs_cp = outputs_cp.logits / 0.6
                            if torch.isfinite(probs_target).all() and torch.isfinite(probs_cp).all():
                                probs_target = F.softmax(probs_target[:, 256:, :], dim=-1).flatten(0, 1)
                                probs_cp = F.log_softmax(probs_cp[:, 256:, :], dim=-1).flatten(0, 1)
                                loss = F.kl_div(probs_cp, probs_target, reduction='batchmean')
                            else:
                                loss = torch.tensor(0.0, device=dev)
                        elif "ppl" in self.sensitivity_loss:
                            lm_logits = outputs_cp.logits
                            if torch.isfinite(lm_logits).all():
                                shift_logits = lm_logits[:, :-1, :].contiguous()
                                shift_labels = batched_inputs["input_ids"][:, 1:].contiguous()
                                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                                loss_ = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1))
                                nlls.append(loss_)
                                loss = torch.tensor(0.0, device=dev)
                        elif "wasserstein" in self.sensitivity_loss:
                            w_distance = wasserstein_from_logits_3d_fast(outputs_cp.logits, batched_refs)
                            loss = torch.mean(w_distance) / self.sequence_length
                        elif self.sensitivity_loss == "jsd":
                            loss = jsd_from_logits_3d(outputs_cp.logits, batched_refs)
                        elif self.sensitivity_loss == "bild":
                            logits_s = outputs_cp.logits
                            t_ld_loss = bild_loss(logits_s, batched_refs, top_k=8, temperature=3.0, student_led=False)
                            s_ld_loss = bild_loss(logits_s, batched_refs, top_k=8, temperature=3.0, student_led=True)
                            loss = torch.mean(t_ld_loss + s_ld_loss)

                    total_loss_tensor += loss * (current_chunk_len / total_samples)
                    
                    del batched_inputs, batched_refs
                    del outputs_cp
                    gc.collect() 
                    torch.cuda.empty_cache()
                
                success = True 

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    chunk_size //= 2
                    print(f"OOM detected in forward pass. Scaling batch chunk down to {chunk_size}...")
                    if 'outputs_cp' in locals(): del outputs_cp
                    if 'batched_inputs' in locals(): del batched_inputs
                    if 'batched_refs' in locals(): del batched_refs
                    gc.collect()
                    torch.cuda.empty_cache()
                else:
                    raise e

        if hook_register_model_copy:
            hook_register_model_copy.clear_hooks()

        avg_loss = total_loss_tensor.item()
        metric = np.exp(torch.cat(nlls, dim=-1).mean().item()) if self.sensitivity_loss == "ppl" else avg_loss
        return metric
    
    def _eval_vision(self, cp_model, original_outputs):
        dev = torch.device(torch.cuda.current_device())
        cp_model = cp_model.to(dev)
        cp_model = cp_model.eval()
        #loss_fc = nn.CrossEntropyLoss()
        sensitivity = torch.tensor(0.0, device=dev)
        if "mse" in self.sensitivity_loss:
            # prepare hooks for last feature extraction needed for MSE
            hook_register_model_copy = LastFeatureHook(cp_model)
            hook_register_model_copy.attach_hooks()
        with torch.no_grad():
            for (samples, targets), orig_out in zip(self.eval_data, original_outputs):
                model_inputs, labels = samples.to(dev), targets.to(dev)
                outputs_cp = cp_model(model_inputs)
                if "mse" in self.sensitivity_loss:
                    outputs_neutral = orig_out.to(dev)
                    L_fm = F.mse_loss(orig_out, cp_model.last_feat)
                    loss = L_fm / torch.mean(cp_model.last_feat**2)
                elif "kl" in self.sensitivity_loss:
                    outputs_neutral = orig_out.to(dev)
                    outputs_neutral = outputs_neutral / 0.6
                    outputs_cp = outputs_cp / 0.6
                    probs_target = F.softmax(outputs_neutral.detach(), dim=-1)
                    probs_cp = F.log_softmax(outputs_cp, dim=-1)
                    del model_inputs
                    loss = F.kl_div(probs_cp, probs_target, reduction='batchmean')
                elif "ce" in self.sensitivity_loss:
                    outputs_cp = F.softmax(outputs_cp.detach(), dim=-1)
                    loss = F.cross_entropy(outputs_cp, labels, reduction='mean')
                else:
                    raise ValueError(f"Unknown sensitivity loss type: {self.sensitivity_loss}")
                sensitivity += loss
                # ppl = torch.exp(loss)
                del loss, outputs_cp, outputs_neutral, orig_out# , probs_target, probs_cp

                with torch.cuda.device(torch.cuda.current_device()):
                    torch.cuda.empty_cache()
        
        if "mse" in self.sensitivity_loss:
            hook_register_model_copy.clear_hooks()

        return sensitivity.item()

    def _get_layer_sensitivity(self, model: nn.Module, spec_tensor=None):
        cache_loaded = False

        model_name = get_model_name(model)
        
        if self.use_cache:
            try:
                with open(f"{self.sensitivity_cache_dir}layer_sensitivity_{self.lrd_method.get_cache_name()}_{self.sensitivity_loss}_{self.measurement_point_name}_{model_name}.pkl", "rb") as f:
                    layer_sensitivity = pickle.load(f)
                with open(f"{self.sensitivity_cache_dir}size_dict_{model_name}.pkl", "rb") as f:
                    size_dict = pickle.load(f)
                cache_loaded = True
                for layer_name, sensitivity_data in layer_sensitivity.items():
                    layer_sensitivity[layer_name] = {k: v for k, v in sensitivity_data.items() if k >= 0.1 and k <= 0.95}
                print("Loaded cached layer sensitivity data.")
            except FileNotFoundError:
                print("No cached sensitivity data found. Proceeding with new calculations.")

        
        if not cache_loaded:
            layer_sensitivity, size_dict = self._compute_layer_sensitivity(model, spec_tensor)
            if self.use_cache:
                if not os.path.exists(self.sensitivity_cache_dir):
                    os.makedirs(self.sensitivity_cache_dir)
                with open(f"{self.sensitivity_cache_dir}layer_sensitivity_{self.lrd_method.get_cache_name()}_{self.sensitivity_loss}_{self.measurement_point_name}_{model_name}.pkl", "wb") as f:
                    pickle.dump(layer_sensitivity, f)
                with open(f"{self.sensitivity_cache_dir}size_dict_{model_name}.pkl", "wb") as f:
                    pickle.dump(size_dict, f)
                print("Saved layer sensitivity data to cache.")
        
        return layer_sensitivity, size_dict

    def _compute_layer_sensitivity(self, model: nn.Module, spec_tensor=None):
        model_forwards_required = "energy" not in self.sensitivity_loss or "_klscaled" in self.sensitivity_loss or "_msescaled" in self.sensitivity_loss or "_pplscaled" in self.sensitivity_loss
        if model_forwards_required:
            original_outputs = self._precompute_original_outputs(model)
        
        dev = torch.device(torch.cuda.current_device())
        model = model.to(dev)

        sensitivity_dict = {}
        size_dict = {}
        
        for name, module_sub in list(model.named_modules()):
            if isinstance(module_sub, nn.Linear):
                if any(n in name for n in self.name_omit) or module_sub.out_features < 10:
                    continue

                print(f"Evaluating sensitivity for layer {name}")
                start_time = datetime.now()

                base, localname = model, name
                while "." in localname:
                    prefix, localname = localname.split(".", 1)
                    base = base.__getattr__(prefix)

                sensitivity_dict[name] = {}
                size_dict[name] = module_sub.weight.numel()
                
                factorized_matrix = self.lrd_method.factorize_matrix(
                    name=name, matrix=module_sub.weight, ratio=1.0
                )
                
                if "energy" in self.sensitivity_loss:
                    max_rank = factorized_matrix.eq_rank
                    
                    # FIX: Move singular values to the GPU right away
                    sv_float = factorized_matrix.singular_values.float().to(dev)
                    
                    # 1. Power the singular values once (on GPU)
                    sv_pow = torch.pow(sv_float, self.power_for_energy)
                    
                    # 2. Cumulative sum on CPU to respect determinism, then back to GPU
                    sv_cumsum = torch.cumsum(sv_pow.cpu(), dim=0).to(dev)
                    
                    total_energy = sv_cumsum[-1]
                    
                    klscaled_idx = max(0, int(max_rank * self.measurements_points[0]) - 1)
                    klscaled_reference_point_energy = sv_cumsum[klscaled_idx]
                    eq_rank_energy_loss = total_energy - sv_cumsum[max_rank - 1]
                    
                    # 3. Vectorize across all target ranks
                    start_rank = int(max_rank * 0.1)
                    end_rank = int(max_rank * 0.95)
                    
                    if start_rank < end_rank:
                        # ranks is on dev, and now sv_cumsum is too
                        ranks = torch.arange(start_rank, end_rank, device=dev)
                        remaining_energies = sv_cumsum[ranks - 1] 
                        
                        # Apply your specific loss math vectorized
                        if self.sensitivity_loss == "energy2_eqoffset":
                            removed_energies = (total_energy - remaining_energies - eq_rank_energy_loss) / max_rank
                        elif "scaled" in self.sensitivity_loss and "normal_" in self.sensitivity_loss:
                            denom = torch.max(1 - klscaled_reference_point_energy / total_energy, torch.tensor(1e-6, device=dev))
                            removed_energies = (1 - remaining_energies / total_energy) / denom
                        elif "normal" in self.sensitivity_loss:
                            removed_energies = (total_energy - remaining_energies) / total_energy
                        else:
                            removed_energies = (total_energy - remaining_energies) / sv_float.shape[0]
                        
                        # 4. Pull to CPU exactly ONCE, then populate the dictionary
                        removed_energies_np = removed_energies.cpu().numpy()
                        ranks_np = ranks.cpu().numpy()
                        
                        for r, e in zip(ranks_np, removed_energies_np):
                            sensitivity_dict[name][r / max_rank] = e.item()
                
                post_energy_time = datetime.now()

                
                if model_forwards_required:
                    for ratio in self.measurements_points:
                        start_measurement_time = datetime.now()
                        eval_rank = int(factorized_matrix.eq_rank * ratio)
                        factorized_matrix.active_rank = eval_rank
                        
                        seq_replacement = self.lrd_method.create_factorized_sequential(
                            factorized_matrix=factorized_matrix, original_module=module_sub
                        ).to(dev)

                        setattr(base, localname, seq_replacement)
                        get_replacement_time = datetime.now()
                        if self.lrd_method.vision:
                            metric = self._eval_vision(model, original_outputs)
                        else:
                            metric = self._eval_llm(model, original_outputs)
                        if "_klscaled" in self.sensitivity_loss or "_msescaled" in self.sensitivity_loss or "_pplscaled" in self.sensitivity_loss:
                            # FIXED loop variable shadowing bug (ratio -> r)
                            for r, sensitivity in sensitivity_dict[name].items():
                                sensitivity_dict[name][r] = sensitivity * metric
                        else:
                            sensitivity_dict[name][ratio] = metric
                                
                setattr(base, localname, module_sub)
                post_measurement_time = datetime.now()
                
        model.cpu()
        with torch.cuda.device(dev):
            torch.cuda.empty_cache()
                        
        return sensitivity_dict, size_dict
    
    def get_layer_shape_dict(self, model: nn.Module):
        shape_dict = {}
        valid_modules = get_valid_layers(model, self.name_omit, white_list=[])
        for (name, module) in valid_modules:
            shape_dict[name] = (module.in_features, module.out_features)
        return shape_dict

    # ------------------------------------------------------------------
    #  Shared compress / restore helpers  (used by Optuna grid search)
    # ------------------------------------------------------------------

    @staticmethod
    def _compress_model_ratios(module_dict, module_bkup_dict, rank_dict,
                               lrd_method: BaseFactorization):
        """Compress *module_dict* in-place using ratio-valued *rank_dict*."""
        dev = torch.device(torch.cuda.current_device())
        for name, module_sub in module_dict.items():
            if rank_dict[name] == 1.0:
                module_sub.weight.data.copy_(module_bkup_dict[name].weight.data)
            else:
                factorized_matrix = lrd_method.factorize_matrix(
                    module_sub.weight, ratio=rank_dict[name], name=name, verbose=False
                )
                tensor_to_copy = (factorized_matrix.mat_l.to(dev)
                                  @ factorized_matrix.mat_r.to(dev))
                module_sub.weight.data.copy_(tensor_to_copy)

    @staticmethod
    def _compress_model_ranks(module_dict, module_bkup_dict, rank_dict,
                              layer_data, lrd_method: BaseFactorization):
        """Compress *module_dict* in-place using absolute-rank *rank_dict*."""
        dev = torch.device(torch.cuda.current_device())
        for name, module_sub in module_dict.items():
            if name not in rank_dict:
                continue
            dense_weight = module_bkup_dict[name].weight.data
            eq_rank = layer_data[name]["eq_rank"]
            rank = rank_dict[name]
            if rank in (-1, None) or rank >= eq_rank:
                module_sub.weight.data.copy_(dense_weight)
                continue
            ratio = float(rank) / float(eq_rank)
            factorized_matrix = lrd_method.factorize_matrix(
                dense_weight, ratio=ratio, name=name, verbose=False
            )
            tensor_to_copy = (factorized_matrix.mat_l.to(dev)
                              @ factorized_matrix.mat_r.to(dev))
            module_sub.weight.data.copy_(tensor_to_copy)

    @staticmethod
    def _restore_model(module_dict, module_bkup_dict):
        """Restore all weights from the dense backup."""
        for name, module_sub in module_dict.items():
            module_sub.weight.data.copy_(module_bkup_dict[name].weight.data)

    # ------------------------------------------------------------------
    #  Shared vision-model FLOPs estimation
    # ------------------------------------------------------------------

    def get_layer_wise_flops(self, model):
        """Estimate per-layer FLOPs by running a dummy forward pass with *ShapeHook*.

        Returns ``(flops_per_layer, input_shapes)`` dicts keyed by layer name.
        """
        extractor = ShapeHook(
            model=model, name_omit=self.name_omit,
            dump_shape=False, name_prefix="", white_list=[],
        )
        extractor.attach_hooks()
        device = next(model.parameters()).device
        if device.type == "cpu" and torch.cuda.is_available():
            device = torch.device("cuda")
            model = model.to(device)
        dummy_input = torch.randn(20, 3, 224, 224, device=device)
        model(dummy_input)
        input_shapes = dict(extractor.input_shape)
        del dummy_input
        extractor.clear_hooks()
        flops_per_layer = {
            name: s[0] * s[1] * s[2] / 1000 * s[3]
            for name, s in input_shapes.items()
        }
        return flops_per_layer, input_shapes

