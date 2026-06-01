import torch
from torch import nn
from collections import defaultdict

from ..factorization._interface import BaseFactorization
from ._interface import BaseSearch

from ..factorization._interface import Hookstuff
import torch.nn.functional as F
import gc
from ..factorization._interface import get_valid_layers

class LayerwiseSensitivityHook(Hookstuff):
    """
    A hook to collect sensitivity data for each layer during the forward pass.
    """
    def __init__(self, lrd_method, ratio_target, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sensitivity_data = {}
        self.lrd_method = lrd_method
        self.ratio_target = ratio_target
    
    @property
    def requires_decomposed_model_for_search(self):
        return True
    
    def _hook_fn(self, layer_name, last_feat=False):
        def get_reconstruction_error(module, input, output):
            
            activation = input[0].detach().float()  # Get the input to the layer
            #if isinstance(module, nn.Linear):
            factorized_matrix = self.lrd_method.factorize_matrix(
                    name=layer_name, matrix=module.weight, ratio=1.0
                )
            xW = F.linear(activation,module.weight)  # xW is the output of the linear layer
            rank = int(factorized_matrix.eq_rank*self.ratio_target)
            # TODO: change to support ConvAsLinear.
            W_reconstruct = factorized_matrix.mat_l[:, :rank] @ factorized_matrix.mat_r[:rank, :]
            xW_reconstruct = F.linear(activation, W_reconstruct.to(activation.device).to(activation.dtype))  # Reconstruct using only the first singular value
            # frobenius norm
            loss = torch.sqrt(torch.sum((xW - xW_reconstruct) ** 2))
            #loss = torch.sqrt(torch.sum(S[:len(S)*self.ratio_target]**2))
            #loss = torch.sqrt(torch.sum(S[int(len(S)*self.ratio_target):]))
            if layer_name in self.profile.keys():
                # If the layer is already in the profile, accumulate the loss
                self.profile[layer_name] += loss
            else:
                self.profile[layer_name] = loss
            activation = xW = W_reconstruct = xW_reconstruct = loss = factorized_matrix = None
            del activation, xW, W_reconstruct, xW_reconstruct, loss, factorized_matrix
            torch.cuda.empty_cache()
            gc.collect()

        return get_reconstruction_error

class SVD_LLMV2Search(BaseSearch):
    """
    Implements the heterogeneous compression ratio allocation from SVD-LLMv2 (Algorithm 1).
    
    Instead of a sensitivity search, this method analytically computes the optimal
    per-layer compression ratio based on each layer's theoretical SVD reconstruction error.

    @inproceedings{wang-etal-2025-svd-llm,
        title = "{SVD}-{LLM} V2: Optimizing Singular Value Truncation for Large Language Model Compression",
        author = "Wang, Xin  and Alam, Samiul and Wan, Zhongwei  and Shen, Hui and Zhang, Mi",
        booktitle = "Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
        year = "2025",
        url = "https://aclanthology.org/2025.naacl-long.217/",
        doi = "10.18653/v1/2025.naacl-long.217",
        pages = "4287--4296",
        ISBN = "979-8-89176-189-6",
    }
    """
    def __init__(self, eval_data, ratio_target=0.5, *args, **kwargs):
        # We only need eval_data for calibration and ratio_target for the global goal.
        super().__init__(eval_data=eval_data, *args, **kwargs)
        self.ratio_target = ratio_target  # Global target compression ratio
    
    @property
    def requires_decomposed_model_for_search(self):
        return True

    def _group_layers_by_type(self, model: nn.Module) -> dict:
        """Groups linear layers based on common naming conventions."""
        groups = defaultdict(list)
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if any(n in name for n in self.name_omit):
                    continue
                # Add your model's specific naming patterns here
                if 'q_proj' in name: groups['q_proj'].append(name)
                elif 'qkv' in name: groups['qkv'].append(name)
                elif 'k_proj' in name: groups['k_proj'].append(name)
                elif 'v_proj' in name: groups['v_proj'].append(name)
                elif 'o_proj' in name: groups['o_proj'].append(name)
                elif 'gate_proj' in name: groups['mlp_gate'].append(name)
                elif 'up_proj' in name: groups['mlp_up'].append(name)
                elif '.fc1' in name: groups['mlp_up'].append(name)  # For models like BERT/opt
                elif 'down_proj' in name: groups['mlp_down'].append(name)
                elif '.fc2' in name: groups['mlp_down'].append(name)  # For models like BERT/opt
                elif 'out_proj' in name: groups['out_proj'].append(name)

                else: groups['other'].append(name)
        return groups

    def initialize_search(self, lrd_method: BaseFactorization, model: nn.Module, *args, **kwargs):
        """
        Calculates and stores the per-layer compression ratios based on Algorithm 1.
        """
        print("Initializing SVD-LLMv2 search: Allocating heterogeneous ratios...")
        self.lrd_method = lrd_method
        dev = device = torch.device(torch.cuda.current_device())
        model = model.to(dev) #.float() # Move model to GPU once
        self.frobenius_scores = {}
        # activation_extractor = LayerwiseSensitivityHook(model=model, name_omit=self.name_omit, lrd_method=self.lrd_method, ratio_target=self.ratio_target, dump_shape=False)
        # activation_extractor.attach_hooks()
        # with torch.no_grad():
        #     for batch in self.eval_data:
        #         batch = {k: v.to(device) for k, v in batch.items()}
        #         _ = model(**batch)
        #         break
        # self.frobenius_scores = activation_extractor.profile
        # activation_extractor.clear_hooks()

        copied_modules = get_valid_layers(model, name_omit=self.name_omit)
        for layer_name, module in copied_modules:
            
            factorized_matrix = self.lrd_method.factorize_matrix(
                    name=layer_name, matrix=module.weight, ratio=1.0
                )
            rank =  int(factorized_matrix.eq_rank * self.ratio_target)
            removed_energy = sum(torch.pow(factorized_matrix.singular_values[rank:], 1))
            
            self.frobenius_scores[layer_name] = removed_energy

        print("Ratio allocation complete.")

    def search(self, model: nn.Module) -> dict:
        print("=== SVD-LLMv2 search complete, returning analytical ratios... ===")

        layer_groups = self._group_layers_by_type(model)
        ratios = {}

        # Your public API uses KEEP ratio.
        target_keep = float(self.ratio_target)
        target_compress = 1.0 - target_keep

        for group_name, layer_names in layer_groups.items():
            if not layer_names:
                continue

            print(f"Processing group: {group_name} with {len(layer_names)} layers")

            group_losses = torch.stack([
                self.frobenius_scores[name].float() for name in layer_names
            ])

            # Paper-faithful scoring rule.
            scores = 1.0 / torch.log(group_losses + 1e-12)
            score_sum = scores.sum()

            # Fallback: if scoring becomes invalid, keep the uniform target.
            if (not torch.isfinite(score_sum)) or score_sum <= 0:
                for name in layer_names:
                    ratios[name] = target_keep
                continue

            # Allocate COMPRESSION ratios first, exactly in the paper's direction:
            # low loss -> larger compression
            compress = len(layer_names) * target_compress * scores / score_sum

            # Convert to KEEP ratios for the rest of your framework:
            keep = 1.0 - compress

            # Return keep ratios
            for name, k in zip(layer_names, keep):
                ratios[name] = float(k.item())

        return ratios