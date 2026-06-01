import torch
from torch import nn

from ._sensitivity_base import SensitivityBasedSearch
from ..factorization._interface import BaseFactorization
from ..factorization._interface import _find_decoder_layers
from ..factorization._interface import get_valid_layers
from .lems import compress_model, restore_model
import copy

def linearly_decreasing_values(n, beta, c):
    values = [beta * (i/(n-1) - 0.5) + c for i in range(0, n)]
    return values[::-1]  # Reverse to have decreasing order

class ATPSearch(SensitivityBasedSearch):
    """
    @inproceedings{
        huang2025determining,
        title={Determining Layer-wise Sparsity for Large Language Models Through a Theoretical Perspective},
        author={Weizhong Huang and Yuxin Zhang and Xiawu Zheng and Fei Chao and Rongrong Ji},
        booktitle={Forty-second International Conference on Machine Learning},
        year={2025},
        url={https://openreview.net/forum?id=otNB7BzsiR}
    }
    """
    def __init__(self, ratio_target=0.5, beta=0.01, *args, **kwargs):
        # ATP only uses _eval_llm / _precompute_original_outputs from the
        # base class.  Sensitivity profiling is skipped entirely, so we
        # hardcode sensitivity_loss and a minimal measurements_points.
        super().__init__(*args, sensitivity_loss="ppl", measurements_points="0.1", **kwargs)
        self.lrd_method = None
        self.ratio_target = ratio_target
        self.beta = beta
    
    @property
    def requires_decomposed_model_for_search(self):
        return True

    def initialize_search(
            self, lrd_method: BaseFactorization, model: nn.Module, spec_tensor=None
    ):
        # ATP only makes use of the eval function of sensitivity based 
        # search. The other componetns of it - especially the individual 
        # layer profiling - are not required for ATP. This is why we fully
        # overwrite the initialize_search function.
        self.lrd_method = lrd_method
        self.n_steps = 10

    def search(self, model: nn.Module):
        # Backup original model (uncompressed)
        model_bkup = copy.deepcopy(model)
        valid_modules_tuples = get_valid_layers(model_bkup, self.name_omit)
        module_bkup_dict = {name: module for name, module in valid_modules_tuples}

        # Put model on device
        dev = torch.device(torch.cuda.current_device())
        model = model.to(dev)
        valid_modules_tuples = get_valid_layers(model, self.name_omit)
        module_dict = {name: module for name, module in valid_modules_tuples}

        layer_compression_dict = {name: self.ratio_target for name in module_dict.keys()}
        default_param_ratio = 1.0

        max_headroom = min(1-self.ratio_target, self.ratio_target)
        valid_betas = [max_headroom*2/self.n_steps *i for i in range(0, self.n_steps+1)]
        print("Valid betas to try:", valid_betas)

        original_outputs = self._precompute_original_outputs(model)

        best_metric = 1e12
        best_compression = {}

        for beta in valid_betas:
            stages, _ = _find_decoder_layers(model)
            ratios = linearly_decreasing_values(n = len(stages), c=self.ratio_target, beta=beta)
            print("Testing compression ratios:", ratios)
            # replace name omit layer compression with 1.0
            for name, module in model.named_modules():
                if name not in layer_compression_dict:
                    if isinstance(module, nn.Linear):
                        layer_compression_dict[name] = default_param_ratio

            for i in range(len(stages)):
                for layername, param_ratio in layer_compression_dict.items():
                    if f".{i}." in layername:
                        layer_compression_dict[layername] = ratios[i]
            
            # Compress to current configuration. Undoes compression for ratio=1.0
            compress_model(module_dict, module_bkup_dict, layer_compression_dict, self.lrd_method)
            # Evaluate
            metric = self._eval_llm(model, original_outputs)
            print(metric)
            if metric < best_metric:
                best_metric = metric
                best_beta = beta
                best_compression = copy.deepcopy(layer_compression_dict)
        print(f"Best beta: {best_beta}, Best metric: {best_metric}")
        restore_model(module_dict=module_dict, module_bkup_dict=module_bkup_dict)
        return best_compression

