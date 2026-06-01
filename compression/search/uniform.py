from torch import nn

from ._interface import BaseSearch


class UNIFORMSearch(BaseSearch):
    def __init__(self, eval_data, mixup_fn, name_omit=[], ratio_target=0.5, enforce_rank_multiples_of=0, **kwargs):
        self.name_omit = name_omit
        self.sensitivity_dict = {}  # kept for svd_core.py visualisation check
        self.ratio_target = ratio_target
        self.enforce_rank_multiples_of = enforce_rank_multiples_of
    
    @property
    def requires_decomposed_model_for_search(self):
        return False

    def search(self, model: nn.Module):
        default_param_ratio = 1.0
        layer_compression_dict = {
            name: self.ratio_target
            for name, module_sub in model.named_modules()
            if all(omit not in name for omit in self.name_omit) and isinstance(module_sub, nn.Linear)
        } # TODO: use centralized valid Linear functions.

        # replace name omit layer compression with 1.0
        for name, module in model.named_modules():
            if name not in layer_compression_dict:
                if isinstance(module, nn.Linear):
                    layer_compression_dict[name] = default_param_ratio

        return layer_compression_dict
