from torch import nn

from ._interface import BaseSearch
import json


class LOADCONFIGSearch(BaseSearch):
    def __init__(
        self,
        eval_data,
        mixup_fn,
        name_omit=[],
        layer_compression_json_path='path/to/layer_compression.json',
        **kwargs,
    ):
        self.name_omit = name_omit
        self.sensitivity_dict = {}  # kept for svd_core.py visualisation check
        self.layer_compression_json_path = layer_compression_json_path
    
    @property
    def requires_decomposed_model_for_search(self):
        return False

    def search(self, model: nn.Module):
        default_param_ratio = 1.0
        layer_compression_dict = {
            name: default_param_ratio
            for name, module_sub in model.named_modules()
            if all(omit not in name for omit in self.name_omit)
            and isinstance(module_sub, nn.Linear)
        }  # TODO: use centralized valid Linear functions.
        
        with open(self.layer_compression_json_path, 'r') as f:
            layer_compression_data = json.load(f)

        if "dobi" in self.layer_compression_json_path:
            layer_compression_data = {name: int(round(ratio*2)) for name, ratio in layer_compression_data.items()}
        layer_compression_dict.update(layer_compression_data)

        # replace name omit layer compression with 1.0
        # for name, module in model.named_modules():
        #     if name not in layer_compression_dict:
        #         if isinstance(module, nn.Linear):
        #             layer_compression_dict[name] = default_param_ratio

        return layer_compression_dict

    def search_blockwise(self, model: nn.Module, stage_name: str, calib_data=None):
        raise NotImplementedError(
            "LOADCONFIGSearch does not support blockwise search. Use search method instead."
        )
