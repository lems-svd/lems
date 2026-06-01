from typing import List

from torch import nn

from ..factorization._interface import BaseFactorization

# TODO, make correct
class BaseSearch:
    def __init__(self, eval_data, mixup_fn, name_omit: List[str] = None, **kwargs):
        self.eval_data = eval_data
        self.name_omit = name_omit
        self.mixup_fn = mixup_fn
    
    @property
    def requires_decomposed_model_for_search(self):
        raise NotImplementedError("Subclasses should implement this attribute.")
        return False

    def search(self, model: nn.Module):
        raise NotImplementedError("Subclasses should implement this method.")

    def search_blockwise(self, model: nn.Module, stage_name: str, calib_data=None):
        raise NotImplementedError("Subclasses should implement this method.")

    def initialize_search(
        self, lrd_method: BaseFactorization, model: nn.Module, spec_tensor=None
    ):
        # Initialize the search parameters
        self.lrd_method = lrd_method

    def get_model_blocks(self, model, stage_name):
        # create an iterable list with all model blocks and the layer names within it.
        def get_valid_layer_names(block: nn.Module, stage_idx: int, block_idx: int):
            layer_names = []
            copied_modules = {
                name: module_sub
                for name, module_sub in block.named_modules()
                if all(omit not in name for omit in self.name_omit)
                and isinstance(module_sub, nn.Linear)
            }
            # TODO: use centralized valid Linear functions.
            for name, module_sub in copied_modules.items():
                if module_sub.out_features < 10:
                    continue  # for some head matrix, such as image-text match head

                if stage_idx == -1:
                    full_name = f"blocks.{block_idx}.{name}"
                else:
                    full_name = (
                        f"{stage_name}.{stage_idx}."
                        f"blocks.{block_idx}.{name}"
                    )
                layer_names.append((full_name, name))
            return layer_names

        model_blocks = []
        layer_names = []
        if hasattr(model, stage_name):
            # for every model stage
            for current_stage_idx, stage in enumerate(getattr(model, stage_name)):
                # for every block in a model stage
                for current_block_idx, block in enumerate(
                    getattr(stage, "blocks", stage)
                ):
                    model_blocks.append(block)
                    layer_names.append(
                        get_valid_layer_names(
                            block, current_stage_idx, current_block_idx
                        )
                    )
        else:
            # for every block in the model
            for current_block_idx, block in enumerate(getattr(model, "blocks", model)):
                model_blocks.append(block)
                layer_names.append(get_valid_layer_names(block, -1, current_block_idx))

        return model_blocks, layer_names
