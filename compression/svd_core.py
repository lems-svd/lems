import importlib
import time

import torch.nn as nn
from torch.utils import data

from .factorization._interface import BaseFactorization
from .search._interface import BaseSearch
from all_utils.model_io import restore_state_dict, backup_model

import copy


def get_lr_module(module_name: str):
    parent_module = "compression.factorization"
    module = importlib.import_module(f"{parent_module}.{module_name}")
    return getattr(module, f"{module_name.upper()}Factorization")


def get_search_module(module_name: str):
    parent_module = "compression.search"
    module = importlib.import_module(f"{parent_module}.{module_name}")
    return getattr(module, f"{module_name.upper()}Search")


class ModelFactorizer:
    """
    Implementation of Low-Rank Decomposition for compressing the model's weights.
    """

    def __init__(
        self,
        svd_method: str,
        svd_method_args: dict,
        search_method: str,
        search_method_args: dict,
    ) -> None:
        self.svd_method = svd_method
        self.svd_method_args = svd_method_args
        self.search_method = search_method
        self.search_method_args = search_method_args

    def factorize_and_search(
        self,
        model: nn.Module,
        calib_data: data.DataLoader,
        eval_data: data.DataLoader,
        mixup_fn,
        calib_dataset_name: str,
        name_omit: list = [],
        blockwise_search: bool = False,
        visualization_mode: bool = False,
        calib_ds_seed : int = -1,
    ):
        """
        This function applies SVD decomposition to the models layers.
        """
        print(f"{' Compressing model ':=^115}")
        start_time = time.time()
        LrdModule = get_lr_module(self.svd_method)
        SearchMethod = get_search_module(self.search_method)

        self.svd_method_args["calib_dataset_name"] = calib_dataset_name
        if "_shared" in self.svd_method:
            self.svd_method_args = self.svd_method_args | {
                "use_shared_basis": True,
                "shared_group_size": 2,
                "shared_part": ["q", "k", "v", "up", "gate"],
                "private_part": ["down", "o"],
            }
            print("Using shared basis factorization with the following settings:"
                  f"\nShared group size: {self.svd_method_args.get('shared_group_size', 'N/A')}")
        lrd_method: BaseFactorization = LrdModule(**self.svd_method_args)
        search_method: BaseSearch = SearchMethod(
            name_omit=name_omit,
            eval_data=eval_data,
            mixup_fn=mixup_fn,
            **self.search_method_args,
        )

        backup = backup_model(model)
        if search_method.requires_decomposed_model_for_search or not lrd_method.post_search_calibration:
            lrd_method.factorization_computations(
                model=model, name_omit=name_omit, calib_data=calib_data, mixup_fn=mixup_fn
            )
            decomposition_time = (time.time() - start_time) / 60
            print(f"\nTook {decomposition_time:.3} min to decompose model\n")
            model = restore_state_dict(model, backup)
        else:
            print(f"Skipping initial factorization for search as {search_method.__class__.__name__} "
                  f"does not require a decomposed model before search and {lrd_method.__class__.__name__}"
                  f" requires the final ranks for its correct calibration.")
        
        search_start_time = time.time()
        search_method.initialize_search(lrd_method, model)
        layerwise_rank_dict = search_method.search(model)
        search_time = (time.time() - search_start_time) / 60
        print(f"\nTook {search_time:.3} min to search\n")
        if visualization_mode:
            if hasattr(search_method, 'sensitivity_dict'):
                import pickle, os
                from all_utils.model_io import get_model_name
                model_name = get_model_name(model)
                setting_str = model_name + '_' + self.svd_method + '_' + self.search_method + '_' + calib_dataset_name + '_' + str(self.search_method_args.get('sensitivity_loss', 'none')) + '_' + str(self.search_method_args.get('measurements_points', 'none') + '_' + str(calib_ds_seed))
                out_folder = './paper_llm_images/sensitivity_data/'
                os.makedirs(out_folder, exist_ok=True)
                with open(f'{out_folder}sensitivity_dict_{setting_str}.pkl', 'wb') as f:
                    pickle.dump(search_method.sensitivity_dict, f)
        
        print(f"\n{' Post search statistics recalibration  ':=^115}\n")
        if lrd_method.post_search_calibration:
            # recalibration is always block-wise with progressive compression
            # as it is ineffective otherwise. This is also the reason why we
            # do not make use of cached values here.
            self.svd_method_args["use_cache"] = False
            self.svd_method_args["calibration_ranks"] = layerwise_rank_dict
            self.svd_method_args["blockwise_factorization"] = True
            self.svd_method_args["progressive_compression"] = True
            del lrd_method
            
            lrd_method: BaseFactorization = LrdModule(**self.svd_method_args)
            lrd_method.factorization_computations(
                model=model, name_omit=name_omit, calib_data=calib_data, mixup_fn=mixup_fn
            )
            model = restore_state_dict(model, backup)
            calibration_time = (time.time() - start_time) / 60
            print(f"\nTook {calibration_time:.3} min to calibrate after search\n")


        # inplace factorization based on layerwise rank dict
        lrd_method.factorize_model(model, layerwise_rank_dict, name_omit=name_omit)

        num_params = sum(p.numel() for p in model.parameters())
        print(f"\n\nNumber of parameters in model: {num_params}")
        total_time = (time.time() - start_time) / 60
        print(f"\nTook {total_time:.3} min to compress model\n")
        return total_time, search_time, num_params, model, layerwise_rank_dict