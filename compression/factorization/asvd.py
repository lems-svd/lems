import gc

import torch
from tqdm import tqdm

from ._interface import BaseFactorization
from ._interface import FactorizedMatrix
from ._interface import Hookstuff


class ASVD_Hook(Hookstuff):
    def _hook_fn(self, layer_name, last_feat=False):
        def get_scaling_mat(module, input, output):
            x = self._reshape_input(input[0].detach().float())
            if self._maybe_record_shape(layer_name, x, module):
                return
            outpro_sum = x.abs().amax(dim=-2).detach().amax(-2)

            if layer_name not in self.profile:  # First run through each layer
                self.profile[layer_name] = outpro_sum
            else:
                self.profile[layer_name] += outpro_sum

            del module, input, outpro_sum

        return get_scaling_mat


class ASVDFactorization(BaseFactorization):
    def __init__(self, alpha, vision, *args, **kwargs):
        super().__init__(vision=vision, *args, **kwargs)
        self.alpha = alpha
        self.vision = vision
    
    @torch.no_grad()
    def _compute_scaling(self, model, hook_module, name_prefix, calib_data, name_omit, mixup_fn=None, white_list=[], tqdm_message="Gathering "):
        dev = self.dev

        extractor = ASVD_Hook(hook_module, name_omit, False, name_prefix=name_prefix, white_list=white_list)
        extractor.attach_hooks()
        if self.vision:
            with torch.no_grad():
                for samples, targets in calib_data:
                    model_inps, targets = mixup_fn(samples, targets) if mixup_fn is not None else (samples, targets)
                    model_inps = model_inps.to(self.dev)
                    model(model_inps)
                    del model_inps, targets
            # get shapes of layer inputs
            shapes_getter = ASVD_Hook(model, name_omit, True, white_list=white_list)
            shapes_getter.attach_hooks()
            dummy_input = torch.randn(20, 3, 224, 224).to(self.dev)
            model(dummy_input)
            shapes_getter.clear_hooks()
            for key, value in shapes_getter.input_shape.items():
                self.input_shapes[key] = value
            del shapes_getter, dummy_input
        else:
            with torch.no_grad():
                with torch.autocast(device_type="cuda", enabled=False):
                    for batch in tqdm(calib_data, desc=tqdm_message):
                        batch = {k: v.to(dev) for k, v in batch.items()}
                        _ = model(**batch)
            
        extractor.clear_hooks()
        for key, value in extractor.profile.items():
            self.scaling_dict[key] = value.cpu()
        del extractor

        return

    def _factorize_matrix(self, matrix, name, eq_rank, rank, dev, verbose=False):
        raw_profile = self.scaling_dict[name].to(dev)
        scale_diag = raw_profile**self.alpha + 1e-6

        mat_scaled = matrix * scale_diag.view(1, -1)

        u, s, vh = torch.svd_lowrank(mat_scaled, q=rank)
        s_val = torch.sqrt(torch.diag(s))  # half singular value
        vh = (vh / scale_diag.view(-1, 1)).t()

        s_val = torch.sqrt(torch.diag(s))  # half singular value
        mat_l = u @ s_val
        mat_r = s_val @ vh
        mat_l = mat_l[:, :rank]
        mat_r = mat_r[:rank, :]

        return FactorizedMatrix(
            mat_l=mat_l.cpu(),  # Left singular vectors
            mat_r=mat_r.cpu(),  # Right singular vectors
            eq_rank=eq_rank,  # Equivalent rank
            active_rank=rank,  # Active rank
            singular_values=s
        )
