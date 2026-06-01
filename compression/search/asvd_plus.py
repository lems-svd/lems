from torch import nn
import torch
import copy
import gc

from ..factorization._interface import BaseFactorization, get_valid_layers, get_eq_rank
from .asvd import ASVDSearch
from .lems import get_depth_multiplier, optuna_inner_search
from ..factorization._interface import _find_decoder_layers


class ASVD_PLUSSearch(ASVDSearch):
    """
    ASVD with LEMS-style cross-layer depth bias and optional Optuna tuning.

    Inherits the binary-search core from :class:`ASVDSearch` and overrides
    ``_build_sensitivity_list`` to apply depth-aware importance scaling and
    optional rank-multiple filtering.
    """

    def __init__(
        self,
        ratio_target=0.5,
        sensitivity_loss="energy2_normal_klscaled",
        measurements_points="0.1",
        crosslayer_term="harmonicv2",
        halpha=0.0,
        hgamma=0.0,
        one_shot=False,
        min_ratio=0.3,
        max_ratio=1.0,
        optuna_trials=10,
        alpha_range=(0.0, 3.0),
        gamma_range=(0.0, 7.0),
        enforce_rank_multiples_of=None,
        *args,
        **kwargs,
    ):
        super().__init__(
            ratio_target=ratio_target,
            sensitivity_loss=sensitivity_loss,
            measurements_points=measurements_points,
            min_ratio=min_ratio,
            max_ratio=max_ratio,
            *args,
            **kwargs,
        )
        self.crosslayer_term = crosslayer_term
        self.alpha = halpha
        self.gamma = hgamma
        self.one_shot = one_shot

        self.optuna_trials = optuna_trials
        self.alpha_range = alpha_range
        self.gamma_range = gamma_range

        self.enforce_rank_multiples_of = enforce_rank_multiples_of

        if self.crosslayer_term == "harmonicv2" and self.gamma < 0:
            raise ValueError("hgamma must be non-negative")

    @property
    def requires_decomposed_model_for_search(self):
        return False

    def initialize_search(
        self, lrd_method: BaseFactorization, model: nn.Module, spec_tensor=None
    ):
        super().initialize_search(lrd_method, model, spec_tensor)
        self.shape_dict = self.get_layer_shape_dict(model)

    # ------------------------------------------------------------------
    #  Override: depth-biased sensitivity list
    # ------------------------------------------------------------------

    def _build_sensitivity_list(self, model: nn.Module):
        stages, _ = _find_decoder_layers(model)
        layers_per_block = sum(
            1 for mod in stages[0].modules() if isinstance(mod, nn.Linear)
        )
        total_blocks = max(len(self.sensitivity_dict) // layers_per_block, 1)

        sensitivity_list = []
        for i, (layername, v) in enumerate(self.sensitivity_dict.items()):
            current_block = i // layers_per_block
            depth_multiplier = get_depth_multiplier(
                current_block=current_block,
                total_blocks=total_blocks,
                crosslayer_term=self.crosslayer_term,
                alpha=self.alpha,
                gamma=self.gamma,
                is_vision=self.lrd_method.vision,
            )

            if self.enforce_rank_multiples_of is not None:
                n, m = self.shape_dict[layername]
                eq_rank = get_eq_rank(n, m)
                v = {
                    k: score for k, score in v.items()
                    if int(k * eq_rank) % self.enforce_rank_multiples_of == 0
                }

            for param_ratio, score in v.items():
                if param_ratio >= self.max_ratio or param_ratio < self.min_ratio:
                    continue
                sensitivity_list.append(
                    (layername, param_ratio, score * depth_multiplier)
                )

        return sensitivity_list

    # ------------------------------------------------------------------
    #  Optuna grid search
    # ------------------------------------------------------------------

    def grid_search(self, model: nn.Module):
        """Fit harmonicv2 alpha/gamma via Optuna."""
        self.sensitivity_loss = "kl"
        if self.crosslayer_term != "harmonicv2":
            print("WARNING: crosslayer_term changed to harmonicv2 for optuna search.")
            self.crosslayer_term = "harmonicv2"

        dev = torch.device(torch.cuda.current_device())

        model_bkup = copy.deepcopy(model)
        module_bkup_dict = dict(get_valid_layers(model_bkup, self.name_omit))

        model = model.to(dev)
        module_dict = dict(get_valid_layers(model, self.name_omit))

        original_outputs = self._precompute_original_outputs(model)

        def objective(trial):
            alpha = trial.suggest_float("alpha", self.alpha_range[0], self.alpha_range[1])
            gamma = trial.suggest_float("gamma", self.gamma_range[0], self.gamma_range[1])
            self.alpha = alpha
            self.gamma = gamma

            search_ranks = self.single_search(model)
            self._compress_model_ratios(module_dict, module_bkup_dict, search_ranks, self.lrd_method)

            if self.lrd_method.vision:
                metric = self._eval_vision(model, original_outputs)
            else:
                metric = self._eval_llm(model, original_outputs)

            trial.set_user_attr("search_ranks", copy.deepcopy(search_ranks))
            print(f"Trial {trial.number}: alpha={alpha:.4f}, gamma={gamma:.4f}, metric={metric:.4f}")
            return metric

        best_trial = optuna_inner_search(
            objective_fn=objective,
            n_trials=self.optuna_trials,
        )

        self.alpha = best_trial.params["alpha"]
        self.gamma = best_trial.params["gamma"]
        best_ranks = best_trial.user_attrs["search_ranks"]

        print(f"\nBest metric {best_trial.value} found with alpha={self.alpha}, gamma={self.gamma}")
        self._restore_model(module_dict, module_bkup_dict)

        del model_bkup
        gc.collect()
        return best_ranks

    def search(self, model: nn.Module):
        if self.one_shot or self.crosslayer_term == "constant":
            return self.single_search(model)
        return self.grid_search(model)