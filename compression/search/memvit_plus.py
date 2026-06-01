import torch
import torch.nn as nn
import copy
import gc
from collections import OrderedDict

from ..factorization._interface import (
    BaseFactorization,
    get_valid_layers,
)
from .memvit import MEMVITSearch
from .lems import get_depth_multiplier, optuna_inner_search
from ..factorization._interface import _find_decoder_layers


class MEMVIT_PLUSSearch(MEMVITSearch):
    """
    MemViT greedy search with LEMS-style cross-layer depth bias and
    optional Optuna fitting of the harmonicv2 bias parameters.

    Inherits the full greedy loop from :class:`MEMVITSearch` and injects
    bias via the :meth:`_score_candidate` hook.
    """

    def __init__(
        self,
        eval_data,
        mixup_fn,
        name_omit=[],
        ratio_target=0.5,
        sensitivity_loss="kl",
        target_metric="params",
        measurements_points="0.1-0.9",
        lower_bound=-1.0,
        enforce_rank_multiples_of=None,
        step_count_candidates=None,
        max_target_overshoot=0.02,
        crosslayer_term="harmonicv2",
        halpha=0.0,
        hgamma=0.0,
        one_shot=False,
        optuna_trials=10,
        alpha_range=(0.0, 3.0),
        gamma_range=(0.0, 7.0),
        *args,
        **kwargs,
    ):
        super().__init__(
            eval_data=eval_data,
            mixup_fn=mixup_fn,
            name_omit=name_omit,
            ratio_target=ratio_target,
            sensitivity_loss=sensitivity_loss,
            target_metric=target_metric,
            measurements_points=measurements_points,
            lower_bound=lower_bound,
            enforce_rank_multiples_of=enforce_rank_multiples_of,
            step_count_candidates=step_count_candidates,
            max_target_overshoot=max_target_overshoot,
            *args,
            **kwargs,
        )

        self.crosslayer_term = crosslayer_term
        self.alpha = halpha
        self.hgamma = hgamma
        self.one_shot = one_shot

        self.optuna_trials = optuna_trials
        self.alpha_range = alpha_range
        self.gamma_range = gamma_range

        if self.crosslayer_term == "harmonicv2" and self.hgamma < 0:
            raise ValueError("hgamma must be non-negative")

    @property
    def requires_decomposed_model_for_search(self):
        return False

    # ------------------------------------------------------------------
    #  Initialization – adds layer bias on top of base
    # ------------------------------------------------------------------

    def initialize_search(
        self, lrd_method: BaseFactorization, model: nn.Module, spec_tensor=None
    ):
        super().initialize_search(lrd_method, model, spec_tensor)
        self._refresh_layer_bias(model)

    # ------------------------------------------------------------------
    #  Depth bias helpers
    # ------------------------------------------------------------------

    def _get_depth_multiplier(self, current_block: int, total_blocks: int) -> float:
        return get_depth_multiplier(
            current_block=current_block,
            total_blocks=total_blocks,
            crosslayer_term=self.crosslayer_term,
            alpha=self.alpha,
            gamma=self.hgamma,
            is_vision=self.lrd_method.vision,
        )

    def _refresh_layer_bias(self, model: nn.Module):
        stages, _ = _find_decoder_layers(model)
        layers_per_block = sum(
            1 for mod in stages[0].modules() if isinstance(mod, nn.Linear)
        )
        total_blocks = max(len(self.layer_data) // layers_per_block, 1)

        self.layer_bias = OrderedDict()
        for i, layer_name in enumerate(self.layer_data.keys()):
            current_block = i // layers_per_block
            self.layer_bias[layer_name] = self._get_depth_multiplier(
                current_block, total_blocks
            )

    # ------------------------------------------------------------------
    #  Hook override – inject bias into greedy loop
    # ------------------------------------------------------------------

    def _score_candidate(self, layer_name: str, base_loss: float) -> float:
        return float(base_loss) * float(self.layer_bias[layer_name])

    # ------------------------------------------------------------------
    #  Optuna grid search with step-count sweeps
    # ------------------------------------------------------------------

    def grid_search(self, model: nn.Module):
        import optuna

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

        original_max_iter = self.max_iter
        original_schedule_gamma = self.gamma

        global_best = None

        for step_count in self.step_count_candidates:
            print("\n" + "=" * 80)
            print(f"Starting bias search for step_count={step_count}")
            self._set_schedule_from_step_count(step_count)

            def objective(trial):
                self._restore_model(module_dict, module_bkup_dict)

                alpha = trial.suggest_float("alpha", self.alpha_range[0], self.alpha_range[1])
                gamma = trial.suggest_float("gamma", self.gamma_range[0], self.gamma_range[1])
                self.alpha = alpha
                self.hgamma = gamma
                self._refresh_layer_bias(model)

                search_ranks = self.single_search(model)
                realized_ratio = self.last_search_stats["realized_ratio"]

                trial.set_user_attr("search_ranks", copy.deepcopy(search_ranks))
                trial.set_user_attr("realized_ratio", realized_ratio)
                trial.set_user_attr("step_count", step_count)
                trial.set_user_attr("steps_used", self.last_search_stats["steps_used"])

                if realized_ratio > self.ratio_target + self.max_target_overshoot:
                    print(
                        f"Trial {trial.number} rejected: realized_ratio={realized_ratio:.4f} "
                        f"> allowed={self.ratio_target + self.max_target_overshoot:.4f}"
                    )
                    raise optuna.TrialPruned(
                        f"Compression too weak: ratio {realized_ratio:.4f}"
                    )

                self._compress_model_ranks(
                    module_dict, module_bkup_dict, search_ranks,
                    self.layer_data, self.lrd_method,
                )

                if self.lrd_method.vision:
                    metric = self._eval_vision(model, original_outputs)
                else:
                    metric = self._eval_llm(model, original_outputs)

                print(
                    f"Trial {trial.number}: "
                    f"step_count={step_count}, alpha={alpha:.4f}, gamma={gamma:.4f}, "
                    f"ratio={realized_ratio:.4f}, metric={metric:.4f}"
                )
                return metric

            best_trial = optuna_inner_search(
                objective_fn=objective,
                n_trials=self.optuna_trials,
            )

            if best_trial is None:
                print(f"No feasible completed trials for step_count={step_count}.")
                continue

            local_result = {
                "metric": best_trial.value,
                "alpha": best_trial.params["alpha"],
                "gamma": best_trial.params["gamma"],
                "step_count": best_trial.user_attrs["step_count"],
                "realized_ratio": best_trial.user_attrs["realized_ratio"],
                "search_ranks": best_trial.user_attrs["search_ranks"],
            }

            print(
                f"Best feasible trial for step_count={step_count}: "
                f"metric={local_result['metric']:.4f}, "
                f"alpha={local_result['alpha']:.4f}, "
                f"gamma={local_result['gamma']:.4f}, "
                f"ratio={local_result['realized_ratio']:.4f}"
            )

            if global_best is None or local_result["metric"] < global_best["metric"]:
                global_best = local_result

        self.max_iter = original_max_iter
        self.gamma = original_schedule_gamma

        if global_best is None:
            self._restore_model(module_dict, module_bkup_dict)
            raise RuntimeError(
                "All Optuna trials were rejected/pruned because they overshot the target "
                f"by more than {self.max_target_overshoot:.2%}."
            )

        self.alpha = global_best["alpha"]
        self.hgamma = global_best["gamma"]
        self._set_schedule_from_step_count(global_best["step_count"])
        self._refresh_layer_bias(model)

        print(
            f"\nBest metric {global_best['metric']} found with "
            f"step_count={global_best['step_count']}, "
            f"alpha={global_best['alpha']}, gamma={global_best['gamma']}, "
            f"ratio={global_best['realized_ratio']:.4f}"
        )

        self._restore_model(module_dict, module_bkup_dict)

        del model_bkup
        gc.collect()
        return global_best["search_ranks"]

    def search(self, model: nn.Module):
        if self.one_shot or self.crosslayer_term == "constant":
            return self.single_search(model)
        return self.grid_search(model)