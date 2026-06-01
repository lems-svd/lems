import torch
import torch.nn as nn
import math
from collections import OrderedDict

from ..factorization._interface import BaseFactorization, get_eq_rank
from ._sensitivity_base import SensitivityBasedSearch


def tau_schedule(N0: int, N_target: int, t: int, gamma: float) -> float:
    """Exponential scheduling function (Eq. 9 of the MemViT paper)."""
    return N_target + (N0 - N_target) * math.exp(-t / gamma)


class MEMVITSearch(SensitivityBasedSearch):
    """
    MemViT-style greedy rank search.

    At each iteration the layer whose rank reduction incurs the lowest
    (possibly bias-adjusted) loss is selected.  Subclasses can override
    :meth:`_score_candidate` to inject cross-layer bias.
    """

    def __init__(
        self,
        eval_data,
        mixup_fn,
        name_omit=[],
        ratio_target=0.5,
        sensitivity_loss="energy1",
        target_metric="params",
        measurements_points="0.1-0.9",
        lower_bound=-1.0,
        enforce_rank_multiples_of=None,
        step_count_candidates=None,
        max_target_overshoot=0.02,
        *args,
        **kwargs,
    ):
        self.memvit_sensitivity_loss = sensitivity_loss
        super().__init__(
            eval_data=eval_data,
            mixup_fn=mixup_fn,
            name_omit=name_omit,
            ratio_target=ratio_target,
            sensitivity_loss=sensitivity_loss,
            measurements_points=measurements_points,
            *args,
            **kwargs,
        )

        self.gamma: float = 80.0  # original parameter
        self.max_iter: int = 500  # original parameter
        self.energy_power: int = self.power_for_energy
        self.target_metric: str = target_metric
        self.lower_bound = lower_bound
        self.enforce_rank_multiples_of = enforce_rank_multiples_of

        self._base_schedule_gamma = self.gamma
        self._base_max_iter = self.max_iter

        self.max_target_overshoot = max_target_overshoot
        self.step_count_candidates = [2000]
        if self.sensitivity_loss in ("energy2_normal_klscaled",):
            self._set_schedule_from_step_count(2000)

        self.last_search_stats = None

    @property
    def requires_decomposed_model_for_search(self):
        return True

    # ------------------------------------------------------------------
    #  Initialization
    # ------------------------------------------------------------------

    def initialize_search(
        self, lrd_method: BaseFactorization, model: nn.Module, spec_tensor=None
    ):
        self.lrd_method = lrd_method
        layer_data = OrderedDict()

        if self.memvit_sensitivity_loss in ("energy1", "energy2"):
            with torch.no_grad():
                for name, module in model.named_modules():
                    if all(omit not in name for omit in self.name_omit) and isinstance(module, nn.Linear):
                        factorized_matrix = self.lrd_method.factorize_matrix(
                            module.weight, ratio=1.0, name=name
                        )
                        layer_data[name] = {
                            "S": factorized_matrix.singular_values,
                            "eq_rank": factorized_matrix.eq_rank,
                            "shape": module.weight.shape,
                        }
        else:
            layer_sensitivity, _ = self._get_layer_sensitivity(model, spec_tensor)
            for layer_name, sensitivity_data in layer_sensitivity.items():
                matrix = model.get_submodule(layer_name).weight
                eq_rank = get_eq_rank(matrix.shape[0], matrix.shape[1])

                if self.enforce_rank_multiples_of is not None:
                    sensitivity_data = {
                        k: v
                        for k, v in sensitivity_data.items()
                        if int(k * eq_rank) % self.enforce_rank_multiples_of == 0
                    }

                layer_sensitivity[layer_name] = sensitivity_data
                layer_data[layer_name] = {
                    "S": None,
                    "eq_rank": eq_rank,
                    "shape": matrix.shape,
                }
            self.layer_sensitivity = layer_sensitivity

        if self.lower_bound == -1.0:
            self.lower_bound = 0.1 if self.lrd_method.vision else 0.3
        else:
            self.lower_bound = float(self.lower_bound)

        self.layer_data = layer_data

    # ------------------------------------------------------------------
    #  Hook for subclass bias injection (identity in base)
    # ------------------------------------------------------------------

    def _score_candidate(self, layer_name: str, base_loss: float) -> float:
        """Return the (possibly bias-adjusted) loss for *layer_name*.

        The base implementation returns *base_loss* unchanged.  Override in
        subclasses (e.g. MEMVIT_PLUS) to inject cross-layer depth bias.
        """
        return base_loss

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _get_lower_bound_rank(self, eq_rank: int) -> int:
        raw_lb = int(math.floor(eq_rank * self.lower_bound))
        if self.enforce_rank_multiples_of is None:
            return max(1, raw_lb)
        k = self.enforce_rank_multiples_of
        lb = int(math.ceil(raw_lb / k) * k)
        lb = min(lb, eq_rank)
        return max(1, lb)

    def _project_rank(self, proposed_rank: int, lower_bound_rank: int) -> int:
        proposed_rank = int(proposed_rank)
        if self.enforce_rank_multiples_of is not None:
            k = self.enforce_rank_multiples_of
            proposed_rank = (proposed_rank // k) * k
        return max(1, proposed_rank)

    def _set_schedule_from_step_count(self, step_count: int):
        """Scale the exponential schedule together with ``max_iter``."""
        step_count = int(step_count)
        if step_count <= 0:
            raise ValueError("step_count must be positive")
        self.max_iter = step_count
        self.gamma = self._base_schedule_gamma * (step_count / self._base_max_iter)
        print(f"Using step_count={self.max_iter}, schedule_gamma={self.gamma:.4f}")

    def compute_energy_loss(self, singular_values: torch.Tensor, new_rank: int) -> float:
        new_rank = int(max(0, new_rank))
        total_energy = torch.sum(singular_values ** self.energy_power)
        if total_energy == 0:
            return 0.0
        lost_energy = torch.sum(singular_values[new_rank:] ** self.energy_power)
        return (lost_energy / total_energy).item()

    def get_task_loss(self, layer_sensitivity, layer_data, name: str, new_rank: int) -> float:
        eq_rank = layer_data[name]["eq_rank"]
        target_ratio = round(new_rank / eq_rank, 5)
        closest_key = min(
            layer_sensitivity[name].keys(),
            key=lambda x: abs(x - target_ratio),
        )
        return layer_sensitivity[name][closest_key]

    # ------------------------------------------------------------------
    #  Greedy search
    # ------------------------------------------------------------------

    def single_search(self, model: nn.Module):
        if self.target_metric == "flops":
            self.flops_dict, _ = self.get_layer_wise_flops(model=model)

        initial_complexity = 0
        current_ranks = OrderedDict()
        lower_bound_ranks = OrderedDict()

        for name, data in self.layer_data.items():
            n, d = data["shape"]
            if self.target_metric == "flops":
                initial_complexity += self.flops_dict[name]
            else:
                initial_complexity += n * d
            current_ranks[name] = data["eq_rank"]
            lower_bound_ranks[name] = self._get_lower_bound_rank(data["eq_rank"])

        p_total = initial_complexity
        p_current = initial_complexity
        p_target = p_total * self.ratio_target

        print(f"Initial parameters (decomposed): {p_total:,}")
        print(f"Target parameters (alpha={self.ratio_target}): {int(p_target):,}\n")

        t = 1
        while p_current > p_target and t <= self.max_iter:
            p_target_t = tau_schedule(p_total, p_target, t, self.gamma)
            p_target_t_minus_1 = tau_schedule(p_total, p_target, t - 1, self.gamma)
            p_to_remove = p_target_t_minus_1 - p_target_t

            if p_to_remove <= 0:
                print("Parameter reduction schedule saturated. Halting.")
                break

            candidate_losses = {}
            temp_ranks = {}

            for name, data in self.layer_data.items():
                current_rank = current_ranks[name]
                if current_rank <= lower_bound_ranks[name]:
                    continue

                n, d = data["shape"]
                eq_rank = data["eq_rank"]

                if self.target_metric == "flops":
                    delta_r = p_to_remove * eq_rank / self.flops_dict[name]
                else:
                    delta_r = p_to_remove / (n + d)

                m_t = math.floor(current_rank - delta_r)
                m_t = self._project_rank(m_t, lower_bound_ranks[name])

                if m_t >= current_rank or m_t < lower_bound_ranks[name]:
                    continue

                temp_ranks[name] = m_t

                if self.memvit_sensitivity_loss in ("energy1", "energy2"):
                    base_loss = self.compute_energy_loss(data["S"], m_t)
                else:
                    if len(self.layer_sensitivity[name]) == 0:
                        continue
                    base_loss = self.get_task_loss(
                        self.layer_sensitivity, self.layer_data, name, m_t
                    )

                candidate_losses[name] = self._score_candidate(name, base_loss)

            if not candidate_losses:
                print("No further rank reduction possible. Halting.")
                break

            l_star = min(candidate_losses, key=candidate_losses.get)
            new_rank_l_star = temp_ranks[l_star]
            old_rank_l_star = current_ranks[l_star]
            current_ranks[l_star] = new_rank_l_star

            p_current = 0
            for name, r in current_ranks.items():
                n, d = self.layer_data[name]["shape"]
                if self.target_metric == "flops":
                    p_current += r / self.layer_data[name]["eq_rank"] * self.flops_dict[name]
                else:
                    p_current += r * (n + d)

            print(
                f"Iter {t}: Best layer to compress is '{l_star}'. "
                f"Rank reduced from {old_rank_l_star} to {new_rank_l_star}. "
                f"Current params: {int(p_current):,}"
            )
            t += 1

        print("\n--- Compression Finished ---")
        print(f"Final parameter count: {int(p_current):,} (Target was {int(p_target):,})")
        final_compression = p_total / p_current
        print(f"Achieved compression ratio: {final_compression:.2f}x")

        self.last_search_stats = {
            "final_complexity": float(p_current),
            "target_complexity": float(p_target),
            "realized_ratio": float(p_current / p_total),
            "steps_used": int(t - 1),
        }

        return current_ranks

    def search(self, model: nn.Module):
        return self.single_search(model)
