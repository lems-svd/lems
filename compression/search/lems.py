from torch import nn

from ..factorization._interface import BaseFactorization, get_valid_layers
from ..factorization._interface import get_eq_rank
from ._sensitivity_base import SensitivityBasedSearch

import pulp
from ..factorization._interface import _find_decoder_layers
from gurobipy import Model, GRB, quicksum
import gurobipy as gp
from dataclasses import dataclass
import json
import os
import torch
import copy
import gc


# ------------------------------------------------------------------
#  Cross-layer depth bias (used by LEMS, ASVD+, MemViT+)
# ------------------------------------------------------------------

def get_depth_multiplier(
    current_block: int,
    total_blocks: int,
    crosslayer_term: str,
    alpha: float = 0.0,
    gamma: float = 0.0,
    is_vision: bool = False,
) -> float:
    """Compute a per-block importance multiplier for cross-layer bias scoring.

    Supported ``crosslayer_term`` values: ``constant``, ``harmonic``,
    ``harmonicv2``, ``linear``.
    """
    if is_vision:
        return 1.0
    if crosslayer_term == "constant":
        return 1.0
    if crosslayer_term == "harmonic":
        total_blocks_2 = total_blocks * 2
        return sum(
            [1.0] + [1.0 / (k + 1) for k in range(current_block * 2, total_blocks_2)]
        )
    if "harmonicv2" in crosslayer_term:
        alpha = float(alpha)
        gamma = float(gamma)
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        curr_scale_val = sum(1.0 / (k + 1) ** alpha for k in range(current_block, total_blocks))
        start_val = sum(1.0 / (k + 1) ** alpha for k in range(0, total_blocks))
        end_val = sum(1.0 / (k + 1) ** alpha for k in range(total_blocks - 1, total_blocks))
        min_val, max_val = min(end_val, start_val), max(end_val, start_val)
        denom = max(max_val - min_val, 1e-12)
        curr_scale_val_norm = (curr_scale_val - min_val) / denom
        depth_multiplier = 1.0 + gamma * curr_scale_val_norm
        if depth_multiplier < 0:
            raise ValueError(f"depth_multiplier must be non-negative a: {alpha} g: {gamma}")
        return depth_multiplier
    if crosslayer_term == "linear":
        return total_blocks - current_block
    raise ValueError(f"Unknown crosslayer_term: {crosslayer_term}")


# ------------------------------------------------------------------
#  Shared Optuna search loop (used by LEMS, ASVD+, MemViT+)
# ------------------------------------------------------------------

def optuna_inner_search(objective_fn, *, n_trials):
    """Run a single Optuna study and return the best trial.

    Parameters
    ----------
    objective_fn : callable(trial) → float
        The Optuna objective.  It will be called *n_trials* times.

    Returns
    -------
    best_trial : optuna.trial.FrozenTrial or None
        ``None`` when every trial was pruned / no completed trial exists.
    """
    import optuna
    from optuna.samplers import TPESampler
    from optuna.trial import TrialState

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective_fn, n_trials=n_trials)

    complete = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if len(complete) == 0:
        return None
    return study.best_trial

@dataclass
class ILPSettings:
    layer_wise_monotone: bool
    block_wise_monotone: bool
    in_block_monotone: bool
    in_block_monotone_qkv: bool
    use_lower_bound: bool = False
    solver: str = "gurobi"  # Default solver is CBC, can be changed to "gurobi" if available

    def __post_init__(self):
        if self.in_block_monotone and self.in_block_monotone_qkv:
            raise ValueError("in_block_monotone and in_block_monotone_qkv cannot be both True. Choose one.")
        if self.in_block_monotone or self.in_block_monotone_qkv and not self.block_wise_monotone:
            raise ValueError("in_block_monotone or in_block_monotone_qkv can only be True if block_wise_monotone is also True.")
        if self.solver not in ["cbc", "gurobi"]:
            raise ValueError("Solver must be either 'cbc' or 'gurobi'. Gurobi requires a license.")
        if self.layer_wise_monotone and self.block_wise_monotone:
            raise ValueError("layer_wise_monotone and block_wise_monotone cannot be both True. Choose one.")
        if self.solver == "cbc" and (self.in_block_monotone or self.in_block_monotone_qkv):
            raise ValueError("Contraints are not implemented for this solver. You must port them from our gurobi implementation.")
        if self.solver == "cbc" and self.layer_wise_monotone:
            print("WARNING: layer wise optimization requires many constraints, which may lead to",
                  "long solve times with CBC - if it can solve iot at all. Consider using Gurobi instead.")
        

class LEMSSearch(SensitivityBasedSearch):
    """
    ELASTIC: Efficient Layerwise Allocation of Sparsity through The Interplay of Error Modelling and Constraints
    Our propised search framework.
    """
    def __init__(self, ratio_target=0.5, sensitivity_loss="energy2_normal_klscaled", crosslayer_term="harmoic", halpha=0, hgamma=0, measurements_points="0.1", target_metric="params", enforce_rank_multiples_of=None, solver="gurobi", *args, **kwargs):
        super().__init__(
            ratio_target=ratio_target,
            sensitivity_loss=sensitivity_loss,
            measurements_points=measurements_points,
            *args,
            **kwargs
        )
        self.crosslayer_term = crosslayer_term  # how to combine the sensitivity measurements across layers
        self.alpha = halpha
        self.gamma = hgamma
        assert hgamma >= 0, "gamma must be non-negative"
        self.one_shot = False
        self.target_metric = target_metric  # or "flops"
        self.enforce_rank_multiples_of = enforce_rank_multiples_of  # set to an integer to enforce ranks to be multiples of this number
        self.solver = solver

    def initialize_search(
        self, lrd_method: BaseFactorization, model: nn.Module, spec_tensor=None
    ):
        self.lrd_method = lrd_method
        self.ilp_settings = ILPSettings(
            layer_wise_monotone=False,
            block_wise_monotone=False,# if self.lrd_method.vision else True,
            in_block_monotone=False,
            in_block_monotone_qkv=False,
            solver=self.solver,
        )
        layer_sensitivity, size_dict = self._get_layer_sensitivity(model, spec_tensor)
        self.sensitivity_dict = layer_sensitivity
        self.size_dict = size_dict
        self.shape_dict = self.get_layer_shape_dict(model)

    def prepare_data(self, size_dict, layer_sensitivity, compression_target, layers_per_block):
        # --- 1. Data Setup ---
        # A list of lists. Each inner list represents one of the N variables.
        # Each tuple inside the inner list is a (compression_ratio, error_caused) pair.


        is_vision = self.lrd_method.vision
        data = []
        layer_name_list = list(layer_sensitivity.keys())
        compression_list = []
        active_layer_sizes = []
        upper_bound_offset = 1.0
        if self.lrd_method.vision or compression_target < 0.5:
            lower_bound = 0.1
        else:
            lower_bound = 0.3
        upper_bound = compression_target + upper_bound_offset

        for i, (layer_name, sensitivity_data) in enumerate(layer_sensitivity.items()):
            if self.enforce_rank_multiples_of is not None:
                n, m = self.shape_dict[layer_name]
                eq_rank = get_eq_rank(n, m)
                sensitivity_data = {k: v for k, v in sensitivity_data.items() if int(k * eq_rank) % self.enforce_rank_multiples_of == 0}
                print(f"Layer {layer_name}: enforcing ranks to be multiples of {self.enforce_rank_multiples_of}, with eq_rank {eq_rank}, keeping {len(sensitivity_data)} points.")
            current_block = i // layers_per_block
            total_blocks = max(len(layer_name_list) // layers_per_block, 1)
            depth_multiplier = get_depth_multiplier(
                current_block=current_block,
                total_blocks=total_blocks,
                crosslayer_term=self.crosslayer_term,
                alpha=self.alpha,
                gamma=self.gamma,
                is_vision=is_vision,
            )
            layer_data = [(size_dict[layer_name], 0.0)]+[(compression * size_dict[layer_name], sensitivity * depth_multiplier) for compression, sensitivity in sensitivity_data.items() if compression >= lower_bound and compression <= upper_bound]
            layer_compression_list = [1.0] + [key for key in sensitivity_data.keys() if key >= lower_bound and key <= upper_bound]
            #[key for key in layer_sensitivity[layer_name_list[i]].keys() if key >= lower_bound and key <= upper_bound]
            data.append(layer_data)
            compression_list.append(layer_compression_list)
            active_layer_sizes.append(size_dict[layer_name])

        total_parameters = sum(active_layer_sizes)
        print(f"Total parameters in model: {total_parameters}")
        compression_param_target = total_parameters * compression_target
        print(f"Target compression ratio: {compression_target} ({compression_param_target} parameters)")
        return data, layer_name_list, compression_list, compression_param_target


    def search(self, model: nn.Module):
        default_param_ratio = 1.0
        stages, _ = _find_decoder_layers(model)
        layers_per_block = sum([1 for mod in stages[0].modules() if isinstance(mod, nn.Linear)])
        if self.target_metric == "flops":
            flops_per_layer, _ = self.get_layer_wise_flops(model)
            self.size_dict = flops_per_layer
        else:
            self.size_dict = self.size_dict
        print("Layers per block", layers_per_block)
        if self.one_shot or self.crosslayer_term == "constant":
            return self.single_search(layers_per_block, default_param_ratio)
        else:
            return self.grid_search(model, layers_per_block, default_param_ratio)

    def grid_search(self, model: nn.Module, layers_per_block, default_param_ratio,
                    n_trials: int = 10, alpha_range=(0.0, 3.0), gamma_range=(0.0, 7.0)):
        """Hyperparameter search with Optuna."""

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
            alpha = trial.suggest_float("alpha", alpha_range[0], alpha_range[1])
            gamma = trial.suggest_float("gamma", gamma_range[0], gamma_range[1])
            self.alpha = alpha
            self.gamma = gamma

            search_ranks = self.single_search(layers_per_block, default_param_ratio)
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
            n_trials=n_trials,
        )

        best_ranks = best_trial.user_attrs["search_ranks"]
        print(f"\nBest metric {best_trial.value} found with "
              f"alpha={best_trial.params['alpha']}, gamma={best_trial.params['gamma']}")
        self._restore_model(module_dict, module_bkup_dict)

        del model_bkup
        gc.collect()
        return best_ranks


    
    def single_search(self, layers_per_block, default_param_ratio):
        data, layer_name_list, compression_list, compression_param_target = self.prepare_data(
            size_dict=self.size_dict,
            layer_sensitivity=self.sensitivity_dict,
            compression_target=self.ratio_target,
            layers_per_block=layers_per_block
        )

        if self.ilp_settings.solver == "gurobi":
            # NOTE: Gurobi requires a license and is not free for commercial use.
            compression_dict = ilp_search_gurobi(
                data=data,
                compression_list=compression_list,
                layer_name_list=layer_name_list,
                compression_param_target=compression_param_target,
                layers_per_block=layers_per_block,
                ilp_settings=self.ilp_settings,
            )
        else:
            # CBC with pulp is free, but is slower and may not find the (optimal) solution.
            compression_dict = ilp_search_cbc(
                data=data,
                compression_list=compression_list,
                layer_name_list=layer_name_list,
                compression_param_target=compression_param_target,
                layers_per_block=layers_per_block,
                ilp_settings=self.ilp_settings,
            )

        layers_min_ratio = {
            layername: default_param_ratio for layername in self.sensitivity_dict.keys()
        }
        for layername, param_ratio in compression_dict.items():
            layers_min_ratio[layername] = param_ratio

        return layers_min_ratio

def ilp_search_cbc(data, compression_list, layer_name_list, compression_param_target: int,
                   ilp_settings, layers_per_block=None,
                   rank_list=None, shared_rank_groups=None):    
    num_variables = len(data)
    print(f"Optimizing {len(data)} variables using PuLP.")

    try:
        cpu_cores = os.cpu_count()/2 or 1
    except:
        cpu_cores = 1
    print(f"Detected {cpu_cores} CPU cores for PuLP.")
    # --- 2. Model Definition ---
    # In PuLP, we initialize the problem with a name and the direction (Minimize)
    model = pulp.LpProblem("Minimize_Compression_Error", pulp.LpMinimize)

    # --- 3. Decision Variables ---

    # Create a list of lists to hold the binary decision variables
    # variables[i][j] will be 1 if we choose pair j for variable i, and 0 otherwise
    variables = []
    for i in range(num_variables):
        var_choices = []
        for j in range(len(data[i])):
            # Create a binary variable for each possible choice
            # PuLP uses LpBinary for binary variables
            var = pulp.LpVariable(name=f"x_{i}_{j}", cat=pulp.LpBinary)
            var_choices.append(var)
        variables.append(var_choices)

    # --- 4. Objective Function ---

    # The objective is to minimize the sum of errors from the chosen pairs.
    # pulp.lpSum is equivalent to Gurobi's quicksum
    model += pulp.lpSum(
        data[i][j][1] * variables[i][j] 
        for i in range(num_variables) 
        for j in range(len(data[i]))
    ), "Total_Error_Objective"

    # --- 5. Constraints ---

    # --- Constraint 1: The sum of compression ratios (param counts) must meet the target.
    total_params_expr = pulp.lpSum(
        data[i][j][0] * variables[i][j] 
        for i in range(num_variables) 
        for j in range(len(data[i]))
    )
    
    model += (total_params_expr <= compression_param_target), "Compression_Constraint_Upper_Bound"

    # --- Constraint 2: For each variable, exactly one choice MUST be made.
    for i in range(num_variables):
        model += (
            pulp.lpSum(variables[i][j] for j in range(len(data[i]))) == 1
        ), f"Select_One_From_Var_{i}"

    # --- 6. Solve the Problem ---

    print("Solving the ILP problem with PuLP...")
    
    # We use CBC solver (default for PuLP) and set a time limit of 180 seconds.
    # msg=True enables logging to console similar to Gurobi's output
    time_limit_seconds = max(18, int(180 / cpu_cores))  # Adjust time limit based on CPU cores
    solver = pulp.PULP_CBC_CMD(timeLimit=time_limit_seconds, msg=True, threads=cpu_cores)
    model.solve(solver)
    
    print("Done!")

    # --- 7. Extract the Results ---

    # Map PuLP status integer to readable string
    status_str = pulp.LpStatus[model.status]
    print(f"\nStatus: {status_str}")
    print(f"Target Compression: {compression_param_target}")

    total_achieved_compression = 0
    minimized_total_error = 0
    
    compression_dict = {}

    print("\nOptimal selections:")
    for i in range(num_variables):
        for j in range(len(data[i])):
            # In PuLP, use .varValue to get the result. 
            # Floating point tolerance check is safer than exact == 1
            if variables[i][j].varValue is not None and variables[i][j].varValue > 0.99:
                selected_compression = data[i][j][0]
                selected_error = data[i][j][1]
                total_achieved_compression += selected_compression
                minimized_total_error += selected_error
                compression_dict[layer_name_list[i]] = compression_list[i][j]
                print(f" - Variable {i}: Choose pair {j} -> (Layer {layer_name_list[i]} Compression: {compression_list[i][j]}, Error: {selected_error})")
    
    print("\n--- Summary ---")
    print(f"Minimum Total Error: {minimized_total_error:.2f}")
    print(f"Achieved Total Compression: {total_achieved_compression:.2f}")
    
    return compression_dict

def ilp_search_gurobi(data, compression_list, layer_name_list, compression_param_target: int,
                      ilp_settings: ILPSettings, layers_per_block=None,
                      rank_list=None, shared_rank_groups=None):

    num_variables = len(data)
    print(f"Optimizing {len(data)} variables.")

    # Load Gurobi license information from a JSON file
    if os.path.exists('gurobi_license.json'):
        license_found = True
        print("Using Gurobi license from 'gurobi_license.json'.")
        with open('gurobi_license.json') as f:
            license_info = json.load(f)
    else:
        license_found = False
        print("Gurobi license file 'gurobi_license.json' not found. Trying to proceed without it.")

 
    with gp.Env(logfilename='logfile.log', empty=True, params=None) as env:
        # Set Gurobi parameters using the loaded license information
        if license_found:
            env.setParam("WLSACCESSID", license_info["WLSACCESSID"])
            env.setParam("WLSSECRET", license_info["WLSSECRET"])
            env.setParam("LICENSEID", license_info["LICENSEID"])
        env.start()

        # --- 2. Model Definition ---

        with gp.Model(env=env, name="Minimize_Compression_Error") as model:

            # --- 3. Decision Variables ---

            # Create a list of lists to hold the binary decision variables
            # variables[i][j] will be 1 if we choose pair j for variable i, and 0 otherwise
            variables = []
            for i in range(num_variables):
                var_choices = []
                for j in range(len(data[i])):
                    # Create a binary variable for each possible choice
                    var = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
                    var_choices.append(var)
                variables.append(var_choices)

            # Update the model to integrate new variables
            model.update()

            # --- 4. Objective Function ---

            # The objective is to minimize the sum of errors from the chosen pairs.
            model.setObjective(
                quicksum(data[i][j][1] * variables[i][j] for i in range(num_variables) for j in range(len(data[i]))),
                GRB.MINIMIZE
            )

            # --- 5. Constraints ---

            # --- Constraint 1: The sum of compression ratios must meet the target.
            model.addConstr(
                quicksum(data[i][j][0] * variables[i][j] for i in range(num_variables) for j in range(len(data[i]))) <= compression_param_target,
                "Compression_Constraint_Upper_Bound"
            )
            if ilp_settings.use_lower_bound:
                model.addConstr(
                    quicksum(data[i][j][0] * variables[i][j] for i in range(num_variables) for j in range(len(data[i]))) >= compression_param_target * 0.97,
                    "Compression_Constraint_Lower_Bound"
                )

            # --- Constraint 2: For each variable, exactly one choice MUST be made.
            for i in range(num_variables):
                model.addConstr(
                    quicksum(variables[i][j] for j in range(len(data[i]))) == 1,
                    f"Select_One_From_Var_{i}"
                )

            # --- Constraint 3: Subsequent blocks must have monotonically decreasing average compression ratios
            # following findings from H. Weizhong et al. (ICML'25)
            if ilp_settings.layer_wise_monotone:
                block_size = 1  # trick to reuse block constraint for layerwise 
            else:
                block_size = layers_per_block
            num_blocks = num_variables // block_size
            if ilp_settings.block_wise_monotone:
                for b in range(num_blocks - 1):
                    # Average param count in block b
                    avg_block_b = (1 / block_size) * quicksum(
                        data[i][j][0] * variables[i][j]
                        for i in range(b * block_size, (b + 1) * block_size)
                        for j in range(len(data[i]))
                    )

                    # Average param count in block b+1
                    avg_block_next = (1 / block_size) * quicksum(
                        data[i][j][0] * variables[i][j]
                        for i in range((b + 1) * block_size, (b + 2) * block_size)
                        for j in range(len(data[i]))
                    )

                    # Enforce monotonic decrease (or equal)
                    model.addConstr(avg_block_next <= avg_block_b, f"monotonic_block_{b}")
            
            # --- Constraint 4a (optional): Enforce full monotonicity within each bloc.
            # Mutually exclusive to 4b.
            if ilp_settings.in_block_monotone:
                for b in range(num_blocks):
                    for i in range(b * block_size, (b + 1) * block_size - 1):
                        print(layer_name_list[i], "block id", b, "layer idx", i)
                        # Average param count in block b
                        curr_comp = quicksum(
                            compression_list[i][j] * variables[i][j]
                            for j in range(len(compression_list[i]))
                        )

                        # Average param count in block b+1
                        next_comp = quicksum(
                            compression_list[i+1][j] * variables[i+1][j]
                            for j in range(len(compression_list[i+1]))
                        )
                        # Enforce monotonic decrease (or equal)
                        model.addConstr(next_comp <= curr_comp, f"monotonic_layer_{i}_block_{b}")
                
            # --- Constraint 4b (optional): Monotonicity within each block, with QKV independet of each other.
            # Mutually exclusive to 4a.
            if ilp_settings.in_block_monotone_qkv:
                for b in range(num_blocks):
                    for i in range(b * block_size + 3, (b + 1) * block_size - 1):
                        print(layer_name_list[i], "block id", b, "layer idx", i)
                        # Average param count in block b
                        curr_comp = quicksum(
                                compression_list[i][j] * variables[i][j]
                                for j in range(len(compression_list[i]))
                            )
                        if b * block_size + 3 == i:
                            # first three are usually QKV
                            last_comp1 = quicksum(
                                compression_list[i-1][j] * variables[i-1][j]
                                for j in range(len(compression_list[i-1]))
                            )
                            last_comp2 = quicksum(
                                compression_list[i-2][j] * variables[i-2][j]
                                for j in range(len(compression_list[i-2]))
                            )
                            last_comp3 = quicksum(
                                compression_list[i-3][j] * variables[i-3][j]
                                for j in range(len(compression_list[i-3]))
                            )
                            model.addConstr(curr_comp <= last_comp1, f"monotonic_layer_{i-1}_block_{b}")
                            model.addConstr(curr_comp <= last_comp2, f"monotonic_layer_{i-2}_block_{b}")
                            model.addConstr(curr_comp <= last_comp3, f"monotonic_layer_{i-3}_block_{b}")

                        # Average param count in block b+1
                        next_comp = quicksum(
                            compression_list[i+1][j] * variables[i+1][j]
                            for j in range(len(compression_list[i+1]))
                        )
                        # Enforce monotonic decrease (or equal)
                        model.addConstr(next_comp <= curr_comp, f"monotonic_layer_{i}_block_{b}")

            # --- Constraint 5 (optional): Shared-rank groups must have equal compression.
            if shared_rank_groups:
                print("Adding shared rank constraints for groups:", shared_rank_groups)
                for gidx, group in enumerate(shared_rank_groups):
                    ref = group[0]
                    ref_rank = quicksum(compression_list[ref][j] * variables[ref][j] for j in range(len(compression_list[ref])))
                    for i in group[1:]:
                        this_rank = quicksum(compression_list[i][j] * variables[i][j] for j in range(len(compression_list[i])))
                        model.addConstr(this_rank == ref_rank, f"shared_rank_group_{gidx}_{i}")

            # --- 6. Solve the Problem ---

            print("Solving the ILP problem...")
            model.setParam('TimeLimit', 180)
            model.optimize()
            print("Done!")

            # --- 7. Extract the Results ---

            print(f"\nStatus: {model.status}")
            print(f"Target Compression: {compression_param_target}")

            total_achieved_compression = 0
            minimized_total_error = 0
            
            compression_dict = {}

            print("\nOptimal selections:")
            for i in range(num_variables):
                for j in range(len(data[i])):
                    if variables[i][j].x == 1:  # Use .x to get the value of the variable
                        selected_compression = data[i][j][0]
                        selected_error = data[i][j][1]
                        total_achieved_compression += selected_compression
                        minimized_total_error += selected_error
                        compression_dict[layer_name_list[i]] = compression_list[i][j]
                        print(f" - Variable {i}: Choose pair {j} -> (Layer {layer_name_list[i]} Compression: {compression_list[i][j]}, Error: {selected_error})")
            
            print("\n--- Summary ---")
            print(f"Minimum Total Error: {minimized_total_error:.2f}")
            print(f"Achieved Total Compression: {total_achieved_compression:.2f}")
    return compression_dict


# Backward-compatible module-level wrappers (used by atp.py)
def compress_model(module_dict, module_bkup_dict, rank_dict, lrd_method: BaseFactorization):
    SensitivityBasedSearch._compress_model_ratios(module_dict, module_bkup_dict, rank_dict, lrd_method)

def restore_model(module_dict, module_bkup_dict):
    SensitivityBasedSearch._restore_model(module_dict, module_bkup_dict)