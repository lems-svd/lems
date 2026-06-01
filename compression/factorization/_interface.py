import numpy as np
import sys
import torch
from torch import nn
import gc
from tqdm import tqdm
from all_utils.model_io import get_model_name
import pickle
import os

from ._dataclasses import FactorizedMatrix  # noqa: F401 – re-exported

# ---------------------------------------------------------------------------
#  Shared whitening utilities
# ---------------------------------------------------------------------------

@torch.no_grad()
def whitening(dev, raw_scale, name, alpha=0.0, method="cholesky", increment=1e-6, *, double_precision=False):
    """Compute a whitening matrix and its inverse.

    Parameters
    ----------
    dev : torch.device
    raw_scale : dict
        Mapping of layer name → covariance / gram matrix.
    name : str
        Key into *raw_scale*.
    alpha : float, optional
        Ledoit-Wolf style shrinkage intensity (Cholesky only).
    method : ``"cholesky"`` | ``"svd"``
        * ``"cholesky"`` — Cholesky decomposition (fast, requires PD matrix).
        * ``"svd"`` — SVD-based whitening (more numerically stable, works
          with PSD / ill-conditioned matrices).
    double_precision : bool
        When *True* the Cholesky decomposition is computed in float64 for
        extra numerical stability.  Ignored when *method* is ``"svd"``
        (which always uses float64 internally).  Results are always
        returned in float32.

    Returns
    -------
    (whitening_matrix, dewhitening_matrix) : tuple[Tensor, Tensor]
    """
    if method == "svd":
        return _whitening_svd(dev, raw_scale, name)
    return _whitening_cholesky(dev, raw_scale, name, alpha, increment, double_precision=double_precision)


@torch.no_grad()
def _whitening_cholesky(dev, raw_scale, name, alpha=0.0, increment=1e-6,* , double_precision=False):
    """Cholesky-based whitening.

    When *double_precision* is ``True`` the decomposition is computed in
    float64 for extra numerical stability (equivalent to the former
    ``whitening_fast``).  Results are always returned in float32.
    """
    if double_precision:
        raw_scale_mat = raw_scale[name].double().to(dev)
    else:
        raw_scale_mat = raw_scale[name].clone().float().to(dev)

    if alpha > 0.0:
        reg_term = torch.mean(raw_scale_mat.diag()) * alpha
        raw_scale_mat.mul_(1.0 - alpha)
        raw_scale_mat.diagonal().add_(reg_term)

    scale_diag = None
    for attempt in range(4):  # up to 3 eigenvalue-shift retries
        try:
            scale_diag = torch.linalg.cholesky(raw_scale_mat)
            break
        except torch.linalg.LinAlgError:
            if attempt == 3:
                print(f"FATAL: Matrix '{name}' could not be made positive definite.")
                sys.exit()
            eigenvalues = torch.linalg.eigvalsh(raw_scale_mat)
            raw_scale_mat.diagonal().add_(-eigenvalues[0] + increment)
            del eigenvalues

    del raw_scale_mat

    identity = torch.eye(scale_diag.shape[0], device=dev, dtype=scale_diag.dtype)
    scale_diag_inv = torch.linalg.solve_triangular(scale_diag, identity, upper=False)
    del identity

    return scale_diag.float(), scale_diag_inv.float()


@torch.no_grad()
def _whitening_svd(dev, raw_scale, name):
    """SVD-based whitening — numerically stable for PSD / ill-conditioned matrices."""
    raw_scale_mat = raw_scale[name].double().to(dev)

    try:
        U, S, _ = torch.linalg.svd(raw_scale_mat, full_matrices=False)
    except Exception as e:
        print(f"FATAL: SVD failed for '{name}' — {e}")
        sys.exit()

    eps = 1e-6
    whitening_matrix = U @ torch.diag(torch.sqrt(S + eps))
    dewhitening_matrix = torch.diag(torch.rsqrt(S + eps)) @ U.T

    del raw_scale_mat, U, S
    return whitening_matrix.float().to(dev), dewhitening_matrix.float().to(dev)


# FactorizedMatrix is imported from ._dataclasses and re-exported for
# backward compatibility.  See _dataclasses.py for the definition.

# These utilities now live in all_utils.compression_utils but are re-exported
# here for backward compatibility with existing ``from ._interface import …`` sites.
from all_utils.compression_utils import (      # noqa: F401 – re-exports
    is_linear_like_conv,
    get_valid_layers,
    get_eq_rank,
    _find_decoder_layers,
    safe_whitened_svd,
    plot_compression_rates,
)


class BaseFactorization:
    def __init__(self, vision, calib_dataset_name, use_cache=True, blockwise_factorization=False, progressive_compression=False, do_post_calibration="default", calibration_ranks={}, workspace_dir="./workspace", **kwargs):
        self.scaling_dict = {}
        self.input_shapes = {}
        self.vision = vision
        self.use_file_cache = use_cache
        self.debug = kwargs.get("debug", False)
        use_local_cache = True
        self.one_shot_factorization = not blockwise_factorization
        if not blockwise_factorization and progressive_compression:
            print("Warning: progressive compression requires blockwise " \
            "factorization. Setting progressive_compression to False.")
            progressive_compression = False
        self.progressive_compression = progressive_compression
        self.static_progressive_compression_ratio = 0.5
        self.calibration_ranks = calibration_ranks
        # file cache requires local cache
        self.use_local_cache = use_local_cache or self.use_file_cache
        self.factorized_layers_cache: dict[str, FactorizedMatrix] = {}
        self.fact_cache_dir = os.path.join(workspace_dir, "cache", "decomposition") + "/"
        self.dev = torch.device(torch.cuda.current_device())
        if do_post_calibration not in ["default", "True", "False"] and do_post_calibration is not True and do_post_calibration is not False:
            raise ValueError("post_calibration_needed must be 'default', True or False.")
        if do_post_calibration in ["True", "False"]:
            do_post_calibration = True if do_post_calibration == "True" else False
        self._do_post_calibration = do_post_calibration
        self.calib_dataset_name = calib_dataset_name
    
    @property
    def post_search_calibration(self):
        # if the factorization method requires recalibration after search
        # (e.g. because it uses the scaling statistics to determine the rank)
        # Default: no recalibration needed.  Override in subclasses that need it.
        return False if self._do_post_calibration == "default" else self._do_post_calibration

    def _dprint(self, *msg):
        if getattr(self, "debug", False):
            print("[DEBUG]", *msg)

    # ------------------------------------------------------------------
    #  Shared calibration-loop helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _enable_weight_grads(model, hook_module, name_omit, white_list=()):
        """Disable all gradients, then re-enable weight gradients on valid Linear layers."""
        for p in model.parameters():
            p.requires_grad = False
        for _, module in get_valid_layers(hook_module, name_omit, white_list=list(white_list)):
            if isinstance(module, nn.Linear):
                for n, p in module.named_parameters():
                    if "bias" not in n:
                        p.requires_grad = True

    def _run_forward_calib(self, model, calib_data, mixup_fn, dev, desc="Gathering activations"):
        """Forward-only calibration loop (no gradients)."""
        with torch.no_grad():
            if self.vision:
                for data, target in tqdm(calib_data, desc=desc):
                    model_inps, _ = mixup_fn(data, target) if mixup_fn is not None else (data, target)
                    model(model_inps.to(dev))
            else:
                for batch in tqdm(calib_data, desc=desc):
                    batch = {k: v.to(dev) for k, v in batch.items()}
                    model(**batch)

    def _run_backward_calib(self, model, calib_data, mixup_fn, dev,
                            desc="Gathering gradients",
                            loss_style="shift_logits"):
        """Backward calibration loop (vision CE / LLM CE with backward).

        Parameters
        ----------
        loss_style : ``"shift_logits"`` | ``"labels_arg"``
            * ``"shift_logits"`` – manually compute shifted logits CE
              (kfac_svd, shampoo variants).
            * ``"labels_arg"`` – pass ``labels=`` to the model and use
              the built-in loss (fwsvd, gfwsvd).
        """
        if self.vision:
            loss_fn = torch.nn.CrossEntropyLoss()
            for data, target in tqdm(calib_data, desc=desc):
                model_inps, targets = mixup_fn(data, target) if mixup_fn is not None else (data, target)
                model_inps, targets = model_inps.to(dev), targets.to(dev)
                out = model(model_inps)
                loss = loss_fn(out, targets)
                loss.backward()
                del model_inps, targets, loss, out
        else:
            if loss_style == "labels_arg":
                for batch in tqdm(calib_data, desc=desc):
                    input_ids = batch["input_ids"].to(dev)
                    out = model(input_ids=input_ids[:, :-1], labels=input_ids[:, 1:])
                    out.loss.backward()
                    model.zero_grad()
            else:  # shift_logits (default)
                loss_fct = torch.nn.CrossEntropyLoss(reduction="sum", ignore_index=-100)
                for batch in tqdm(calib_data, desc=desc):
                    with torch.autocast(device_type="cuda", enabled=False):
                        batch = {k: v.to(dev) for k, v in batch.items()}
                        out = model(**batch)
                        lm_logits = out.logits
                        if torch.isfinite(lm_logits).all():
                            shift_logits = lm_logits[:, :-1, :].contiguous()
                            shift_labels = batch["input_ids"][:, 1:].clone().contiguous()
                            loss = loss_fct(
                                shift_logits.reshape(-1, shift_logits.size(-1)),
                                shift_labels.view(-1),
                            )
                            loss.backward()
                        else:
                            print("Warning: Non-finite logits detected, skipping batch.")
                            continue
                        model.zero_grad()
                        with torch.cuda.device(torch.cuda.current_device()):
                            torch.cuda.empty_cache()
                        del batch, out, loss

    def _collect_vision_shapes(self, hook_module, name_omit, name_prefix, white_list, HookClass=None):
        """Run a dummy forward pass to collect layer input shapes (vision only)."""
        if HookClass is None:
            HookClass = ShapeHook
        shapes_getter = HookClass(
            model=hook_module, name_omit=name_omit,
            dump_shape=True, name_prefix=name_prefix,
            white_list=white_list,
        )
        shapes_getter.attach_hooks()
        dummy_input = torch.randn(20, 3, 224, 224).to(self.dev)
        hook_module(dummy_input) if isinstance(hook_module, nn.Module) else None
        # If hook_module is a sub-module, run the full model instead.
        # For simplicity callers that need model(dummy) should do it themselves.
        shapes_getter.clear_hooks()
        for key, value in shapes_getter.input_shape.items():
            self.input_shapes[key] = value
        del shapes_getter, dummy_input
    
    def get_cache_name(self) -> str:
        # Descriptive cache key including class name and calibration dataset.
        # Override in subclasses if additional fields affect the decomposition.
        decomp_name = self.__class__.__name__
        if self.progressive_compression:
            decomp_name += "_prog"
        decomp_name += f"_{self.calib_dataset_name}"
        return decomp_name

    def __get_cache_file(self, model) -> str:
        model_name = get_model_name(model=model)
        decomp_name = self.get_cache_name()
        file_name = f'{self.fact_cache_dir}{model_name}_{decomp_name}.pkl'
        return file_name
    
    def is_memory_efficient_mode(self, model):
        if not self.vision and model.is_gradient_checkpointing:
            print("Gradient checkpointing enabled, using train mode.")
            # train mode required for grad checkpointing to work.
            return True
        else:
            return False

    def factorization_computations(self, model, name_omit, calib_data, mixup_fn, white_list=[]):
        # check if the decomposed layer dictionaries already exist
        file_name = self.__get_cache_file(model=model)
        if self.use_file_cache and os.path.exists(file_name):
            print("Loading existing decomposed layer dictionaries...")
            with open(file_name, 'rb') as f:
                self.factorized_layers_cache = pickle.load(f)
            print("Loaded successfully.")
            return
        
        self.compute_memory_efficient = self.is_memory_efficient_mode(model=model)
        # train mode required for grad checkpointing to work.
        model = model.train().to(self.dev) if self.compute_memory_efficient else model.eval().to(self.dev)
        
        if self.one_shot_factorization and not self.compute_memory_efficient:
            self._get_scale_and_factorize_one_shot(
                model, name_omit, calib_data, mixup_fn, white_list=white_list
            )
        else:
            self._get_scale_and_factorize_block_wise(
                model, name_omit, calib_data, mixup_fn, white_list=white_list
            )

        if self.use_file_cache:
            if not os.path.exists(self.fact_cache_dir):
                os.makedirs(self.fact_cache_dir)
            with open(file_name, 'wb') as f:
                pickle.dump(self.factorized_layers_cache, f)
            print(f"Saved decomposed layer dictionaries to {file_name}")
        return
    
    def _get_scale_and_factorize_one_shot(self, model, name_omit, calib_data, mixup_fn, white_list):
        """
        This function computes the factorization in one go, which is fast, but not memory
        efficient. It requires to store all the activations of all layers at once.
        """
        # scaling computations of sub functions, e.g. collect data statistics for whitening.
        self._get_scale_and_factorize_module(
                model=model,
                hook_module=model,  # in one shot, hook the whole model
                calib_data=calib_data,
                name_omit=name_omit,
                name_prefix="",
                white_list=white_list,
                mixup_fn=mixup_fn,
                tqdm_message=f"Gathering ",
            )

    def _get_scale_and_factorize_block_wise(self, model, name_omit, calib_data, mixup_fn, white_list):
        """
        This function computes the factorization block by block, which is slower, but
        more memory efficient. It only requires to store the activations of one layer at a time.
        It optionally to compress the layers that it computed the scores for before moving on to
        the next one, thereby considering the changed statistics of the previous layers.
        """
        layers, mod_name = _find_decoder_layers(model)
        for l_idx, layer in enumerate(layers):
            name_prefix = f"{mod_name}.{l_idx}."

            self._get_scale_and_factorize_module(
                model=model,
                hook_module=layer,
                calib_data=calib_data,
                name_omit=name_omit,
                name_prefix=name_prefix,
                white_list=white_list,
                mixup_fn=mixup_fn,
                tqdm_message=f"Layer {l_idx+1}/{len(layers)}: Gathering ",
            )

            valid_layer_modules = get_valid_layers(layer, name_omit, white_list=white_list)
            for name, module_sub in valid_layer_modules:
                key = f"{name_prefix}{name}"
                if self.progressive_compression:
                    # compress the layer before moving on to the next one.
                    rank, ratio, cntinue = self._get_active_rank(
                        module_sub.weight.shape, key, default_ratio=self.static_progressive_compression_ratio)
                    if cntinue:
                        continue
                    # get the factorization from cache or compute it if not available.
                    factorized_matrix = self.factorize_matrix(
                        module_sub.weight,
                        name=key,
                        rank=rank,
                        ratio=ratio,
                        verbose=False
                    )
                    # update weights of layer to be compressed representation.
                    module_sub.weight.data.copy_(
                        factorized_matrix.mat_l.to(self.dev)
                        @ factorized_matrix.mat_r.to(self.dev)
                    ).cpu()
                if self.compute_memory_efficient:
                    # this removes all cached scalings, activations etc.. However, it 
                    # removes the ability to recompute the decomposition without
                    # rerunning the scaling computations.
                    print("Cleaning up factorization cache to save memory...")
                    self._factorize_cleanup(key)
                    torch.cuda.empty_cache()
    
    def _get_scale_and_factorize_module(
            self, model, hook_module, name_prefix, calib_data,
            name_omit, mixup_fn=None, white_list=[], tqdm_message="Gathering "
        ):
        self._compute_scaling(
            model=model,
            hook_module=hook_module,
            name_prefix=name_prefix,
            calib_data=calib_data,
            name_omit=name_omit,
            mixup_fn=mixup_fn,
            white_list=white_list,
            tqdm_message=tqdm_message + "scalings..."
        )
        torch.cuda.empty_cache()
        # perform factorization and store the results in self.factorized_layers_cache
        # NOTE: factorize_matrix fills up the cache if use_local_cache is True. If it is
        # false, there is no point to call this function here as nothing will be cached.
        if self.use_local_cache:
            copied_modules = get_valid_layers(hook_module, name_omit, white_list=white_list)
            for name, module_sub in tqdm(copied_modules, desc=tqdm_message + "factorizations..."):
                name = f"{name_prefix}{name}"
                # this will always return a ratio of 1.0. Dobi-SVD is the only one, that
                # will ever call this function with a different ratio because it can
                # do progressive compression without the need for doing it block by block.
                rank, ratio, cntinue = self._get_active_rank(module_sub.weight.shape, name)
                if cntinue:
                    continue
                det_weight = module_sub.weight.clone().detach()
                # saves in internal cache, so we don't need to store the returned value.
                _ = self.factorize_matrix(
                    matrix=det_weight,
                    rank=rank, ratio=ratio, name=name, verbose=False
                )
                det_weight = None
                del det_weight

    def _compute_scaling(self, model, hook_module, name_prefix, calib_data, name_omit, mixup_fn=None, white_list=[], tqdm_message="Gathering "):
        print("\nNo scaling method implemented. This is fine" \
        " as long as your method is only requiring weights.")
        pass

    def _get_active_rank(self, shape, key, default_ratio=1.0):
        cntinue = False
        if self.calibration_ranks and key in self.calibration_ranks:
            rank = self.calibration_ranks[key] if isinstance(self.calibration_ranks[key], int) else -1
            ratio = self.calibration_ranks[key] if isinstance(self.calibration_ranks[key], float) else -1
            if ratio != -1 and ratio == 1.0:
                rank = ratio = -1
                cntinue = True
            n, m = shape
            eq_rank = get_eq_rank(n, m)
            if rank != -1 and rank == eq_rank:
                rank = ratio = -1
                cntinue = True
        else:
            rank = -1
            ratio = default_ratio
        return rank, ratio, cntinue

    def factorize_matrix(self, matrix, name, rank=-1, ratio=-1, verbose=True) -> FactorizedMatrix:
        # function that applies the svd technique to a single matrix and return the
        # compressed one (+ meta data?)
        print(f"Factorizing {name} matrix") if verbose else None
        if rank == -1 and ratio == -1:
            print(f"Warning: {name} rank or ratio must be defined!")
            return
        elif rank != -1 and ratio != -1:
            print(
                f"Warning: {name} rank and ratio are both defined, "
                "only one can be used at a time!"
            )
            return
        eq_rank = get_eq_rank(matrix.shape[0], matrix.shape[1])
        if rank == 0:
            rank = eq_rank
        elif ratio != -1:
            # rank = int(np.round(eq_rank * ratio))
            rank = int(eq_rank * ratio)
        elif rank > eq_rank:
            print(f"Warning: {name} rank is larger than equivalent rank!")
            return
        if self.use_local_cache and name in self.factorized_layers_cache:
            fact_mat = self.factorized_layers_cache[name]
        else:
            fact_mat = self._factorize_matrix(
                matrix=matrix, name=name, eq_rank=eq_rank, rank=rank, dev=self.dev, verbose=verbose
            )
        fact_mat.active_rank = rank

        if self.use_local_cache and name not in self.factorized_layers_cache:
            self.factorized_layers_cache[name] = fact_mat
            
        return fact_mat

    def _factorize_matrix(self, matrix, name, eq_rank, rank, dev, verbose=True) -> FactorizedMatrix:
        # function that applies the svd technique to a single matrix and return the
        # compressed one (+ meta data?)
        raise NotImplementedError("Subclasses should implement this method.")

    def _factorize_cleanup(self, name):
        """Free per-layer scaling data after compression.

        The default implementation covers the two most common patterns
        (column-only via ``scaling_dict`` and row+column via
        ``row_scaling_dict`` / ``column_scaling_dict``).  Subclasses with
        non-standard caches (e.g. ``pca_components``, ``grad_dict``) should
        override this method.
        """
        for attr in ("scaling_dict", "row_scaling_dict", "column_scaling_dict"):
            d = getattr(self, attr, None)
            if d is not None and name in d:
                del d[name]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def create_factorized_sequential(
        self, factorized_matrix: FactorizedMatrix, original_module
    ) -> nn.Module:
        dev = original_module.weight.device
        dtype = original_module.weight.dtype
        module_l = nn.Linear(
            original_module.in_features,
            factorized_matrix.active_rank,
            bias=False,
            dtype=dtype,
        )
        module_r = nn.Linear(
            factorized_matrix.active_rank,
            original_module.out_features,
            bias=False,
            dtype=dtype,
        )
        module_l = module_l.to(dev)
        module_r = module_r.to(dev)

        weight_l, weight_r = factorized_matrix.mat_l, factorized_matrix.mat_r
        module_l.weight.data.copy_(weight_r[: factorized_matrix.active_rank, :].to(dev))
        module_r.weight.data.copy_(weight_l[:, :factorized_matrix.active_rank].to(dev))
        module = weight_l = weight_r = None
        del weight_l, weight_r, module

        torch.cuda.empty_cache()

        # return nn.Sequential(module_l, module_r).to(dev)
        return SeqSVD(module_l, module_r, original_module.bias if hasattr(original_module, "bias")
                      else None).to(dev)
        # return SeqSVDMemViT(in_features=original_module.in_features, out_features=original_module.out_features, rank_r=factorized_matrix.active_rank, bias=(original_module.bias is not None), init_from=original_module).to(dev)

    def factorize_model(self, uncom_model, rank_dict, name_omit, verbose=True, apply_fact=True) -> dict:
        """
        Apply low-rank decomposition to the model in place. Note that name omit
        is supported implicitly as removing or not mentioning something in the
        compression ratio dict will resul in it not being compressed.

        Args:
            name (str): module name
            module (nn.Linear): the given Linear module
            raw_profile (dict): the raw profile of the given module
        """
        print("\nApplying factorization")
        dev = torch.device(torch.cuda.current_device())
        with torch.cuda.device(dev):
            torch.cuda.empty_cache()

        model = uncom_model.eval().cpu()
        # model = uncom_model.eval().to(dev)
        copied_modules = get_valid_layers(model, name_omit, white_list=[])
        print(rank_dict) if verbose else None
        plot_compression_rates(rank_dict) if verbose else None
        for name, module_sub in tqdm(copied_modules):
            # condition for not applying low rank
            if (rank_dict[name] == -1 or rank_dict[name] == 1.0):
                continue

            factorized = self.factorize_matrix(
                matrix=module_sub.weight,
                rank=rank_dict[name] if isinstance(rank_dict[name], int) else -1,
                ratio=rank_dict[name] if isinstance(rank_dict[name], float) else -1,
                name=name,
                verbose=verbose,
            )
            if factorized is None:
                print(f"Skipping {name} as factorization failed.") if verbose else None
                continue
            
            if not apply_fact:
                continue
            svd_replacement = self.create_factorized_sequential(
                factorized_matrix=factorized, original_module=module_sub
            )

            print(f"Applying low rank on {name:^10}, rank {rank_dict[name]}") if verbose else None

            base, localname = model, name
            while "." in localname:
                prefix, localname = localname.split(".", 1)
                base = base.__getattr__(prefix)

            setattr(base, localname, svd_replacement)

            with torch.cuda.device(dev):
                torch.cuda.empty_cache()
            gc.collect()


class SeqSVD(nn.Module):
    def __init__(self, mod_a, mod_b, bias=None):
        super().__init__()
        self.mod_a = mod_a
        self.mod_b = mod_b
        self.bias = bias
        # to avoid ERROR: falcon_mamba/modeling_falcon_mamba.py", line 497, in forward -- if is_fast_path_available and "cuda" in self.x_proj.weight.device.type:
        # AttributeError: 'SeqSVD' object has no attribute 'weight'
        self.dummy_weight = nn.Parameter(torch.empty(1)).to(mod_a.weight.device)
        self.weight = self.dummy_weight

    def forward(self, x):
        x = self.mod_a(x)
        x = self.mod_b(x)
        if self.bias is not None:
            x += self.bias
        return x

class SeqSVDMemViT(nn.Module):
    def __init__(self, in_features, out_features, rank_r, rank_q=None, bias=True, init_from=None, gy_ratio=0.05):
        """
        Args:
            in_features: input dimension (k)
            out_features: output dimension (m)
            rank_r: low-rank dimension for U,V
            rank_q: optional override for G,Y rank (if None, use gy_ratio)
            bias: whether to include bias
            init_from: optional nn.Linear to initialize from
            gy_ratio: fraction of original W size to allocate for G,Y (default 5%)
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank_r = rank_r

        # Determine rank_q automatically if not provided
        if rank_q is None:
            k, m = in_features, out_features
            rank_q = int(round((gy_ratio * k * m) / (k + m)))
        self.rank_q = rank_q

        # Define factors
        self.U = nn.Parameter(torch.randn(in_features, rank_r) * 0.02)
        self.V = nn.Parameter(torch.randn(out_features, rank_r) * 0.02)
        self.G = nn.Parameter(torch.randn(in_features, rank_q) * 0.02)
        self.Y = nn.Parameter(torch.randn(out_features, rank_q) * 0.02)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # Optional initialization from pretrained linear
        if init_from is not None:
            with torch.no_grad():
                W = init_from.weight.data  # [out_features, in_features]
                U_svd, S_svd, Vt_svd = torch.linalg.svd(W, full_matrices=False)
                # Fill U,V with top singular components
                self.U.copy_(Vt_svd[:rank_r, :].T)
                self.V.copy_(U_svd[:, :rank_r] * S_svd[:rank_r])
                # Fill G,Y with next components (or repeat if fewer)
                if rank_q > 0:
                    start = rank_r
                    end = min(W.shape[0], start + rank_q)
                    self.G.copy_(Vt_svd[start:end, :].T)
                    self.Y.copy_(U_svd[:, start:end] * S_svd[start:end])

    def forward(self, x):
        # x: [batch_size, in_features]
        out = (x @ self.U) @ self.V.T + (x @ self.G) @ self.Y.T
        if self.bias is not None:
            out = out + self.bias
        return out



class Hookstuff:
    # this is one unified hook to obtain the scalings for everybody.
    def __init__(self, model: nn.Module, name_omit=[], dump_shape=False, white_list=[], name_prefix=""):
        self.model = model
        self.name_omit = name_omit
        self.white_list = white_list
        self.dump_shape = dump_shape
        self.name_prefix = name_prefix

        self.column_scale =  {}
        self.row_scale = {}
        self.activation_cache = {}
        self.buf_1 = {}
        self.buf_2 = {}
        self.layer_trigger = None

        self.profile = {}
        self.profile_gout = {}
        self.input_shape = {}
        self.hooks = []
        # NOTE: if you print/ access a reversed object once, it is gone afterwards.
        self.cp_modules = reversed(
            get_valid_layers(model, self.name_omit, self.white_list)
        )
        self.bw_cp_modules = reversed(
            get_valid_layers(model, self.name_omit, self.white_list)
        )

    @staticmethod
    def _reshape_input(x):
        """Normalise activation / gradient tensor to 3-D ``(B, T, D)``."""
        if x.dim() > 3:      # e.g. ConvNeXt [B, H, W, D]
            x = x.reshape(x.shape[0], -1, x.shape[-1])
        elif x.dim() == 2:   # e.g. Mamba / OPT  [T, D]
            x = x.unsqueeze(0)
        return x

    def _maybe_record_shape(self, layer_name, x, module):
        """If ``dump_shape`` is active, record the input shape and return *True*.

        Typical usage inside ``_hook_fn``::

            x = self._reshape_input(input[0].detach().float())
            if self._maybe_record_shape(layer_name, x, module):
                return
        """
        if self.dump_shape:
            self.input_shape[layer_name] = list(x.shape)
            self.input_shape[layer_name].extend([module.out_features, 0])
            return True
        return False

    def _hook_fn(self, layer_name):
        def get_scaling_mat(module, input, output):
            pass

        return get_scaling_mat
    
    def _bw_hook_fn(self, layer_name):
        def get_scaling_mat_grad(module, ginput, goutput):
            pass

        return get_scaling_mat_grad
    
    def _perturb_hook_fn(self, layer_name):
        def perturb_activations(module, input, output):
            # This function can be used to perturb the weights of the layer
            # For example, you can add noise or apply some transformation
            # Here we just pass the input through without modification
            return output

        return perturb_activations

    def _register_hooks_recursive(self, cp_modules: dict, prefix=""):
        print("Registering forward hooks...")
        for name, layer in cp_modules:
            layer_name = self.name_prefix + name
            hook = layer.register_forward_hook(self._hook_fn(layer_name))
            self.hooks.append(hook)

    def _register_bw_hooks_recursive(self, cp_modules: dict, prefix=""):
        print("Registering backward hooks...")
        for name, layer in cp_modules:
            layer_name = self.name_prefix + name
            bw_hook = layer.register_full_backward_hook(self._bw_hook_fn(layer_name))
            self.hooks.append(bw_hook)

    def _register_bw_hooks_singular(self, cp_modules: dict, prefix=""):
        # Iterate through the layers
        for name, layer in cp_modules:
            # Remove the previous hook if it exists
            if self.hooks:
                prev_hook = self.hooks.pop()
                prev_hook.remove()
            
            # Register a new backward hook for the current layer
            layer_name = self.name_prefix + name
            bw_hook = layer.register_full_backward_hook(self._bw_hook_fn(layer_name))
            self.hooks.append(bw_hook)
            
            # Yield the current hook
            yield name, layer

    def _register_hooks_singular(self, cp_modules: dict, prefix=""):
        # Iterate through the layers
        for name, layer in cp_modules:
            # Remove the previous hook if it exists
            if self.hooks:
                prev_hook = self.hooks.pop()
                prev_hook.remove()
            
            # Register a new backward hook for the current layer
            layer_name = self.name_prefix + name
            hook = layer.register_forward_hook(self._perturb_hook_fn(layer_name))
            bw_hook = layer.register_full_backward_hook(self._bw_hook_fn(layer_name))
            self.hooks.append(hook)
            self.hooks.append(bw_hook)
            
            # Yield the current hook
            yield name, layer

    def attach_hooks(self):
        self._register_hooks_recursive(self.cp_modules)

    def attach_bw_hooks(self):
        self._register_bw_hooks_recursive(self.bw_cp_modules)

    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    # ---- shared helpers for backward hooks ----

    @staticmethod
    def clip_token_norms(tensor, min_ratio=0.1, max_ratio=10.0, boost=True):
        """
        Clip (and optionally boost) per-token gradient norms so they lie
        within ``[mean * min_ratio, mean * max_ratio]``.

        Parameters
        ----------
        tensor : Tensor  [B, T, D]
        min_ratio, max_ratio : float
            Bounds expressed as multiples of the per-batch mean norm.
        boost : bool
            When *True* (default) tokens whose norm is below
            ``mean * min_ratio`` are scaled **up**.  When *False* only the
            upper clip is applied (small / zero tokens are left untouched).

        Returns the rescaled tensor (new allocation, original is unchanged).
        """
        token_norms = torch.norm(tensor, dim=2, keepdim=True)

        mean_norms = token_norms.clone()
        mean_norms[token_norms < 1e-10] = 0
        batch_mean_norms = mean_norms.sum(dim=1) / torch.maximum(
            (mean_norms > 0).float().sum(dim=1),
            torch.ones_like((mean_norms > 0).float().sum(dim=1)),
        )
        batch_mean_norms = torch.maximum(
            batch_mean_norms, torch.ones_like(batch_mean_norms) * 1e-10
        )

        max_allowed = (batch_mean_norms * max_ratio).unsqueeze(1)
        safe_norms = torch.clamp(token_norms, min=1e-10)

        if boost:
            min_allowed = (batch_mean_norms * min_ratio).unsqueeze(1)
            target_norms = torch.clamp(token_norms, min=min_allowed, max=max_allowed)
            # Guard near-zero tokens: leave them untouched instead of mega-boosting.
            target_norms = torch.where(token_norms < 1e-10, token_norms, target_norms)
        else:
            target_norms = torch.clamp(token_norms, max=max_allowed)

        scale_factors = target_norms / safe_norms
        return tensor * scale_factors

    def _accumulate_scaling(self, layer_name, row_scaling, column_scaling):
        """
        Double-buffered, pinned-memory accumulation of row/column scaling
        matrices from GPU to CPU.
        """
        if layer_name not in self.row_scale:  # first batch
            self.buf_1[layer_name] = torch.zeros_like(row_scaling, device='cpu', pin_memory=True)
            self.buf_2[layer_name] = torch.zeros_like(column_scaling, device='cpu', pin_memory=True)
            self.row_scale[layer_name] = torch.zeros_like(row_scaling, device='cpu', pin_memory=True)
            self.column_scale[layer_name] = torch.zeros_like(column_scaling, device='cpu', pin_memory=True)

            torch.cuda.synchronize()
            self.buf_1[layer_name].copy_(row_scaling, non_blocking=True)
            self.buf_2[layer_name].copy_(column_scaling, non_blocking=True)
        else:
            if self.layer_trigger == layer_name:
                torch.cuda.synchronize()
            self.row_scale[layer_name] += self.buf_1[layer_name]
            self.column_scale[layer_name] += self.buf_2[layer_name]

            self.buf_1[layer_name].copy_(row_scaling, non_blocking=True)
            self.buf_2[layer_name].copy_(column_scaling, non_blocking=True)


class ShapeHook(Hookstuff):
    def _hook_fn(self, layer_name, last_feat=False):
        def get_intermediate_shapes(module, input, output):
            x = self._reshape_input(input[0].detach().float())
            self.input_shape[layer_name] = list(x.shape)
            self.input_shape[layer_name].extend([module.out_features, 0])
            del input, output, module, x
            return
        return get_intermediate_shapes