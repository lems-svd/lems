import gc
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
# The external code uses a custom IncrementalPCAonGPU.
# For this example, we"ll use scikit-learn"s IncrementalPCA, which works on the CPU.
# A custom GPU implementation would be needed for optimal performance.
from sklearn.decomposition import IncrementalPCA

from ._interface import BaseFactorization
from ._interface import FactorizedMatrix
from ._interface import Hookstuff
from ._interface import get_valid_layers, get_eq_rank

class IncrementalPCAonGPU():
    """
    An implementation of Incremental Principal Components Analysis (IPCA) that leverages PyTorch for GPU acceleration.

    This class provides methods to fit the model on data incrementally in batches, and to transform new data 
    based on the principal components learned during the fitting process.

    Attributes:
        n_components (int, optional): Number of components to keep. If `None`, it"s set to the minimum of the 
                                      number of samples and features. Defaults to None.
        whiten (bool): When True, the `components_` vectors are divided to ensure uncorrelated outputs with 
                       unit component-wise variances. Defaults to False.
        copy (bool): If False, input data will be overwritten. Defaults to True.
        batch_size (int, optional): The number of samples to use for each batch. If `None`, it"s inferred from 
                                    the data and set to `5 * n_features`. Defaults to None.
    """

    def __init__(self, device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")), n_components=None, *, whiten=False, copy=True, batch_size=None):
        self.n_components = n_components
        self.whiten = whiten
        self.copy = copy
        self.batch_size = batch_size
        self.device = device
        
        # Set n_components_ based on n_components if provided
        if n_components:
            self.n_components_ = n_components

        # Initialize attributes to avoid errors during the first call to partial_fit
        self.mean_ = None  # Will be initialized properly in partial_fit based on data dimensions
        self.var_ = None  # Will be initialized properly in partial_fit based on data dimensions
        self.n_samples_seen_ = 0

    def _validate_data(self, X, dtype=torch.float32, copy=True):
        """
        Validates and converts the input data `X` to the appropriate tensor format.

        This method ensures that the input data is in the form of a PyTorch tensor and resides on the correct device (CPU or GPU). 
        It also provides an option to create a copy of the tensor, which is useful when the input data should not be overwritten.

        Args:
            X (Union[np.ndarray, torch.Tensor]): Input data which can be a numpy array or a PyTorch tensor.
            dtype (torch.dtype, optional): Desired data type for the tensor. Defaults to torch.float32.
            copy (bool, optional): Whether to clone the tensor. If True, a new tensor is returned; otherwise, the original tensor 
                                   (or its device-transferred version) is returned. Defaults to True.

        Returns:
            torch.Tensor: Validated and possibly copied tensor residing on the specified device.
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=dtype).to(self.device)
        elif X.device != self.device:
            X = X.to(self.device)
        if copy:
            X = X.clone()
        return X

    @staticmethod
    def _incremental_mean_and_var(X, last_mean, last_variance, last_sample_count):
        """
        Computes the incremental mean and variance for the data `X`.

        Args:
            X (torch.Tensor): The batch input data tensor with shape (n_samples, n_features).
            last_mean (torch.Tensor): The previous mean tensor with shape (n_features,).
            last_variance (torch.Tensor): The previous variance tensor with shape (n_features,).
            last_sample_count (torch.Tensor): The count tensor of samples processed before the current batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, int]: Updated mean, variance tensors, and total sample count.
        """
        if X.shape[0] == 0:
            return last_mean, last_variance, last_sample_count

        # If last_mean or last_variance is None, initialize them with zeros
        if last_mean is None:
            last_mean = torch.zeros(X.shape[1], device=X.device)
        if last_variance is None:
            last_variance = torch.zeros(X.shape[1], device=X.device)

        new_sample_count = X.shape[0]
        new_mean = torch.mean(X, dim=0)
        new_sum_square = torch.sum((X - new_mean) ** 2, dim=0)
        
        updated_sample_count = last_sample_count + new_sample_count
        
        updated_mean = (last_sample_count * last_mean + new_sample_count * new_mean) / updated_sample_count
        updated_variance = (last_variance * (last_sample_count + new_sample_count * last_mean ** 2) + new_sum_square + new_sample_count * new_mean ** 2) / updated_sample_count - updated_mean ** 2
        
        return updated_mean, updated_variance, updated_sample_count

    @staticmethod
    def _svd_flip(u, v, u_based_decision=True):
        """
        Adjusts the signs of the singular vectors from the SVD decomposition for deterministic output.

        This method ensures that the output remains consistent across different runs.

        Args:
            u (torch.Tensor): Left singular vectors tensor.
            v (torch.Tensor): Right singular vectors tensor.
            u_based_decision (bool, optional): If True, uses the left singular vectors to determine the sign flipping. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Adjusted left and right singular vectors tensors.
        """
        if u_based_decision:
            max_abs_cols = torch.argmax(torch.abs(u), dim=0)
            signs = torch.sign(u[max_abs_cols, range(u.shape[1])])
        else:
            max_abs_rows = torch.argmax(torch.abs(v), dim=1)
            signs = torch.sign(v[range(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs[:, None]
        return u, v
    
    def fit(self, X, check_input=True):
        """
        Fits the model with data `X` using minibatches of size `batch_size`.

        Args:
            X (torch.Tensor): The input data tensor with shape (n_samples, n_features).

        Returns:
            IncrementalPCAGPU: The fitted IPCA model.
        """
        if check_input:
            X = self._validate_data(X)
        n_samples, n_features = X.shape
        if self.batch_size is None:
            self.batch_size_ = 5 * n_features
        else:
            self.batch_size_ = self.batch_size

        for start in range(0, n_samples, self.batch_size_):
            end = min(start + self.batch_size_, n_samples)
            X_batch = X[start:end]
            self.partial_fit(X_batch, check_input=False)

        return self

    def partial_fit(self, X, check_input=True):
        """
        Incrementally fits the model with batch data `X`.

        Args:
            X (torch.Tensor): The batch input data tensor with shape (n_samples, n_features).
            check_input (bool, optional): If True, validates the input. Defaults to True.

        Returns:
            IncrementalPCAGPU: The updated IPCA model after processing the batch.
        """
        first_pass = not hasattr(self, "components_")

        if check_input:
            X = self._validate_data(X)
        n_samples, n_features = X.shape

        if first_pass:
            self.components_ = None
        if self.n_components is None:
            self.n_components_ = min(n_samples, n_features)

        col_mean, col_var, n_total_samples = self._incremental_mean_and_var(
            X, self.mean_, self.var_, torch.tensor([self.n_samples_seen_], device=self.device)
        )

        # Whitening
        if self.n_samples_seen_ == 0:
            X -= col_mean
        else:
            col_batch_mean = torch.mean(X, dim=0)
            X -= col_batch_mean
            mean_correction_factor = torch.sqrt(
                torch.tensor((self.n_samples_seen_ / n_total_samples.item()) * n_samples, device=self.device)
            )
            mean_correction = mean_correction_factor * (self.mean_ - col_batch_mean)

            if self.singular_values_ is not None and self.components_ is not None:
                X = torch.vstack(
                    (
                        self.singular_values_.view((-1, 1)) * self.components_,
                        X,
                        mean_correction,
                    )
                )

        U, S, Vt = torch.linalg.svd(X, full_matrices=False)
        U, Vt = self._svd_flip(U, Vt, u_based_decision=False)
        explained_variance = S**2 / (n_total_samples.item() - 1)
        explained_variance_ratio = S**2 / torch.sum(col_var * n_total_samples.item())

        self.n_samples_seen_ = n_total_samples.item()
        self.components_ = Vt[: self.n_components_]
        self.singular_values_ = S[: self.n_components_]
        self.mean_ = col_mean
        self.var_ = col_var
        self.explained_variance_ = explained_variance[: self.n_components_]
        self.explained_variance_ratio_ = explained_variance_ratio[: self.n_components_]
        if self.n_components_ not in (n_samples, n_features):
            self.noise_variance_ = explained_variance[self.n_components_ :].mean().item()
        else:
            self.noise_variance_ = 0.0
        return self

    def transform(self, X):
        """
        Applies dimensionality reduction to `X`.

        The input data `X` is projected on the first principal components previously extracted from a training set.

        Args:
            X (torch.Tensor): New data tensor with shape (n_samples, n_features) to be transformed.

        Returns:
            torch.Tensor: Transformed data tensor with shape (n_samples, n_components).
        """
        X = X.to(self.device)
        return torch.mm(X - self.mean_, self.components_.T)
    
    def inverse_transform(self, X):
        """
        Transform data back to its original space.

        In other words, return an input `X_original` whose transform would be X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            New data, where `n_samples` is the number of samples
            and `n_components` is the number of components.

        Returns
        -------
        X_original array-like of shape (n_samples, n_features)
            Original data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        """
        X = X.to(self.device)
        return X @ self.components_ + self.mean_
    
    def save_vars(self, save_path):
        """
        Move all tensor to cpu and save all the varialbes expect the "device"
        """
        state_dict = vars(self).copy()
        for key, value in state_dict.items():   
            if type(value) is torch.Tensor:
                state_dict[key] = value.detach().cpu()
        state_dict.pop("device")
        torch.save(state_dict, save_path)
    
    def load_vars(self, load_path):
        state_dict = torch.load(load_path)
        for key, value in state_dict.items():
            vars(self)[key] = value.to(self.device) if type(value) is torch.Tensor else value

    def get_vars(self):
        return vars(self)


class DOBI_SVD_Hook(Hookstuff):
    """
    Hook to capture the right singular vectors (V) of the intermediate matrix A = x @ W.T
    and fit them into an Incremental PCA model for each layer.
    """
    def __init__(self, pca_instances, ranks, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pca_instances = pca_instances
        self.ranks = ranks
        self.acc_bs = 2
        self.acc_list_dict = {}
        for layer_name in self.pca_instances.keys():
            self.acc_list_dict[layer_name] = []
        
    def _hook_fn(self, layer_name, last_feat=False):
        def collect_principal_vectors(module, input, output):
            if layer_name not in self.pca_instances:
                return
            x = input[0].detach().float()
            if x.dim() == 3:
                x = x.squeeze(0)
            assert x.dim() == 2, f"Expected 2D input, got {x.dim()}D"

            W_T = module.weight.T.float()
            A = x @ W_T
            smaller_dim = min(A.shape)
            smaller_dim_w = min(W_T.shape)
            
            correction = smaller_dim / smaller_dim_w
            target_rank = self.ranks.get(layer_name, -1)
            target_rank = int(target_rank * correction)
            
            if target_rank == -1:
                return

            try:
                # Perform SVD on the intermediate matrix A
                _U, _S, Vf = torch.svd_lowrank(A, q=target_rank, niter=2)
            except torch._C._LinAlgError:
                print(f"Warning: SVD failed for layer {layer_name}. Skipping batch.")
                return

            # Fit the resulting right singular vectors into the PCA model
            Vf1=Vf.detach().to("cpu")
            self.acc_list_dict[layer_name].append(Vf1)
            batches_to_accumulate = max(1, int(correction))
            if len(self.acc_list_dict[layer_name]) == batches_to_accumulate:
                Vf_full = torch.concatenate(self.acc_list_dict[layer_name], axis=1).to("cuda")
                ipca = self.pca_instances.get(layer_name)
                if ipca is not None:
                    ipca.partial_fit(Vf_full.T)
                self.acc_list_dict[layer_name]=[]
                del Vf_full
                torch.cuda.empty_cache()
            orig_dtype = output.dtype
            orig_device = output.device
            output = _U @ torch.diag_embed(_S) @ Vf.T

            output = output + module.bias if module.bias is not None else output
            del Vf, _U
            return output.to(orig_device).to(orig_dtype) 

        return collect_principal_vectors


class DOBI_SVDFactorization(BaseFactorization):
    """
    Implements matrix factorization by projecting weights onto the principal components
    of the activation-output space, with a soft truncation factor.
    
    This method is based on collecting right singular vectors from the SVD of (x @ W.T)
    over a calibration dataset.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pca_instances = {}
        self.pca_components = {}

        print("Warning: DOBI-SVD is finnicky and does only support the following settings:", 
              "one_shot_factorization=True, blockwise_factorization=False, progressive_compression=False")
        self.one_shot_factorization = True
        self.blockwise_factorization = False
        self.progressive_compression = False
        # dobi does not support multi caching as its inversing only works on smaller ratios than was used
        # for computation in the first place.
        self.use_cache = False
    
    @property
    def post_search_calibration(self):
        return True if self._do_post_calibration == "default" else self._do_post_calibration

    @torch.no_grad()
    def _compute_scaling(self, model, hook_module, name_prefix, calib_data, name_omit, mixup_fn=None, white_list=[], tqdm_message="Gathering "):
        """
        Runs calibration data through the model to compute principal components for each target layer.
        This is the equivalent of the `compute_scaling` step in ASVD.
        """
        model = model.to("cuda")
        model.eval()
        
        # Initialize an IncrementalPCA instance for each layer to be factorized
        ranks = {}
        copied_modules = get_valid_layers(
            model=hook_module, name_omit=name_omit, white_list=white_list)
        for name, module in copied_modules:
            full_name = name_prefix + name
            n, m = module.weight.shape
            eq_rank = get_eq_rank(n, m)
            rank = self.calibration_ranks[full_name]
            if isinstance(rank, float) and (0 < rank < 1):
                rank = int(rank * eq_rank)
            elif isinstance(rank, int) and (rank > 0):
                rank = min(rank, int(eq_rank))
            else:
                raise ValueError(f"Invalid rank specification {rank} for layer {full_name}. Must be a positive int or a float in (0, 1).")
            ranks[full_name] = rank
            self.pca_instances[full_name] = IncrementalPCAonGPU(n_components=ranks[full_name])
        # Instantiate and attach hooks
        extractor = DOBI_SVD_Hook(
            pca_instances=self.pca_instances,
            ranks=ranks, model=hook_module,
            name_omit=name_omit, dump_shape=False,
            name_prefix=name_prefix, white_list=white_list
        )
        extractor.attach_hooks()

        with torch.no_grad():
            for batch in tqdm(calib_data, desc="Calibrating for Dobi-SVD"):
                if self.vision:
                    inputs, _ = batch
                    model(inputs.to(self.dev))
                else:
                    inputs = batch["input_ids"].to(self.dev)

                    model(inputs)

        extractor.clear_hooks()

        # Extract the learned components from each PCA instance
        for name, ipca in self.pca_instances.items():
            if hasattr(ipca, "components_"):
                # components_ has shape (n_components, n_features)
                self.pca_components[name] = ipca.components_.clone().detach()
        
        print("Principal component computation finished.")
        del self.pca_instances
        torch.cuda.empty_cache()

    def _factorize_cleanup(self, name):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        del self.pca_components[name]

    def _factorize_matrix(self, matrix, name, eq_rank, rank, dev, verbose=False):
        """
        Factorizes the matrix by creating a new low-rank matrix W_new = (P @ W),
        and then decomposing W_new into mat_l and mat_r.
        """
        if name not in self.pca_components:
            raise ValueError(f"Principal components were not computed for layer {name}. "
                             f"Ensure it was included in the ranks dictionary during calibration.")

        W = matrix.to(dev).float()  # Original weight matrix

        # Principal components from sklearn have shape (rank, out_features)
        # This corresponds to V_pca.T in the conceptual formula
        V_pca_T = self.pca_components[name].to(dev).float()
        V_pca = V_pca_T.T  # V_pca has shape (out_features, rank)

        G = torch.eye(rank, device=dev)

        # The external code calculates a new weight matrix W_new = (V_pca @ G @ V_pca.T) @ W.
        # We need to decompose this into two matrices, mat_l and mat_r,
        # such that mat_l @ mat_r approximates W_new.
        # A valid decomposition is:
        # mat_l = V_pca
        # mat_r = G @ V_pca.T @ W

        W_new = (((W.T @ V_pca)[:,:rank]) @ G @V_pca.T[:rank]).T

        W_T = W_new.T.to(torch.float32)
        U, S, V = torch.svd_lowrank(W_T, q=int(rank), niter = 10)

        diag_S = torch.diag(S)
        sqrt_S = torch.sqrt(diag_S)
        A_weight_T = (U @ sqrt_S).to(torch.float16)
        B_weight_T = (sqrt_S @V.T).to(torch.float16)

        print(A_weight_T.T.shape) if verbose else None
        print(B_weight_T.T.shape) if verbose else None
        
        mat_l = V_pca
        mat_r = G @ V_pca_T @ W
        print(mat_l.shape) if verbose else None
        print(mat_r.shape) if verbose else None


        return FactorizedMatrix(
            mat_l=B_weight_T.T.cpu(),
            mat_r=A_weight_T.T.cpu(),
            eq_rank=eq_rank,
            active_rank=rank,
            singular_values=None  # This method doesn't directly compute singular values
        )