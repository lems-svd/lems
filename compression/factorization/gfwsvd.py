import gc

import torch
from torch import nn

from ._interface import BaseFactorization
from ._interface import FactorizedMatrix
from tqdm import tqdm

# Requires the CuPy package (pip install cupy-cuda12x) for GPU-accelerated
# sparse SVD.  See https://cupy.dev for installation instructions.
import cupy as cp
from cupyx.scipy.sparse.linalg import svds, LinearOperator
import numpy as np

from ._interface import get_valid_layers

GRAD_ACC_STEPS = 16 # Number of gradient accumulation steps, can be adjusted based on memory constraints

def is_positive_definite(matrix: np.ndarray) -> bool:
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False

# taken from main branch of public code (works better and faster in our experience)
# https://github.com/sayankotor/FisherKronecker/blob/main/llama/compress_llama_with_kronsvd_llama2.py#L125C5-L137C22
def matrix_sqrt_invsqrt(X: torch.Tensor, lmbd: float = 1e-6, alpha_increase_factor=1e-1, max_reg_tries=10, reg_alpha: float = 1e-1):
    def regularize_factor(XF_reg, factor, diag_mean, max_reg_tries, reg_alpha, eye):
        for i in range(max_reg_tries + 1):
            if is_positive_definite(XF_reg):
                print(f"  Factor is positive definite (alpha={reg_alpha:.2e})")
                break
            if i == max_reg_tries:
                raise RuntimeError(f"Failed to regularize factor after {max_reg_tries} attempts.")
            print(f"  Regularizing factor (try {i+1}, alpha={reg_alpha:.2e})")
            reg_alpha += alpha_increase_factor
            XF_reg = (1 - reg_alpha) * factor + reg_alpha * eye * diag_mean
        return XF_reg
    out_features, in_features = X.shape
    XF = X if not type(X) is torch.Tensor else X.cpu().numpy()
    eye_X = np.eye(in_features, dtype=np.float32)
    diag_mean_X = max(np.mean(np.diag(XF)), 1e-6)
    XF_reg = np.copy(XF)
    XF_reg = regularize_factor(XF_reg, X, diag_mean_X, max_reg_tries, reg_alpha, eye_X)

    try:
        X_chol = np.linalg.cholesky(XF_reg)
        print("  Cholesky decomposition successful.")
    except np.linalg.LinAlgError as e:
        print(f"ERROR: Cholesky decomposition failed: {e}")
        raise e
    try:
        inv_X_chol = np.linalg.inv(X_chol)
        print("  Cholesky factor inverses computed.")
    except np.linalg.LinAlgError as e:
        print(f"ERROR: Failed to invert Cholesky factors: {e}")
        raise e
    return torch.tensor(X_chol, dtype=X.dtype), torch.tensor(inv_X_chol, dtype=X.dtype)

# taken from release branch of public code
# https://github.com/sayankotor/FisherKronecker/blob/release/gfwsvd/compress_llama_with_kronsvd.py#L75
# def matrix_sqrt_invsqrt(X: torch.Tensor, lmbd: float = 1e-6):
#     """
#     Computes matrix square root and inverse square root using eigendecomposition.

#     This function is crucial for the Kronecker-factored SVD method, as it transforms
#     the Kronecker factors into a form suitable for stable SVD decomposition.

#     Args:
#         X: Input matrix (must be symmetric positive semi-definite)
#         lmbd: Initial regularization parameter for numerical stability

#     Returns:
#         Tuple of (X^{1/2}, X^{-1/2}) matrices
#     """
#     # Iteratively increase regularization until matrix is positive definite
#     while lmbd < 0.1:
#         eigvals, Q = torch.linalg.eigh(X + torch.eye(X.shape[0], device=X.device) * lmbd * X.diag())
#         if torch.all(eigvals.real > -1e-7):
#             break
#         lmbd *= 2

#     # Clamp eigenvalues to ensure numerical stability
#     eigvals = eigvals.clamp(1e-12)

#     # Compute matrix functions using eigendecomposition
#     X_sqrt = Q @ torch.diag(eigvals.sqrt()) @ Q.T
#     X_inv_sqrt = Q @ torch.diag(eigvals.rsqrt()) @ Q.T

#     return X_sqrt, X_inv_sqrt


def get_kron_factors(list_of_grads, top_k=10, layer_name="linear", device="cuda:0", chunk_size=4):
    """
    Perform parallel by input layers Fisher Matrix approximation in the form of Kronecker Decomposition.
    """

    def matvec(vec, grad_vectors, chunk_size=4):
        k, m, n = grad_vectors.shape
        V = vec.reshape(n, n, order="F")
        result = cp.zeros((m, m), dtype=cp.float32)
        for i in range(0, k, chunk_size):
            chunk = grad_vectors[i : i + chunk_size]
            prod = chunk @ V @ chunk.transpose(0, 2, 1)
            result += cp.sum(prod, axis=0)
        return (result / k).T.ravel()

    def r_matvec(vec, grad_vectors, chunk_size=4):
        k, m, n = grad_vectors.shape
        V = vec.reshape(m, m, order="F")
        result = cp.zeros((n, n), dtype=cp.float32)
        for i in range(0, k, chunk_size):
            chunk = grad_vectors[i : i + chunk_size]
            prod = chunk.transpose(0, 2, 1) @ V @ chunk
            result += cp.sum(prod, axis=0)
        return (result / k).T.ravel()

    device_id = 0
    num_devices = cp.cuda.runtime.getDeviceCount()
    device_pool = [cp.cuda.Device(i) for i in range(num_devices)]

    m, n = list_of_grads[0].shape

    # ADDDED
    total_grad_norm = 0.0
    for i, g in enumerate(list_of_grads):
        # Check for NaN or Inf values first.
        if not torch.all(torch.isfinite(g)):
            print(f"⚠️  Warning: Gradients for layer {layer_name} {i}/{len(list_of_grads)} are non-finite. Replacing with small value.")
            list_of_grads[i] = torch.where(torch.isfinite(g) == False, torch.tensor(1e-15, device=g.device), g)
        if not torch.all(torch.isfinite(list_of_grads[i])):
            print("Still not finite after replacement, skipping SVD.")
        total_grad_norm += torch.sum(torch.abs(g))

    # If gradients are non-finite OR all-zero, skip SVD and return zero factors.
    if total_grad_norm < 1e-9:
        print(f"⚠️  Warning: Gradients for layer {layer_name} are all-zero. Skipping SVD.")
        zero_factor_m = torch.zeros((m, m), dtype=torch.float32)
        zero_factor_n = torch.zeros((n, n), dtype=torch.float32)
        return zero_factor_m, zero_factor_n
    try:
        with device_pool[device_id]:
            print(n, m)
            grad_vectors = cp.stack([cp.asarray(grad).reshape(m, n, order="F") for grad in list_of_grads], dtype=cp.float32)
            linop = LinearOperator(
                shape=(m * m, n * n),
                matvec=lambda vec: matvec(vec, grad_vectors),
                rmatvec=lambda vec: r_matvec(vec, grad_vectors),
                dtype=cp.float32,
            )

            u, s, vt = svds(linop, k=top_k, return_singular_vectors=True)
            print(f"✔ Layer {layer_name} on device {device_id} done | singular values: {s}")
            sidx = cp.argsort(-s)
            s = s[sidx]
            u = u[:, sidx]
            v = vt[sidx, :].T

            XF = (u[:, 0] * s[0] ** 0.5).reshape(m, m, order="F")
            YF = (s[0] ** 0.5 * v[:, 0]).reshape(n, n, order="F")

            return torch.tensor(XF.get(), dtype=torch.float32), torch.tensor(YF.get(), dtype=torch.float32)
    except:
        print(f"⚠️  Warning: SVD failed for layer {layer_name}. Returning zero factors.")
        zero_factor_m = torch.zeros((m, m), dtype=torch.float32)
        zero_factor_n = torch.zeros((n, n), dtype=torch.float32)
        return zero_factor_m, zero_factor_n


class GFWSVDFactorization(BaseFactorization):
    def __init__(self, processing_chunk_size=64, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_dict = {}
        self.scaling_matrices = {}
        self.processing_chunk_size = processing_chunk_size

    def _compute_scaling(
        self,
        model,
        hook_module,
        name_prefix,
        calib_data,
        name_omit,
        mixup_fn=None,
        white_list=[],
        tqdm_message="Gathering ",
    ):
        model = model.to(self.dev)

        for p in model.parameters():
            p.requires_grad = False

        copied_modules = get_valid_layers(
            hook_module,
            name_omit,
            white_list=white_list,
        )

        for _, module in copied_modules:
            if isinstance(module, nn.Linear):
                for n, p in module.named_parameters():
                    if "bias" not in n:
                        p.requires_grad = True

        model.zero_grad(set_to_none=True)

        # CPU fp32 accumulation.
        # This avoids persistent fp32 GPU copies.
        grad_accum_cpu = {}
        accum_steps = 0

        # Reusable CPU staging buffers, keyed by shape.
        # This avoids allocating a new CPU tensor for every layer every step.
        staging_buffers = {}

        dev = torch.device(self.dev)
        use_pinned_cpu = dev.type == "cuda"

        def get_staging_buffer(grad):
            shape_key = tuple(grad.shape)

            if shape_key not in staging_buffers:
                staging_buffers[shape_key] = torch.empty(
                    grad.shape,
                    dtype=torch.float32,
                    device="cpu",
                    pin_memory=use_pinned_cpu,
                )

            return staging_buffers[shape_key]

        def accumulate_current_grads_cpu_fp32():
            """
            Copy each current gradient directly into a reusable CPU fp32 buffer,
            then add it to a CPU fp32 accumulator.

            Important: this does NOT create a persistent fp32 GPU tensor.
            """
            for name, module in copied_modules:
                key = name_prefix + name

                grad = module.weight.grad
                if grad is None:
                    continue

                if key not in grad_accum_cpu:
                    grad_accum_cpu[key] = torch.zeros(
                        grad.shape,
                        dtype=torch.float32,
                        device="cpu",
                    )

                staging = get_staging_buffer(grad)

                # Blocking copy is intentional here:
                # it lets us safely clear module.weight.grad immediately after.
                # The copy also casts into fp32 on CPU.
                staging.copy_(grad.detach(), non_blocking=False)

                grad_accum_cpu[key].add_(staging)

                # Free this grad buffer as soon as possible.
                module.weight.grad = None

        def flush_grad_accum(denom):
            if denom <= 0:
                return

            for key, grad_sum in grad_accum_cpu.items():
                grad_cpu = grad_sum / denom
                self.grad_dict.setdefault(key, []).append(grad_cpu)

            grad_accum_cpu.clear()

        for batch in tqdm(
            calib_data,
            desc=tqdm_message + " (generalized fisher information)",
        ):
            if self.vision:
                loss_fn = nn.CrossEntropyLoss()
                data, target = batch

                if mixup_fn is not None:
                    model_inputs, target_mix = mixup_fn(data, target)
                else:
                    model_inputs, target_mix = data, target

                model_inputs = model_inputs.to(self.dev, non_blocking=True)
                target_mix = target_mix.to(self.dev, non_blocking=True)

                out = model(model_inputs)
                loss = loss_fn(out, target_mix)
                batch_dim = data.shape[0]

            else:
                input_ids = batch["input_ids"].to(self.dev, non_blocking=True)

                out = model(
                    input_ids=input_ids[:, :-1],
                    labels=input_ids[:, 1:],
                )

                loss = out.loss
                batch_dim = input_ids.shape[0]

            large_batch = batch_dim > GRAD_ACC_STEPS

            if large_batch and accum_steps > 0:
                flush_grad_accum(accum_steps)
                accum_steps = 0

            loss.backward()

            # Copy this backward pass into CPU fp32 accumulation.
            accumulate_current_grads_cpu_fp32()
            accum_steps += 1

            # Clear any remaining grad buffers.
            model.zero_grad(set_to_none=True)

            if large_batch:
                flush_grad_accum(1)
                accum_steps = 0

            elif accum_steps == GRAD_ACC_STEPS:
                flush_grad_accum(GRAD_ACC_STEPS)
                accum_steps = 0

            # Help Python release references from this iteration.
            del loss, out

        if accum_steps > 0:
            flush_grad_accum(accum_steps)

        model.zero_grad(set_to_none=True)

        del grad_accum_cpu
        del staging_buffers

        gc.collect()
        torch.cuda.empty_cache()

        model = model.eval()
        return

    def _factorize_cleanup(self, name):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if name in self.grad_dict:
            del self.grad_dict[name]
        if name in self.scaling_matrices:
            del self.scaling_matrices[name]

    def _factorize_matrix(self, matrix, eq_rank, rank, name, dev, verbose=False):
        dev = matrix.device
        dtype = matrix.dtype

        if name in self.scaling_matrices:
            A_scale_inv, B_scale_inv, A_scale, B_scale = self.scaling_matrices[name]
        else:
            # OPTIMIZATION: Pass CPU gradients and chunk size to the memory-efficient function
            print(len(self.grad_dict[name])) if verbose else None
            A, B = get_kron_factors(
                torch.stack(self.grad_dict[name]).float(), 
                top_k=1, 
                layer_name=name,
                device=dev,
                chunk_size=self.processing_chunk_size
            )
            A_scale, A_scale_inv = matrix_sqrt_invsqrt(A)
            B_scale, B_scale_inv = matrix_sqrt_invsqrt(B)
            self.scaling_matrices[name] = [A_scale_inv, B_scale_inv, A_scale, B_scale]
            print(f"Hessian whitening matrix A {name} min: {torch.diag(A_scale).min()}, max: {torch.diag(A_scale).max()}, median: {torch.diag(A_scale).median()}") if verbose else None
            print(f"Hessian whitening matrix B {name} min: {torch.diag(B_scale).min()}, max: {torch.diag(B_scale).max()}, median: {torch.diag(B_scale).median()}") if verbose else None

        if rank == 0:
            rank = eq_rank
        elif rank > eq_rank:
            print(f"Warning: {name} rank ({rank}) is larger than equivalent rank ({eq_rank})!")
            rank = eq_rank

        mat_scaled = A_scale.to(dev) @ matrix.to(torch.float32).to(dev) @ B_scale.to(dev)
        u, s, vh = torch.linalg.svd(mat_scaled, full_matrices=False)
        
        active_rank = min(rank, len(s))
        s_val = torch.sqrt(s[:active_rank])

        mat_l = (A_scale_inv.to(dev) @ u[:, :active_rank]) * s_val
        mat_r = (vh[:active_rank, :] @ B_scale_inv.to(dev)) * s_val.unsqueeze(1)
        
        return FactorizedMatrix(
            mat_l=mat_l.to(dtype),
            mat_r=mat_r.to(dtype),
            eq_rank=eq_rank,
            active_rank=active_rank,
            singular_values=s,
        )