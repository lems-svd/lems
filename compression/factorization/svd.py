import torch
import gc

from ._interface import BaseFactorization
from ._interface import FactorizedMatrix


class SVDFactorization(BaseFactorization):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _factorize_matrix(self, matrix, name, eq_rank, rank, dev, verbose=False):
        if rank == 0:
            rank = eq_rank
        elif rank > eq_rank:
            print(f"Warning: {name} rank is larger than equivalent rank!")
            return

        mat_scaled = matrix.to(dev)
        
        # Convert to float32 to avoid "svd_cuda_gesvdj" error for attempting svd on float16
        dtype = mat_scaled.dtype
        mat_scaled = mat_scaled.float()
            
        u, s, vh = torch.linalg.svd(mat_scaled, full_matrices=False)
        s_val = torch.sqrt(torch.diag(s))  # half singular value
        mat_l = u @ s_val
        mat_l = mat_l[:, :rank].cpu().to(dtype)
        mat_r = s_val @ vh
        mat_r = mat_r[:rank, :].cpu().to(dtype)
        
        torch.cuda.empty_cache()
        gc.collect()

        return FactorizedMatrix(
            mat_l=mat_l,  # Left singular vectors
            mat_r=mat_r,  # Right singular vectors
            eq_rank=eq_rank,  # Equivalent rank
            active_rank=rank,  # Active rank
            singular_values=s
        )
