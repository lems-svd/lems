"""
Shared dataclasses for low-rank factorized weight representations.
"""

from dataclasses import dataclass, field, InitVar
import torch


@dataclass
class FactorizedMatrix:
    """
    Stores the result of a low-rank matrix factorization  W ≈ mat_l @ mat_r.

    The full-rank factors are kept internally; ``mat_l`` / ``mat_r`` properties
    return views sliced to ``active_rank`` so that the rank can be changed
    dynamically without re-factorising.
    """
    mat_l: InitVar[torch.Tensor] = None
    mat_r: InitVar[torch.Tensor] = None
    eq_rank: int = 0
    active_rank: int = 0
    singular_values: torch.Tensor = None

    _mat_l: torch.Tensor = field(init=False, default=None, repr=False)
    _mat_r: torch.Tensor = field(init=False, default=None, repr=False)

    @property
    def mat_l(self):
        return self._mat_l[:, :self.active_rank]

    @property
    def mat_r(self):
        return self._mat_r[:self.active_rank, :]

    @mat_l.setter
    def mat_l_view(self, value):
        self._mat_l = value

    @mat_r.setter
    def mat_r_view(self, value):
        self._mat_r = value

    def __post_init__(self, mat_l, mat_r):
        if mat_l is not None:
            mat_l_cpu = mat_l.cpu()
            self._mat_l = mat_l_cpu
            del mat_l
        if mat_r is not None:
            mat_r_cpu = mat_r.cpu()
            self._mat_r = mat_r_cpu
            del mat_r

        if self._mat_l is not None and self._mat_l.shape[1] > self.eq_rank:
            print("Truncating mat_l to eq_rank for storage efficiency")
            self._mat_l = self._mat_l[:, :self.eq_rank]

        if self._mat_r is not None and self._mat_r.shape[0] > self.eq_rank:
            print("Truncating mat_r to eq_rank for storage efficiency")
            self._mat_r = self._mat_r[:self.eq_rank, :]

        if self.singular_values is not None and self.singular_values.is_cuda:
            self.singular_values = self.singular_values.cpu()


@dataclass
class UnefficientFactorizedMatrix(FactorizedMatrix):
    """
    Like :class:`FactorizedMatrix` but **skips** the ``eq_rank`` truncation
    in ``__post_init__``.  Used by the ``*_no_truncate`` factorization
    variants that need to keep all singular components for later re-ranking.
    """
    def __post_init__(self, mat_l, mat_r):
        if mat_l is not None:
            mat_l_cpu = mat_l.cpu()
            self._mat_l = mat_l_cpu
            del mat_l
        if mat_r is not None:
            mat_r_cpu = mat_r.cpu()
            self._mat_r = mat_r_cpu
            del mat_r

        if self.singular_values is not None and self.singular_values.is_cuda:
            self.singular_values = self.singular_values.cpu()
