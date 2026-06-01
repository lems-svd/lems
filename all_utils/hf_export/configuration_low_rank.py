"""Low-rank compressed model configuration for HuggingFace Hub.

This file is **self-contained** (depends only on ``transformers``) and is
copied verbatim into every published model repository so that
``trust_remote_code=True`` works without installing extra packages.
"""

from transformers import PretrainedConfig


class LowRankConfig(PretrainedConfig):
    """Configuration for a low-rank compressed causal language model.

    Extends :class:`PretrainedConfig` with compression metadata — most
    importantly a per-layer ``rank_dict`` that specifies which linear layers
    were replaced with low-rank approximations and the rank used for each.

    Attributes not found on this config fall through to ``base_config``
    (a plain dict snapshot of the original model's config), so that
    generation-related attributes like ``eos_token_id`` or ``vocab_size``
    remain accessible without manual forwarding.
    """

    model_type = "low_rank_compressed"

    def __init__(
        self,
        base_config: dict = None,
        base_model_type: str = None,
        base_model_id: str = None,
        rank_dict: dict = None,
        compression_method: str = None,
        target_ratio: float = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_config = base_config or {}
        self.base_model_type = base_model_type
        self.base_model_id = base_model_id
        self.rank_dict = rank_dict or {}
        self.compression_method = compression_method
        self.target_ratio = target_ratio

    # ------------------------------------------------------------------
    #  Transparent fallback to the wrapped base config
    # ------------------------------------------------------------------

    def __getattr__(self, name):
        """Fall back to ``base_config`` dict for generation-related attrs."""
        # Use __dict__ directly to avoid infinite recursion during init.
        base = self.__dict__.get("base_config")
        if base is not None and isinstance(base, dict) and name in base:
            return base[name]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )
