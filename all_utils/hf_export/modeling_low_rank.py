"""Low-rank compressed causal LM for HuggingFace Hub.

This file is **self-contained** (depends only on ``torch`` and
``transformers``) and is copied verbatim into every published model
repository so that ``trust_remote_code=True`` works without installing
extra packages.

Usage::

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        "your-org/compressed-llama-3-8b",
        trust_remote_code=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "your-org/compressed-llama-3-8b",
    )

    inputs = tokenizer("The capital of France is", return_tensors="pt").to(model.device)
    print(tokenizer.decode(model.generate(**inputs, max_new_tokens=32)[0]))
"""

import torch
from torch import nn
from transformers import AutoModelForCausalLM, PreTrainedModel

# Relative import works when HuggingFace downloads both files into a
# temporary package.  The absolute fallback covers standalone execution.
try:
    from .configuration_low_rank import LowRankConfig
except ImportError:
    from configuration_low_rank import LowRankConfig


# -----------------------------------------------------------------------
#  Low-rank linear layer
# -----------------------------------------------------------------------

class LowRankLinear(nn.Module):
    r"""Drop-in replacement for :class:`nn.Linear` that stores a rank-*r*
    factorisation instead of the full weight matrix.

    .. math::
        y = B\,(A\,x) + \text{bias}

    where :math:`A \in \mathbb{R}^{r \times k}` (``mod_a``) and
    :math:`B \in \mathbb{R}^{m \times r}` (``mod_b``).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.mod_a = nn.Linear(in_features, rank, bias=False)
        self.mod_b = nn.Linear(rank, out_features, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        x = self.mod_a(x)
        x = self.mod_b(x)
        if self.bias is not None:
            x = x + self.bias
        return x

    @property
    def weight(self):
        """Materialise the full weight :math:`W \\approx B A` (read-only)."""
        return (self.mod_b.weight @ self.mod_a.weight).detach()

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, bias={self.bias is not None}"
        )


# -----------------------------------------------------------------------
#  Helpers
# -----------------------------------------------------------------------

def _replace_linear_with_low_rank(model, layer_name: str, rank: int):
    """Navigate to *layer_name* and swap the ``nn.Linear`` for a
    :class:`LowRankLinear` of the given *rank*."""
    parts = layer_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    attr = parts[-1]
    orig = getattr(parent, attr)
    if not isinstance(orig, nn.Linear):
        raise ValueError(
            f"Expected nn.Linear at '{layer_name}', got {type(orig).__name__}"
        )
    replacement = LowRankLinear(
        orig.in_features, orig.out_features, rank,
        bias=(orig.bias is not None),
    )
    setattr(parent, attr, replacement)
    return replacement


# -----------------------------------------------------------------------
#  Wrapper model
# -----------------------------------------------------------------------

class LowRankCausalLM(PreTrainedModel):
    """Thin wrapper around *any* HuggingFace causal LM whose linear layers
    have been partially replaced with :class:`LowRankLinear` modules.

    The wrapped base model is stored as ``self.wrapped_model`` so that the
    state-dict keys are prefixed with ``wrapped_model.`` — this makes the
    mapping between saved weights and module paths unambiguous.
    """

    config_class = LowRankConfig
    supports_gradient_checkpointing = True
    _supports_cache_class = True

    def __init__(self, config: LowRankConfig):
        super().__init__(config)

        from transformers import CONFIG_MAPPING

        # Reconstruct the base model's config object.
        base_config_dict = dict(config.base_config)
        for drop_key in ("auto_map",):
            base_config_dict.pop(drop_key, None)

        config_cls = CONFIG_MAPPING[config.base_model_type]
        base_config = config_cls.from_dict(base_config_dict)

        # Resolve torch_dtype from the stored config string.
        torch_dtype_str = base_config_dict.get("torch_dtype", None)
        _DTYPE_MAP = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = _DTYPE_MAP.get(torch_dtype_str, None)
        extra_kwargs = {"torch_dtype": torch_dtype} if torch_dtype else {}

        # Create the base causal-LM architecture (random weights — the
        # ``from_pretrained`` machinery will overwrite them afterwards).
        self.wrapped_model = AutoModelForCausalLM.from_config(
            base_config, **extra_kwargs,
        )

        # Replace every compressed layer with a LowRankLinear module.
        for layer_name, rank in (config.rank_dict or {}).items():
            _replace_linear_with_low_rank(
                self.wrapped_model, layer_name, int(rank),
            )

    # ------------------------------------------------------------------
    #  Forward — delegate to the inner model
    # ------------------------------------------------------------------

    def forward(self, *args, **kwargs):
        return self.wrapped_model(*args, **kwargs)

    # ------------------------------------------------------------------
    #  Generation helpers (required by GenerationMixin)
    # ------------------------------------------------------------------

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.wrapped_model.prepare_inputs_for_generation(*args, **kwargs)

    def get_input_embeddings(self):
        return self.wrapped_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.wrapped_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.wrapped_model.get_output_embeddings()

    def set_output_embeddings(self, value):
        self.wrapped_model.set_output_embeddings(value)

    def can_generate(self):
        return True

    def _reorder_cache(self, *args, **kwargs):
        if hasattr(self.wrapped_model, "_reorder_cache"):
            return self.wrapped_model._reorder_cache(*args, **kwargs)
        return None
