"""NV-CoT continuous bounding-box action policy.

Coordinates are represented in the original image frame and normalized to [0, 1].
The module deliberately keeps the policy separate from the vocabulary head: text
continues to use the categorical LM policy while designated action positions use
this distribution.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class ContinuousPolicyConfig:
    distribution: str = "laplace"
    box_parameterization: str = "xyxy"
    shared_scale: bool = True
    fixed_scale: bool = False
    min_scale: float = 0.01
    max_scale: float = 0.30
    min_box_size: float = 0.02
    initial_scale: float = 0.15
    max_residual: float = 0.15
    # Optional pretrained tokenizer token appended only to the frozen actor
    # prefix used for coordinate decoding.  The recommended Qwen token is
    # ``<|box_start|>``: unlike a newly resized vocabulary row, it already has
    # a meaningful embedding and remains compatible with head-only training.
    action_query_token: Optional[str] = None
    action_hidden_pool_tokens: int = 1
    head_architecture: str = "linear"
    mlp_hidden_size: int = 256

    def __post_init__(self):
        if self.distribution not in {"gaussian", "laplace"}:
            raise ValueError("distribution must be 'gaussian' or 'laplace'")
        if self.box_parameterization not in {"xyxy", "ordered_xyxy", "text_residual"}:
            raise ValueError(
                "box_parameterization must be 'xyxy', 'ordered_xyxy', or 'text_residual'"
            )
        if not 0 < self.min_scale <= self.max_scale:
            raise ValueError("Require 0 < min_scale <= max_scale")
        if not self.min_scale <= self.initial_scale <= self.max_scale:
            raise ValueError("initial_scale must be within [min_scale, max_scale]")
        if not 0 < self.max_residual < 1:
            raise ValueError("max_residual must be within (0, 1)")
        if self.action_query_token is not None and not isinstance(self.action_query_token, str):
            raise ValueError("action_query_token must be a string or null")
        if int(self.action_hidden_pool_tokens) != self.action_hidden_pool_tokens or self.action_hidden_pool_tokens < 1:
            raise ValueError("action_hidden_pool_tokens must be a positive integer")
        if self.head_architecture not in {"linear", "residual_mlp", "normalized_mlp"}:
            raise ValueError(
                "head_architecture must be 'linear', 'residual_mlp', or 'normalized_mlp'"
            )
        if int(self.mlp_hidden_size) != self.mlp_hidden_size or self.mlp_hidden_size < 1:
            raise ValueError("mlp_hidden_size must be a positive integer")


def sanitize_normalized_boxes(boxes: Tensor, min_size: float = 0.02) -> Tensor:
    """Convert arbitrary 4-vectors into valid xyxy boxes without hard sorting.

    A center/size parameterization provides useful gradients even for initially
    reversed coordinates, then clamps the crop to the global image frame.
    """
    boxes = boxes.float()
    p1, p2 = boxes[..., :2], boxes[..., 2:]
    center = (p1 + p2) * 0.5
    size = (p2 - p1).abs().clamp_min(min_size)
    lo = center - size * 0.5
    hi = center + size * 0.5
    shift_lo = (-lo).clamp_min(0)
    shift_hi = (hi - 1).clamp_min(0)
    lo = (lo + shift_lo - shift_hi).clamp(0, 1 - min_size)
    hi = (hi + shift_lo - shift_hi).clamp(min_size, 1)
    hi = torch.maximum(hi, lo + min_size).clamp(max=1)
    return torch.cat((lo, hi), dim=-1).to(boxes.dtype)


def apply_text_box_residual(
    text_boxes: Tensor,
    residuals: Tensor,
    min_size: float = 0.02,
) -> Tensor:
    """Apply a bounded learned correction to DeepEyes' own textual boxes."""
    if text_boxes.shape != residuals.shape or text_boxes.shape[-1] != 4:
        raise ValueError("text boxes and residuals must have the same [..., 4] shape")
    return sanitize_normalized_boxes(text_boxes.float() + residuals.float(), min_size)


class ContinuousCoordinateHead(nn.Module):
    """Continuous box policy with a linear or residual-MLP location decoder."""

    def __init__(self, hidden_size: int, config: Optional[ContinuousPolicyConfig] = None):
        super().__init__()
        self.config = config or ContinuousPolicyConfig()
        scale_dim = 1 if self.config.shared_scale else 4
        self.location = nn.Linear(hidden_size, 4)
        self.location_adapter = None
        if self.config.head_architecture in {"residual_mlp", "normalized_mlp"}:
            self.location_adapter = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, self.config.mlp_hidden_size),
                nn.GELU(),
                nn.Linear(self.config.mlp_hidden_size, 4),
            )
        self.log_scale = nn.Linear(hidden_size, scale_dim)
        # Start every example from the same useful, broad central crop.  VLM
        # hidden states have a much larger norm than ordinary MLP inputs, so
        # even a seemingly tiny random projection can immediately saturate the
        # sigmoid and collapse ordered boxes to their minimum size.  A zero
        # projection keeps the initial action well-conditioned; supervised
        # gradients make it context-dependent from the first optimizer step.
        nn.init.zeros_(self.location.weight)
        if self.config.box_parameterization == "text_residual":
            initial_location = torch.zeros(4)
        elif self.config.box_parameterization == "ordered_xyxy":
            initial_location = torch.tensor([0.2, 0.2, 0.75, 0.75])
        else:
            initial_location = torch.tensor([0.2, 0.2, 0.8, 0.8])
        with torch.no_grad():
            if self.config.box_parameterization == "text_residual":
                self.location.bias.copy_(initial_location)
            else:
                self.location.bias.copy_(torch.logit(initial_location))
        if self.location_adapter is not None:
            # Preserve the same well-conditioned central initial box as the
            # linear baseline. The zero final projection lets the adapter
            # enter smoothly while LayerNorm protects it from large VLM norms.
            nn.init.zeros_(self.location_adapter[-1].weight)
            nn.init.zeros_(self.location_adapter[-1].bias)
        if self.config.head_architecture == "normalized_mlp":
            # Keep the legacy Linear tensors in the state dict for checkpoint
            # tooling, but make the recommended path exactly
            # LayerNorm -> Linear -> GELU -> Linear with no unnormalized skip.
            self.location.requires_grad_(False)
        nn.init.zeros_(self.log_scale.weight)
        initial = self.config.initial_scale
        nn.init.constant_(self.log_scale.bias, torch.log(torch.expm1(torch.tensor(initial))))
        if self.config.fixed_scale:
            # Fixed-scale policies never consult this projection, so these
            # parameters must not enter the optimizer or gradient synchronizer.
            self.log_scale.requires_grad_(False)

    def parameters_from_hidden(self, hidden: Tensor) -> Tuple[Tensor, Tensor]:
        # Head-only training intentionally keeps this tiny module in FP32 while
        # the frozen FSDP VLM emits BF16 hidden states.  Nested FSDP used to
        # autocast this boundary implicitly; the replicated plain head must do
        # it explicitly.  Casting the activation (rather than the parameters)
        # preserves FP32 optimizer state and numerically stable distribution
        # parameters.
        raw_location, scale = self.raw_parameters_from_hidden(hidden)
        return self.action_from_raw_location(raw_location), scale

    def raw_parameters_from_hidden(self, hidden: Tensor) -> Tuple[Tensor, Tensor]:
        """Return the Gaussian/Laplace parameters in unconstrained latent space."""
        parameter_dtype = self.location.weight.dtype
        if hidden.dtype != parameter_dtype:
            hidden = hidden.to(dtype=parameter_dtype)
        if self.config.head_architecture == "normalized_mlp":
            raw_location = self.location_adapter(hidden)
        else:
            raw_location = self.location(hidden)
        if self.config.head_architecture == "residual_mlp":
            raw_location = raw_location + self.location_adapter(hidden)
        if self.config.fixed_scale:
            scale_dim = 1 if self.config.shared_scale else 4
            scale = torch.full(
                (*raw_location.shape[:-1], scale_dim),
                self.config.initial_scale,
                dtype=raw_location.dtype,
                device=raw_location.device,
            )
        else:
            scale = F.softplus(self.log_scale(hidden))
            scale = scale.clamp(self.config.min_scale, self.config.max_scale)
        return raw_location, scale

    def action_from_raw_location(self, raw_location: Tensor) -> Tensor:
        """Map unconstrained head outputs to normalized xyxy coordinates."""
        if self.config.box_parameterization == "text_residual":
            return self.config.max_residual * torch.tanh(raw_location)
        unit_location = torch.sigmoid(raw_location)
        if self.config.box_parameterization == "ordered_xyxy":
            lo = unit_location[..., :2]
            hi = lo + (1.0 - lo) * unit_location[..., 2:]
            return torch.cat((lo, hi), dim=-1)
        return unit_location

    def unit_location_from_action(self, action: Tensor) -> Tensor:
        """Inverse the ordered action map before the final logit transform.

        This representation is also a stable supervised target for
        ``binary_cross_entropy_with_logits``.  Unlike action-space MSE, that
        objective keeps a useful gradient when an early update sends a sigmoid
        logit into saturation.
        """
        action = action.to(dtype=self.location.weight.dtype, device=self.location.weight.device)
        eps = torch.finfo(action.dtype).eps
        if self.config.box_parameterization == "text_residual":
            return (action / self.config.max_residual).clamp(-1.0 + eps, 1.0 - eps)
        if self.config.box_parameterization == "ordered_xyxy":
            lo = action[..., :2].clamp(eps, 1.0 - eps)
            fraction = ((action[..., 2:] - lo) / (1.0 - lo)).clamp(eps, 1.0 - eps)
            return torch.cat((lo, fraction), dim=-1)
        return action.clamp(eps, 1.0 - eps)

    def location_bias_from_action(self, action: Tensor) -> Tensor:
        """Inverse the location transform for a context-independent init."""
        action = action.to(dtype=self.location.bias.dtype, device=self.location.bias.device)
        if self.config.box_parameterization == "text_residual":
            return torch.atanh(self.unit_location_from_action(action))
        return torch.logit(self.unit_location_from_action(action))

    def raw_location_from_action(self, action: Tensor) -> Tensor:
        """Inverse the action transform for supervised likelihood targets."""
        action = action.to(dtype=self.location.weight.dtype, device=self.location.weight.device)
        if self.config.box_parameterization == "text_residual":
            return torch.atanh(self.unit_location_from_action(action))
        return torch.logit(self.unit_location_from_action(action))

    def forward(self, hidden: Tensor, return_raw: bool = False):
        """Expose distribution parameters through the standard module call.

        PPO uses the convenience methods below, while supervised warm-up wraps
        this tiny head in DDP and therefore must enter through ``forward`` so
        gradient synchronization hooks are installed correctly.
        """
        raw_location, scale = self.raw_parameters_from_hidden(hidden)
        location = self.action_from_raw_location(raw_location)
        if return_raw:
            return raw_location, location, scale
        return location, scale

    def distribution(self, hidden: Tensor):
        location, scale = self.raw_parameters_from_hidden(hidden)
        cls = torch.distributions.Laplace if self.config.distribution == "laplace" else torch.distributions.Normal
        return torch.distributions.Independent(cls(location, scale), 1)

    def sample(self, hidden: Tensor, deterministic: bool = False) -> Dict[str, Tensor]:
        dist = self.distribution(hidden)
        raw = dist.base_dist.loc if deterministic else dist.rsample()
        action = self.action_from_raw_location(raw)
        if self.config.box_parameterization != "text_residual":
            action = sanitize_normalized_boxes(action, self.config.min_box_size)
        # The environment consumes the sanitized crop, while PPO scores the raw
        # stochastic action. For a fixed transform, its Jacobian cancels in the
        # old/new likelihood ratio; keeping the raw action also avoids assigning
        # an incorrect base-density to a clipped or reordered box.
        return {"action": action, "raw_action": raw, "log_prob": dist.log_prob(raw)}

    def log_prob(self, hidden: Tensor, raw_actions: Tensor) -> Tensor:
        return self.distribution(hidden).log_prob(raw_actions)

    def entropy(self, hidden: Tensor) -> Tensor:
        return self.distribution(hidden).entropy()

    def mean_kl(self, hidden: Tensor, reference_hidden: Tensor) -> Tensor:
        """Paper's stable mean-only reference regularizer."""
        mu, _ = self.parameters_from_hidden(hidden)
        ref_mu, _ = self.parameters_from_hidden(reference_hidden)
        return (mu - ref_mu.detach()).square().sum(-1)


def attach_continuous_coordinate_head(model: nn.Module, config) -> ContinuousCoordinateHead:
    """Attach an FSDP/checkpoint-visible head to a HuggingFace causal VLM."""
    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is None and hasattr(model.config, "text_config"):
        hidden_size = model.config.text_config.hidden_size
    if hidden_size is None:
        raise ValueError("Cannot determine language hidden size for NV-CoT")
    cfg = ContinuousPolicyConfig(
        distribution=config.get("distribution", "laplace"),
        box_parameterization=config.get("box_parameterization", "xyxy"),
        shared_scale=config.get("shared_scale", True),
        fixed_scale=config.get("fixed_scale", False),
        min_scale=config.get("min_scale", 0.01),
        max_scale=config.get("max_scale", 0.30),
        min_box_size=config.get("min_box_size", 0.02),
        initial_scale=config.get("initial_scale", 0.15),
        max_residual=config.get("max_residual", 0.15),
        action_query_token=config.get("action_query_token", None) or None,
        action_hidden_pool_tokens=config.get("action_hidden_pool_tokens", 1),
        head_architecture=config.get("head_architecture", "linear"),
        mlp_hidden_size=config.get("mlp_hidden_size", 256),
    )
    head = ContinuousCoordinateHead(hidden_size, cfg)
    model.nv_cot_head = head
    model.config.nv_cot = dict(vars(cfg))
    return head


def get_continuous_coordinate_head(model: nn.Module) -> ContinuousCoordinateHead:
    """Find the NV-CoT head through common FSDP/module wrappers."""
    current = model
    visited = set()
    while id(current) not in visited:
        visited.add(id(current))
        head = getattr(current, "nv_cot_head", None)
        if head is not None:
            return head
        next_module = getattr(current, "_fsdp_wrapped_module", None)
        if next_module is None:
            next_module = getattr(current, "module", None)
        if next_module is None:
            break
        current = next_module
    raise RuntimeError("NV-CoT is enabled but no continuous coordinate head is attached")


def _unwrap_coordinate_head(head: nn.Module) -> ContinuousCoordinateHead:
    current = head
    visited = set()
    while id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, ContinuousCoordinateHead):
            return current
        wrapped = getattr(current, "_fsdp_wrapped_module", None)
        if wrapped is None:
            wrapped = getattr(current, "module", None)
        if wrapped is None:
            break
        current = wrapped
    raise RuntimeError("Could not unwrap the continuous coordinate head")


def distribution_from_parameters(head: nn.Module, location: Tensor, scale: Tensor):
    config = _unwrap_coordinate_head(head).config
    cls = torch.distributions.Laplace if config.distribution == "laplace" else torch.distributions.Normal
    return torch.distributions.Independent(cls(location, scale), 1)


def continuous_distribution(head: nn.Module, hidden: Tensor):
    """Build the policy distribution in unconstrained latent coordinate space."""
    unwrapped = _unwrap_coordinate_head(head)
    location, scale = unwrapped.raw_parameters_from_hidden(hidden)
    return distribution_from_parameters(head, location, scale)


def sample_continuous_coordinate_head(head: nn.Module, hidden: Tensor, deterministic: bool = False):
    dist = continuous_distribution(head, hidden)
    raw = dist.base_dist.loc if deterministic else dist.rsample()
    unwrapped = _unwrap_coordinate_head(head)
    config = unwrapped.config
    action = unwrapped.action_from_raw_location(raw)
    if config.box_parameterization != "text_residual":
        action = sanitize_normalized_boxes(action, config.min_box_size)
    return {"action": action, "raw_action": raw, "log_prob": dist.log_prob(raw)}
