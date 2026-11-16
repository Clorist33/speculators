"""Shared loss selection, weighting, and reduction utilities."""

import json
import math
from collections.abc import Callable
from functools import partial

import torch

from speculators.losses import eager

_LOSS_REDUCTION_EPS = 1e-5

LossConfig = dict[
    str, tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], float]
]


def dflash_loss_decay(
    pos_idx: torch.Tensor, gamma: float, sample_from_anchor: bool = False, **_kwargs
):
    """Compute DFlash-style exponential decay weights per position.

    When ``sample_from_anchor`` is False (DFlash), position 0 is the anchor and
    gets weight 0; position 1 gets weight 1, and subsequent positions decay as
    ``exp(-(pos - 1) / gamma)``.

    When ``sample_from_anchor`` is True (DSpark), position 0 is the first real
    prediction and gets weight 1; subsequent positions decay as
    ``exp(-pos / gamma)``.

    Args:
        pos_idx: Position indices within each speculative block.
        gamma: Decay rate (higher = slower decay).
        sample_from_anchor: Whether position 0 is a prediction (DSpark) or the
            anchor input (DFlash).

    Returns:
        Decay multiplier tensor with same shape as pos_idx.
    """
    if sample_from_anchor:
        # DSpark: slot 0 predicts the token after the anchor, so decay from 0.
        return torch.exp(-pos_idx / gamma)
    # DFlash: slot 0 is the anchor (no prediction); decay from slot 1.
    # pos_idx = 0 1 2 3 0 1 2 3, block_size = 4
    decay_mult = torch.exp(-((pos_idx - 1).clamp(min=0)) / gamma)
    # decay_mult = e^-(0 0 1 2 0 0 1 2) / gamma
    decay_mult = decay_mult * (pos_idx != 0).to(decay_mult.dtype)
    # w = 0 1 e^-1/gamma e^-2/gamma 0 1 e^-1/gamma e^-2/gamma
    return decay_mult  # noqa: RET504


def exp_loss_decay(pos_idx: torch.Tensor, gamma: float, **_kwargs):
    """Compute simple exponential decay weights as gamma^pos_idx.

    Args:
        pos_idx: Position indices within each speculative block.
        gamma: Base of the exponent (typically in (0, 1]).

    Returns:
        Decay multiplier tensor with same shape as pos_idx.
    """
    return gamma**pos_idx


def dpace_loss_decay(
    pos_idx: torch.Tensor,  # noqa: ARG001
    loss_mask: torch.Tensor,
    block_size: int,
    dpace_alpha: float,
    elementwise_loss: torch.Tensor,
    **_kwargs,
):
    """
    Per-position block-drafting loss weight based on D-PACE

    Args:
        elementwise_loss: requires to be cross-entropy loss, negative log-likelihood
            of per-position confidence
        dpace_alpha: confidence smoothing constant

    Returns:
        Decay multiplier tensor with same shape as pos_idx.
    """
    with torch.no_grad():
        # convert CE to per-position confidence
        q = torch.exp(-elementwise_loss).float()

        # reshape loss to [num_anchors, block_size]
        # for intra-block cumulative multiplication
        if q.shape[1] % block_size != 0:
            raise ValueError(
                f"q.shape[1] ({q.shape[1]}) must be divisible by "
                f"block_size ({block_size})"
            )
        num_anchors = q.shape[1] // block_size
        q = q.reshape(num_anchors, block_size)
        mask = loss_mask.reshape(num_anchors, block_size).to(q.dtype)

        # smoothed confidence for numerical stability
        smooth = (1.0 - dpace_alpha) * q + dpace_alpha
        smooth = torch.where(mask > 0, smooth, torch.ones_like(smooth))

        # prefix cumulative production
        prefix = torch.cumprod(smooth, dim=-1)

        # suffix summation: flip -> cumsum -> flip
        weight = torch.flip(
            torch.cumsum(torch.flip(prefix * mask, dims=[-1]), dim=-1), dims=[-1]
        )
        weight = weight * mask

    # reshape weight
    return weight.reshape(1, -1)


@torch.no_grad()
def vat_loss_decay(
    pred_ids: torch.Tensor,
    target_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    block_size: int,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Re-anchor DSpark's decay at each block's first greedy mismatch.

    Only the contiguous valid prefix from slot 0 represents one verification
    attempt. Valid slots after a mask gap retain the original fixed decay.
    Returns flattened weights, first mismatch (``block_size`` if absent), and
    the per-block valid-prefix mask. The loss reducer still divides by the
    number of valid slots, rather than by the sum of these weights.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError(f"gamma must be positive and finite, got {gamma}")
    if pred_ids.ndim != 2 or pred_ids.shape != target_ids.shape:
        raise ValueError("pred_ids and target_ids must have the same [1, T] shape")
    if pred_ids.shape != loss_mask.shape or pred_ids.shape[0] != 1:
        raise ValueError("loss_mask must have the same [1, T] shape as pred_ids")
    if pred_ids.shape[1] % block_size:
        raise ValueError("sequence length must be divisible by block_size")

    num_blocks = pred_ids.shape[1] // block_size
    valid = loss_mask.reshape(num_blocks, block_size).bool()
    prefix_valid = valid.to(torch.int32).cumprod(dim=-1).bool()
    mismatch = prefix_valid & (
        pred_ids.reshape(num_blocks, block_size)
        != target_ids.reshape(num_blocks, block_size)
    )
    positions = torch.arange(block_size, device=pred_ids.device).unsqueeze(0)
    first_mismatch = torch.where(mismatch, positions, block_size).amin(dim=-1)
    distance = (positions - first_mismatch.unsqueeze(-1)).clamp_min(0).float()
    vat_weights = torch.exp(-distance / gamma)
    fixed_weights = torch.exp(-positions.float() / gamma)
    weights = torch.where(prefix_valid, vat_weights, fixed_weights)
    weights = weights * valid.to(weights.dtype)
    return weights.reshape_as(loss_mask), first_mismatch, prefix_valid


def _fused_kernel(name: str):
    """Import a fused loss by name."""
    try:
        from speculators.losses import fused as mod  # noqa: PLC0415
    except ImportError as error:
        raise RuntimeError(
            "Fused losses require Triton; use --loss-implementation eager for "
            "compatibility testing."
        ) from error
    return getattr(mod, name)


def kl_div_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return _fused_kernel("fused_kl_div_loss")(logits, targets)


def reverse_kl_div_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return _fused_kernel("fused_reverse_kl_div_loss")(logits, targets)


def js_div_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return _fused_kernel("fused_js_div_loss")(logits, targets)


def ce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return _fused_kernel("fused_ce_loss")(logits, targets)


def tv_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return _fused_kernel("fused_tv_loss")(logits, targets)


def neg_log_acceptance_loss(
    logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    return _fused_kernel("fused_nla_loss")(logits, targets)


def lk_hybrid_loss(
    logits: torch.Tensor, targets: torch.Tensor, eta: float = 3.0
) -> torch.Tensor:
    return _fused_kernel("fused_lk_hybrid_loss")(logits, targets, eta=eta)


def lk_hybrid_slot_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    loss_mask: torch.Tensor,
    block_size: int,
    eta: float = 3.0,
    implementation: str = "fused",
) -> torch.Tensor:
    """Paper LK hybrid: share one detached KL/TV blend per draft slot.

    Acceptance is averaged over the valid occurrences of each slot in the
    current local batch and sequence. This differs from ``lk_hybrid_loss``,
    which computes a separate blend for every token. The caller applies the
    ordinary loss mask, position decay, and reduction afterwards.
    """
    if block_size <= 0 or logits.shape[1] % block_size:
        raise ValueError("sequence length must be divisible by a positive block_size")
    if loss_mask.shape != logits.shape[:2]:
        raise ValueError("loss_mask must match logits' batch and sequence dimensions")
    if not math.isfinite(eta) or eta <= 0:
        raise ValueError("eta must be positive and finite")

    loss_fns = _LOSS_FN_MAPS[implementation]
    kl = loss_fns["kl_div"](logits, targets)
    tv = loss_fns["tv"](logits, targets)
    valid = loss_mask.bool().reshape(-1, block_size)
    with torch.no_grad():
        overlap = (1.0 - tv.detach()).reshape(-1, block_size)
        slot_acceptance = (overlap * valid).sum(dim=0) / valid.sum(dim=0).clamp_min(1)
        slot_lambda = torch.exp(-eta * slot_acceptance)
    token_lambda = slot_lambda.repeat(logits.shape[0] * logits.shape[1] // block_size)
    token_lambda = token_lambda.reshape_as(kl)
    return token_lambda * kl + (1.0 - token_lambda) * tv


_FUSED_LOSS_FN_MAP: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "kl_div": kl_div_loss,
    "rkl": reverse_kl_div_loss,
    "jsd": js_div_loss,
    "ce": ce_loss,
    "tv": tv_loss,
    "nla": neg_log_acceptance_loss,
    "lk_hybrid": lk_hybrid_loss,
    "lk_hybrid_slot": partial(lk_hybrid_slot_loss, implementation="fused"),
}


_EAGER_LOSS_FN_MAP: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "kl_div": eager.kl_div_loss,
    "rkl": eager.reverse_kl_div_loss,
    "jsd": eager.js_div_loss,
    "ce": eager.ce_loss,
    "tv": eager.tv_loss,
    "nla": eager.neg_log_acceptance_loss,
    "lk_hybrid": eager.lk_hybrid_loss,
    "lk_hybrid_slot": partial(lk_hybrid_slot_loss, implementation="eager"),
}


_LOSS_FN_MAPS = {"fused": _FUSED_LOSS_FN_MAP, "eager": _EAGER_LOSS_FN_MAP}


def resolve_loss_config(spec: str, implementation: str = "fused") -> LossConfig:
    """Parse a loss spec into ``{name: (loss_fn, weight)}``.

    Accepts either a plain loss name (``"kl_div"``) or a JSON dict mapping
    loss names to weights (``'{"ce": 0.1, "tv": 0.9}'``). The paper-style
    ``lk_hybrid_slot`` also needs a block size and mask from ``compound_loss``.
    """
    try:
        loss_fn_map = _LOSS_FN_MAPS[implementation]
    except KeyError:
        raise ValueError(
            f"Unknown loss implementation '{implementation}'. Choose from: "
            f"{sorted(_LOSS_FN_MAPS)}"
        ) from None

    if spec in loss_fn_map:
        return {spec: (loss_fn_map[spec], 1.0)}

    try:
        parsed = json.loads(spec)
    except json.JSONDecodeError:
        raise ValueError(
            f"Unknown loss function '{spec}'. Pass a known name "
            f"({sorted(loss_fn_map)}) or a JSON dict, "
            f'e.g. \'{{"ce": 0.1, "tv": 0.9}}\'.'
        ) from None

    if not isinstance(parsed, dict) or not parsed:
        raise ValueError(
            "Loss config must be a non-empty JSON dict mapping loss names to weights, "
            f'e.g. \'{{"ce": 0.1, "tv": 0.9}}\'. Got: {spec}'
        )

    config: LossConfig = {}
    for name, weight in parsed.items():
        if name not in loss_fn_map:
            raise ValueError(
                f"Unknown loss function '{name}' in loss config. "
                f"Choose from: {sorted(loss_fn_map)}"
            )
        if not isinstance(weight, (int, float)):
            raise ValueError(
                f"Loss weight for '{name}' must be a number, "
                f"got {type(weight).__name__}"
            )
        config[name] = (loss_fn_map[name], float(weight))

    return config


def compound_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    pos_idx: torch.Tensor,
    loss_config: LossConfig,
    decay_fn: Callable[..., torch.Tensor] | None = None,
    block_size: int | None = None,
    lk_eta: float = 3.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute a weighted sum of loss terms.

    Each entry in *loss_config* maps a name to ``(loss_fn, weight)``; the
    result is ``sum(weight * loss_function(logits, targets, ..., loss_fn))``
    over all entries.

    Returns the total loss and a dict of per-term (unweighted) scalar losses
    keyed as ``"{name}_loss"``.  When the config contains a single term the
    dict is empty (the overall loss already captures it).
    """
    total = torch.tensor(0.0, device=logits.device, dtype=torch.float32)
    term_losses: dict[str, torch.Tensor] = {}
    multi = len(loss_config) > 1
    for name, (fn, weight) in loss_config.items():
        if name == "lk_hybrid_slot":
            if block_size is None:
                raise ValueError("lk_hybrid_slot requires a draft block_size")
            fn = partial(fn, loss_mask=loss_mask, block_size=block_size, eta=lk_eta)
        term = loss_function(
            logits,
            targets,
            loss_mask,
            pos_idx,
            loss_fn=fn,
            decay_fn=decay_fn,
        )
        if multi:
            term_losses[f"{name}_loss"] = term.detach()
        total = total + weight * term
    return total, term_losses


def loss_function(
    logits: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
    loss_mask: torch.Tensor,  # shape: [1, seq_len]
    pos_idx: torch.Tensor,  # shape: [1, seq_len]
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = kl_div_loss,
    decay_fn: Callable[..., torch.Tensor] | None = None,
):
    """Compute masked, optionally position-decayed training loss.

    Args:
        logits: Draft model logits.
        targets: Target model logits.
        loss_mask: Boolean mask selecting positions to include in the loss.
        pos_idx: Position indices within each speculative block.
        loss_fn: Per-position loss function (default: kl_div_loss).
        decay_fn: Optional position-dependent decay weighting function.

    Returns:
        Scalar mean loss across the batch.
    """
    elementwise_loss = loss_fn(logits, targets)  # shape: [1, seq_len]

    loss_mask = loss_mask.to(elementwise_loss.dtype)
    elementwise_loss = elementwise_loss * loss_mask

    if decay_fn is not None:
        decay_mult = decay_fn(
            pos_idx.to(elementwise_loss.dtype), elementwise_loss=elementwise_loss
        )
        elementwise_loss = elementwise_loss * decay_mult

    denominator = loss_mask.sum(dim=1) + _LOSS_REDUCTION_EPS

    batch_loss = torch.sum(elementwise_loss, dim=1) / denominator  # shape: [1]
    return batch_loss.mean()  # shape: []
