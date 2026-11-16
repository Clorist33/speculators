"""Unit tests for the DSpark loss and metrics."""

from functools import partial

import torch

from speculators.losses import (
    dflash_loss_decay,
    lk_hybrid_slot_loss,
    resolve_loss_config,
    vat_loss_decay,
)
from speculators.losses.eager import ce_loss, kl_div_loss, lk_hybrid_loss, tv_loss
from speculators.models.dspark.metrics import compute_metrics as _compute_metrics

compute_metrics = partial(_compute_metrics, tv_loss_fn=tv_loss)
_DEFAULT_LOSS = resolve_loss_config('{"ce": 0.1, "tv": 0.9}', "eager")


def _ids_to_logits(ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    logits = torch.zeros(*ids.shape, vocab_size)
    logits.scatter_(-1, ids.unsqueeze(-1), 100.0)
    return logits


def test_lk_hybrid_slot_shares_lambda_and_excludes_masked_tokens():
    logits = torch.tensor(
        [[[4.0, 0, 0], [0, 4, 0], [4, 0, 0], [0, 0, 4]]],
        requires_grad=True,
    )
    targets = torch.tensor([[[0.0, 4, 0], [0, 4, 0], [4, 0, 0], [4, 0, 0]]])
    mask = torch.tensor([[1, 1, 1, 0]])
    actual = lk_hybrid_slot_loss(
        logits, targets, loss_mask=mask, block_size=2, implementation="eager"
    )
    tv = tv_loss(logits, targets)
    kl = kl_div_loss(logits, targets)
    slot_alpha = torch.stack(((2 - tv[0, 0] - tv[0, 2]) / 2, 1 - tv[0, 1]))
    slot_lambda = torch.exp(-3.0 * slot_alpha.detach()).repeat(2).reshape(1, 4)
    expected = slot_lambda * kl + (1 - slot_lambda) * tv
    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual[:, 0], lk_hybrid_loss(logits, targets)[:, 0])

    grad_actual = torch.autograd.grad((actual * mask).sum(), logits, retain_graph=True)[0]
    grad_expected = torch.autograd.grad((expected * mask).sum(), logits)[0]
    torch.testing.assert_close(grad_actual, grad_expected)


def test_lk_hybrid_slot_keeps_ce_and_confidence_loss():
    torch.manual_seed(3)
    logits = torch.randn(1, 4, 6, requires_grad=True)
    targets = torch.randn(1, 4, 6)
    confidence = torch.randn(1, 4, requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 0]])
    config = resolve_loss_config('{"ce":0.1,"lk_hybrid_slot":0.9}', "eager")
    loss, metrics = compute_metrics(
        logits,
        targets,
        confidence,
        mask,
        block_size=2,
        loss_config=config,
        lk_eta=3.0,
        gamma=8.0,
    )
    decay = dflash_loss_decay(torch.arange(4).remainder(2).float()[None], 8, True)
    lk = lk_hybrid_slot_loss(
        logits, targets, loss_mask=mask, block_size=2, implementation="eager"
    )
    compound = ((0.1 * ce_loss(logits, targets) + 0.9 * lk) * mask * decay).sum()
    compound = compound / (mask.sum() + 1e-5)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        confidence, 1 - tv_loss(logits, targets).detach(), reduction="none"
    )
    conf_loss = (bce * mask * decay).sum() / (mask.sum() + 1e-8)
    torch.testing.assert_close(loss, compound + conf_loss)
    torch.testing.assert_close(
        metrics["ce_loss_sum"],
        (ce_loss(logits, targets) * mask * decay).sum() / (mask.sum() + 1e-5),
    )
    torch.testing.assert_close(
        metrics["lk_hybrid_slot_loss_sum"],
        (lk * mask * decay).sum() / (mask.sum() + 1e-5),
    )
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.isfinite(confidence.grad).all()


def test_lk_hybrid_slot_all_masked_has_zero_loss_and_gradients():
    logits = torch.randn(1, 4, 6, requires_grad=True)
    targets = torch.randn(1, 4, 6)
    loss, _ = compute_metrics(
        logits,
        targets,
        None,
        torch.zeros(1, 4),
        block_size=2,
        loss_config=resolve_loss_config("lk_hybrid_slot", "eager"),
    )
    loss.backward()
    assert loss == 0
    assert torch.count_nonzero(logits.grad) == 0


def test_vat_loss_decay_first_mismatch_and_mask_boundaries():
    pred_ids = torch.tensor([[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]])
    target_ids = pred_ids.clone()
    target_ids[0, 0] = 9  # First slot: exactly the original fixed decay.
    target_ids[0, 6] = 9  # Third slot: first three weights stay at one.
    target_ids[0, 15] = 9  # After a mask gap: not part of the same prefix.
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]])

    weights, first, prefix = vat_loss_decay(pred_ids, target_ids, mask, 4, 4.0)
    fixed = torch.exp(-torch.arange(4).float() / 4)
    expected = fixed.repeat(4, 1)
    expected[1, :3] = 1
    expected[1, 3] = fixed[1]
    expected[2] = 1
    expected[3, :2] = 1
    expected[3, 2] = 0
    torch.testing.assert_close(weights.reshape(4, 4), expected)
    torch.testing.assert_close(first, torch.tensor([0, 2, 4, 4]))
    torch.testing.assert_close(prefix[3], torch.tensor([True, True, False, False]))
    assert not weights.requires_grad


def test_vat_loss_decay_invalid_first_slot_and_empty_block():
    ids = torch.tensor([[1, 2, 3, 4, 1, 2, 3, 4]])
    mask = torch.tensor([[0, 1, 1, 1, 0, 0, 0, 0]])
    weights, first, prefix = vat_loss_decay(ids, ids, mask, 4, 4.0)
    fixed = torch.exp(-torch.arange(4).float() / 4)
    expected = torch.cat((fixed * mask[0, :4], torch.zeros(4))).unsqueeze(0)
    torch.testing.assert_close(weights, expected)
    torch.testing.assert_close(first, torch.tensor([4, 4]))
    assert not prefix.any()


def test_vat_uses_valid_slot_denominator_and_keeps_confidence_fixed():
    torch.manual_seed(7)
    logits = torch.randn(1, 4, 6, requires_grad=True)
    targets = logits.detach().clone()
    other_id = (int(logits[0, 2].argmax()) + 1) % 6
    targets[0, 2] = -10
    targets[0, 2, other_id] = 10
    confidence_logits = torch.randn(1, 4, requires_grad=True)
    mask = torch.ones(1, 4)

    loss, metrics = compute_metrics(
        logits,
        targets,
        confidence_logits,
        mask,
        block_size=4,
        loss_config=_DEFAULT_LOSS,
        per_position_loss_weight="vat",
    )
    weights = torch.tensor([[1, 1, 1, float(torch.exp(torch.tensor(-0.25)))]])
    per_pos_ce = ce_loss(logits, targets)
    per_pos_tv = tv_loss(logits, targets)
    expected_compound = (
        (0.1 * per_pos_ce + 0.9 * per_pos_tv) * weights
    ).sum() / (4 + 1e-5)
    fixed = dflash_loss_decay(torch.arange(4).float().unsqueeze(0), 4, True)
    per_pos_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        confidence_logits, 1 - per_pos_tv.detach(), reduction="none"
    )
    expected_conf = (per_pos_bce * fixed).sum() / (4 + 1e-8)
    torch.testing.assert_close(loss, expected_compound + expected_conf)
    torch.testing.assert_close(metrics["confidence_loss_sum"], expected_conf)
    assert metrics["vat_first_mismatch_2_sum"] == 1
    assert metrics["vat_weight_ratio_sum"] > 1
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.isfinite(confidence_logits.grad).all()


def test_vat_all_masked_has_zero_loss_and_gradients():
    logits = torch.randn(1, 4, 6, requires_grad=True)
    targets = torch.randn(1, 4, 6)
    confidence = torch.randn(1, 4, requires_grad=True)
    loss, metrics = compute_metrics(
        logits,
        targets,
        confidence,
        torch.zeros(1, 4),
        block_size=4,
        loss_config=_DEFAULT_LOSS,
        per_position_loss_weight="vat",
    )
    loss.backward()
    assert loss == 0
    assert torch.count_nonzero(logits.grad) == 0
    assert torch.count_nonzero(confidence.grad) == 0
    assert metrics["vat_weight_sum_sum"] == 0


class TestComputeMetrics:
    def test_empty_loss_mask_keeps_backward_graph_with_zero_gradients(self):
        logits = torch.randn(1, 8, 16, requires_grad=True)
        targets = torch.randn(1, 8, 16)
        confidence_logits = torch.randn(1, 8, requires_grad=True)
        loss_mask = torch.zeros(1, 8)

        loss, _ = compute_metrics(
            logits,
            targets,
            confidence_logits,
            loss_mask,
            block_size=2,
            loss_config=_DEFAULT_LOSS,
        )
        loss.backward()

        assert loss.item() == 0
        assert logits.grad is not None
        assert confidence_logits.grad is not None
        assert torch.count_nonzero(logits.grad) == 0
        assert torch.count_nonzero(confidence_logits.grad) == 0

    def test_perfect_draft_low_loss_high_accept(self):
        # block_size=2; with sample_from_anchor=False, position 0 is the anchor
        # (masked) and position 1 supervised.
        ids = torch.tensor([[0, 1, 0, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = logits.clone()
        loss_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.float32)
        loss, metrics = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            2,
            gamma=4.0,
            loss_config=_DEFAULT_LOSS,
            sample_from_anchor=False,
        )
        assert torch.isfinite(loss)
        # Matching distributions -> CE/TV ~ 0 and acceptance ~ 1.
        assert float(loss) < 1e-2
        accept = metrics["accept_rate_sum"] / metrics["accept_rate_total"]
        assert float(accept) > 0.99
        # One draft slot per block accepted w.p. ~1, plus the bonus token -> ~2.
        accept_len = metrics["accept_len_sum"] / metrics["accept_len_total"]
        assert abs(float(accept_len) - 2.0) < 1e-2
        eal = metrics["eal_sum"] / metrics["eal_total"]
        assert abs(float(eal) - 2.0) < 1e-2

    def test_perfect_draft_anchor_sampled_includes_slot0(self):
        # sample_from_anchor=True (default): slot 0 is the first real prediction,
        # so every position is supervised and accept_len counts all draft slots.
        ids = torch.tensor([[0, 1, 0, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = logits.clone()
        loss_mask = torch.ones(1, 4, dtype=torch.float32)
        loss, metrics = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            2,
            gamma=4.0,
            loss_config=_DEFAULT_LOSS,
        )
        assert torch.isfinite(loss)
        assert float(loss) < 1e-2
        accept = metrics["accept_rate_sum"] / metrics["accept_rate_total"]
        assert float(accept) > 0.99
        # Two draft slots per block accepted w.p. ~1, plus the bonus token -> ~3.
        accept_len = metrics["accept_len_sum"] / metrics["accept_len_total"]
        assert abs(float(accept_len) - 3.0) < 1e-2
        eal = metrics["eal_sum"] / metrics["eal_total"]
        assert abs(float(eal) - 3.0) < 1e-2

    def test_accept_rate_equals_softmax_overlap(self):
        """accept_rate (now 1 - tv) matches the explicit softmax-overlap formula."""
        torch.manual_seed(0)
        logits = torch.randn(1, 4, 32) * 3
        targets = torch.randn(1, 4, 32) * 3
        loss_mask = torch.ones(1, 4, dtype=torch.float32)
        _, metrics = compute_metrics(
            logits, targets, None, loss_mask, 2, loss_config=_DEFAULT_LOSS
        )
        draft_p = torch.softmax(logits.float(), dim=-1)
        target_p = torch.softmax(targets.float(), dim=-1)
        overlap = torch.minimum(draft_p, target_p).sum(dim=-1)
        assert torch.isclose(metrics["accept_rate_sum"], overlap.sum(), atol=1e-5)

    def test_confidence_target_is_overlap(self):
        # When draft == target, accept rate == 1, so a confidence logit that is
        # very positive (sigmoid -> 1) yields ~zero abs error.
        ids = torch.tensor([[0, 1, 0, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = logits.clone()
        loss_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.float32)
        confidence_logits = torch.full((1, 4), 20.0)  # sigmoid ~ 1.0
        _, metrics = compute_metrics(
            logits,
            targets,
            confidence_logits,
            loss_mask,
            block_size=2,
            gamma=4.0,
            loss_config=_DEFAULT_LOSS,
        )
        abs_err = (
            metrics["confidence_abs_error_sum"] / metrics["confidence_abs_error_total"]
        )
        assert float(abs_err) < 1e-2
        assert "confidence_loss_sum" in metrics

    def test_confidence_term_changes_loss(self):
        ids = torch.tensor([[0, 1, 0, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = _ids_to_logits(torch.tensor([[0, 3, 0, 4]]), 8)
        loss_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.float32)
        loss_no_conf, _ = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            block_size=2,
            loss_config=_DEFAULT_LOSS,
        )
        # A badly-calibrated confidence head (predicts accept~1 when accept~0)
        # must add positive BCE on top of the base loss.
        confidence_logits = torch.full((1, 4), 20.0)
        loss_conf, _ = compute_metrics(
            logits,
            targets,
            confidence_logits,
            loss_mask,
            block_size=2,
            loss_config=_DEFAULT_LOSS,
            confidence_head_alpha=1.0,
        )
        assert float(loss_conf) > float(loss_no_conf)

    def test_confidence_cumprod_bias_sign(self):
        # Draft != target so accept rate is ~0; an over-confident head (predicts
        # accept ~1) must show a positive cumulative-product calibration bias.
        # sample_from_anchor=False: position 0 is the anchor (masked).
        ids = torch.tensor([[0, 1, 0, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = _ids_to_logits(torch.tensor([[0, 3, 0, 4]]), 8)
        loss_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.float32)
        confidence_logits = torch.full((1, 4), 20.0)  # sigmoid ~ 1.0
        _, metrics = compute_metrics(
            logits,
            targets,
            confidence_logits,
            loss_mask,
            block_size=2,
            loss_config=_DEFAULT_LOSS,
            sample_from_anchor=False,
        )
        bias = (
            metrics["confidence_cumprod_bias_sum"]
            / metrics["confidence_cumprod_bias_total"]
        )
        assert float(bias) > 0.5

    def test_alpha_weighting(self):
        ids = torch.tensor([[0, 1, 0, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = _ids_to_logits(torch.tensor([[0, 3, 0, 4]]), 8)
        loss_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.float32)
        loss_small, _ = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            block_size=2,
            loss_config=resolve_loss_config('{"tv": 0.1}', "eager"),
        )
        loss_large, _ = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            block_size=2,
            loss_config=resolve_loss_config('{"tv": 1.0}', "eager"),
        )
        assert float(loss_large) > float(loss_small)

    def test_metric_keys_present(self):
        ids = torch.tensor([[0, 1, 0, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = logits.clone()
        loss_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.float32)
        _, metrics = compute_metrics(
            logits,
            targets,
            torch.zeros(1, 4),
            loss_mask,
            block_size=2,
            loss_config=_DEFAULT_LOSS,
        )
        for key in (
            "loss_sum",
            "loss_total",
            "ce_loss_sum",
            "tv_loss_sum",
            "full_acc_sum",
            "full_acc_total",
            "position_1_acc_sum",
            "accept_len_sum",
            "accept_len_total",
            "confidence_cumprod_bias_sum",
        ):
            assert key in metrics
        # all metric values must be tensors (so dist.reduce works in the trainer)
        assert all(torch.is_tensor(v) for v in metrics.values())
