import json
import os
import signal
from collections import deque
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import pytest
import torch
from torch.utils.data import DataLoader

from speculators.model import SpeculatorModel
from speculators.train.checkpointer import SingleGPUCheckpointer
from speculators.train.trainer import Trainer, TrainerConfig


def _make_minimal_trainer(tmp_path: Path, checkpoint_freq: int, save_best: bool):
    trainer = Trainer.__new__(Trainer)
    trainer.config = TrainerConfig(
        lr=1e-3,
        num_epochs=0,
        save_path=str(tmp_path),
        resume_from_checkpoint=False,
        checkpoint_freq=checkpoint_freq,
        save_best=save_best,
    )
    trainer.best_val_loss = float("inf")
    trainer.current_epoch = 0
    trainer.global_step = 0
    trainer.is_distributed = False
    trainer.rank = 0
    trainer.local_rank = 0
    trainer.resume_from_checkpoint = False
    trainer.train_loader = cast("DataLoader[Any]", [])
    trainer.val_loader = cast("DataLoader[Any]", [])
    trainer.checkpointer = SingleGPUCheckpointer(str(tmp_path))

    trainer.model = cast("SpeculatorModel", object())
    trainer.optimizers = cast("list[torch.optim.Optimizer]", [object()])
    trainer.schedulers = []
    return trainer


def test_previous_epoch_ignores_checkpoint_best(tmp_path: Path):
    (tmp_path / "0").mkdir()
    (tmp_path / "2").mkdir()
    (tmp_path / "checkpoint_best").symlink_to("0", target_is_directory=True)

    cp = SingleGPUCheckpointer(str(tmp_path))
    assert cp.previous_epoch == 2


def test_step_checkpoints_are_independent_and_best_is_retained(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    trainer = _make_minimal_trainer(tmp_path, checkpoint_freq=1, save_best=False)
    trainer.config = trainer.config._replace(checkpoint_eval_steps=100)
    trainer.model = cast("SpeculatorModel", Mock())
    losses = iter((0.8, 0.3, 0.5))
    trainer.val_epoch = lambda _epoch: {"loss_epoch": next(losses)}
    saved: list[str] = []

    def fake_save(_model, _optimizer, name: str):
        saved.append(name)
        (tmp_path / name).mkdir()

    monkeypatch.setattr(trainer.checkpointer, "save_checkpoint", fake_save)

    for step, local_step, name in (
        (100, 100, "step_00000100"),
        (200, 200, "step_00000200"),
        (250, 0, "epoch0_end"),
    ):
        trainer.global_step = step
        trainer._save_and_validate(0, local_step, name)
        assert (tmp_path / name / "val_metrics.json").exists()

    assert saved == ["step_00000100", "step_00000200", "epoch0_end"]
    assert (tmp_path / "checkpoint_best").resolve() == (
        tmp_path / "step_00000200"
    ).resolve()
    assert trainer.best_val_loss == 0.3

    cp = SingleGPUCheckpointer(tmp_path)
    assert cp.previous_epoch == 0
    assert cp.previous_checkpoint == "epoch0_end"
    assert cp.load_best_val_loss() == 0.3

    with pytest.raises(FileExistsError, match="step_00000100"):
        trainer.maybe_save_checkpoint(0, 100, "step_00000100")


def test_step_checkpoint_resume_uses_latest_completed_step(tmp_path: Path):
    for step in (100, 200):
        checkpoint = tmp_path / f"step_{step:08d}"
        checkpoint.mkdir()
        (checkpoint / "training_state.json").write_text(json.dumps({
            "epoch": 0, "local_step": step, "global_step": step,
        }))
    # A partially written checkpoint must not displace the last usable one.
    (tmp_path / "step_00000300").mkdir()

    cp = SingleGPUCheckpointer(tmp_path)
    assert cp.previous_checkpoint == "step_00000200"
    trainer = Trainer.__new__(Trainer)
    trainer.resume_from_checkpoint = True
    trainer.checkpointer = cp
    trainer.setup_trainer()
    assert trainer.current_epoch == 0
    assert trainer._resume_local_step == 200
    assert trainer.global_step == 200


def test_checkpoint_eval_due_every_100_global_steps(tmp_path: Path):
    trainer = _make_minimal_trainer(tmp_path, checkpoint_freq=1, save_best=False)
    trainer.config = trainer.config._replace(checkpoint_eval_steps=100)
    for step, expected in ((0, False), (99, False), (100, True),
                           (101, False), (200, True)):
        trainer.global_step = step
        assert trainer._checkpoint_eval_due() is expected


def test_train_loop_saves_then_validates_each_interval_and_epoch_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, document_ids, error_records):
            return None, self.weight * document_ids.float().sum(), {}

    trainer = _make_minimal_trainer(tmp_path, checkpoint_freq=1, save_best=False)
    trainer.config = trainer.config._replace(num_epochs=1, checkpoint_eval_steps=2)
    trainer.model = cast("SpeculatorModel", TinyModel())
    trainer.optimizers = [torch.optim.SGD(trainer.model.parameters(), lr=0.01)]
    trainer.train_loader = DataLoader(
        [{"document_ids": torch.tensor([1]), "error_records": 0}] * 4,
        batch_size=None,
    )
    trainer.val_loader = trainer.train_loader
    trainer.device_type = "cpu"
    trainer.local_rank = "cpu"
    trainer._progress_step_seconds = deque(maxlen=10)
    events: list[str] = []

    def fake_save(_model, _optimizer, name: str):
        events.append(f"save:{name}")
        (tmp_path / name).mkdir()

    def fake_val(_epoch: int):
        events.append("validate")
        trainer.model.eval()
        return {"loss_epoch": float(4 - trainer.global_step)}

    trainer.val_epoch = fake_val
    monkeypatch.setattr(trainer.checkpointer, "save_checkpoint", fake_save)

    trainer.run_training()

    assert events == [
        "save:step_00000002", "validate",
        "save:step_00000004", "validate",
        "save:epoch0_end", "validate",
    ]
    assert trainer.model.training
    assert (tmp_path / "checkpoint_best").resolve() == (
        tmp_path / "step_00000004"
    ).resolve()


def test_update_best_symlink_creates_and_updates(tmp_path: Path):
    (tmp_path / "1").mkdir()
    (tmp_path / "3").mkdir()

    cp = SingleGPUCheckpointer(str(tmp_path))
    cp.update_best_symlink(1)

    best_path = tmp_path / "checkpoint_best"
    assert best_path.exists()
    assert best_path.is_symlink()
    assert best_path.resolve() == (tmp_path / "1").resolve()

    cp.update_best_symlink(3)
    assert best_path.exists()
    assert best_path.is_symlink()
    assert best_path.resolve() == (tmp_path / "3").resolve()


def test_run_training_updates_checkpoint_best_among_saved_checkpoints_save_best_false(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    trainer = _make_minimal_trainer(tmp_path, checkpoint_freq=2, save_best=False)
    trainer.config = trainer.config._replace(num_epochs=4)

    saved_epochs = []
    val_losses = {
        0: 0.9,
        1: 0.6,
        2: 0.1,
        3: 0.7,
    }

    def fake_train_epoch(epoch: int):
        return None

    def fake_val_epoch(epoch: int):
        return {"loss_epoch": val_losses[epoch]}

    def fake_cp_save_checkpoint(_model, _opt, epoch: int):
        saved_epochs.append(epoch)
        (tmp_path / str(epoch)).mkdir(exist_ok=True)

    trainer.train_epoch = fake_train_epoch
    trainer.val_epoch = fake_val_epoch
    monkeypatch.setattr(
        trainer.checkpointer, "save_checkpoint", fake_cp_save_checkpoint
    )
    monkeypatch.setattr(
        trainer.checkpointer,
        "save_scheduler_state_dict",
        lambda *_args, **_kwargs: None,
    )

    trainer.run_training()

    assert saved_epochs == [0, 1, 3]

    best_path = tmp_path / "checkpoint_best"
    assert best_path.exists()
    assert best_path.is_symlink()
    assert best_path.resolve() == (tmp_path / "1").resolve()
    assert trainer.best_val_loss == 0.6


@pytest.mark.parametrize(
    (
        "save_best",
        "checkpoint_freq",
        "expected_saved",
        "expected_remaining_dirs",
        "expected_best_target",
    ),
    [
        (False, 3, [0, 2], {"0", "2"}, "2"),
        (True, 3, [0, 1, 3], {"3"}, "3"),
    ],
)
def test_save_best_flag_changes_checkpoint_behavior(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    save_best: bool,
    checkpoint_freq: int,
    expected_saved: list[int],
    expected_remaining_dirs: set[str],
    expected_best_target: str,
):
    case_dir = tmp_path / ("save_best" if save_best else "save_freq")
    case_dir.mkdir()
    trainer = _make_minimal_trainer(
        case_dir, checkpoint_freq=checkpoint_freq, save_best=save_best
    )
    trainer.config = trainer.config._replace(num_epochs=4)

    saved_epochs: list[int] = []
    val_losses = {0: 0.9, 1: 0.8, 2: 0.85, 3: 0.7}

    trainer.train_epoch = lambda _epoch: None
    trainer.val_epoch = lambda epoch: {"loss_epoch": val_losses[epoch]}

    def fake_cp_save_checkpoint(_model, _opt, epoch: int):
        saved_epochs.append(epoch)
        (case_dir / str(epoch)).mkdir(exist_ok=True)

    monkeypatch.setattr(
        trainer.checkpointer, "save_checkpoint", fake_cp_save_checkpoint
    )
    monkeypatch.setattr(
        trainer.checkpointer,
        "save_scheduler_state_dict",
        lambda *_args, **_kwargs: None,
    )

    trainer.run_training()

    assert saved_epochs == expected_saved

    remaining_dirs = {
        p.name for p in case_dir.iterdir() if p.is_dir() and p.name.isdigit()
    }
    assert remaining_dirs == expected_remaining_dirs

    best_path = case_dir / "checkpoint_best"
    assert best_path.exists()
    assert best_path.is_symlink()
    assert best_path.resolve() == (case_dir / expected_best_target).resolve()


def test_checkpoint_freq_flag_controls_saves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    trainer = _make_minimal_trainer(tmp_path, checkpoint_freq=3, save_best=False)
    trainer.config = trainer.config._replace(num_epochs=7)

    saved_epochs: list[int] = []
    val_losses = {0: 0.9, 1: 0.1, 2: 0.8, 3: 0.2, 4: 0.7, 5: 0.6, 6: 0.3}

    trainer.train_epoch = lambda _epoch: None
    trainer.val_epoch = lambda epoch: {"loss_epoch": val_losses[epoch]}

    def fake_cp_save_checkpoint(_model, _opt, epoch: int):
        saved_epochs.append(epoch)
        (tmp_path / str(epoch)).mkdir(exist_ok=True)

    monkeypatch.setattr(
        trainer.checkpointer, "save_checkpoint", fake_cp_save_checkpoint
    )
    monkeypatch.setattr(
        trainer.checkpointer,
        "save_scheduler_state_dict",
        lambda *_args, **_kwargs: None,
    )

    trainer.run_training()

    assert saved_epochs == [0, 2, 5]

    best_path = tmp_path / "checkpoint_best"
    assert best_path.exists()
    assert best_path.is_symlink()
    assert best_path.resolve() == (tmp_path / "5").resolve()


def test_save_and_load_val_metrics(tmp_path: Path):
    cp = SingleGPUCheckpointer(str(tmp_path))

    # No file yet
    assert cp.load_best_val_loss() is None

    # Save val_metrics for epoch 0 and point checkpoint_best at it
    (tmp_path / "0").mkdir()
    cp.save_val_metrics(0, {"loss_epoch": 0.123456, "full_acc_0_epoch": 0.5})
    cp.update_best_symlink(0)
    assert cp.load_best_val_loss() == pytest.approx(0.123456)

    # Save better metrics for epoch 1 and update best
    (tmp_path / "1").mkdir()
    cp.save_val_metrics(1, {"loss_epoch": 0.05, "full_acc_0_epoch": 0.7})
    cp.update_best_symlink(1)
    assert cp.load_best_val_loss() == pytest.approx(0.05)


def test_best_val_loss_restored_on_resume(tmp_path: Path):
    (tmp_path / "4").mkdir()

    cp = SingleGPUCheckpointer(str(tmp_path))
    cp.save_val_metrics(4, {"loss_epoch": 0.42, "full_acc_0_epoch": 0.6})
    cp.update_best_symlink(4)

    trainer = Trainer.__new__(Trainer)
    trainer.resume_from_checkpoint = True
    trainer.is_distributed = False
    trainer.rank = 0
    trainer.local_rank = 0
    trainer.checkpointer = cp

    trainer.current_epoch = cp.previous_epoch + 1
    trainer.global_step = 0
    trainer.best_val_loss = float("inf")

    # Simulate the load that setup_trainer does
    saved = trainer.checkpointer.load_best_val_loss()
    if saved is not None:
        trainer.best_val_loss = saved

    assert trainer.best_val_loss == pytest.approx(0.42)


def test_graceful_shutdown_saves_interrupted_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    trainer = _make_minimal_trainer(tmp_path, checkpoint_freq=99, save_best=False)
    trainer.config = trainer.config._replace(num_epochs=4)

    saved_labels: list[int | str] = []

    def fake_train_epoch(epoch: int):
        if epoch == 2:
            os.kill(os.getpid(), signal.SIGINT)

    def fake_val_epoch(epoch: int):
        return {"loss_epoch": 0.5}

    def fake_cp_save_checkpoint(_model, _opt, epoch: int | str):
        saved_labels.append(epoch)
        (tmp_path / str(epoch)).mkdir(exist_ok=True)

    trainer.train_epoch = fake_train_epoch
    trainer.val_epoch = fake_val_epoch
    monkeypatch.setattr(
        trainer.checkpointer, "save_checkpoint", fake_cp_save_checkpoint
    )
    monkeypatch.setattr(
        trainer.checkpointer,
        "save_scheduler_state_dict",
        lambda *_args, **_kwargs: None,
    )

    trainer.run_training()

    assert "interrupted" in saved_labels
