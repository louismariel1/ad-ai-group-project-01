from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def _ewm_mean(values: list[float], alpha: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, arr.size):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


class TrainingMetrics:
    """Training-curve tracker from Practical 4, adapted to save final figures."""

    def __init__(self, title: str = ""):
        self.title = title
        self.train_loss: list[float] = []
        self.val_loss: list[float] = []
        self.train_acc: list[float] = []
        self.val_acc: list[float] = []
        self.step = 0
        self.epoch = 0
        self.epoch_steps = [0]
        self.training_time_seconds: float | None = None
        self.peak_cuda_memory_bytes: int | None = None
        self.total_parameters: int | None = None
        self.trainable_parameters: int | None = None
        self.fig = None
        self.plot_axes = None

    def log_train(self, loss: float, acc: float) -> None:
        self.train_loss.append(float(loss))
        self.train_acc.append(float(acc))
        self.step += 1

    def log_val(self, loss: float, acc: float) -> None:
        self.val_loss.append(float(loss))
        self.val_acc.append(float(acc))
        self.epoch_steps.append(self.step)
        self.epoch += 1

    @property
    def epoch_loss(self) -> tuple[float, float]:
        cur, last = self.epoch_steps[-1], self.epoch_steps[-2]
        return float(np.mean(self.train_loss[last:cur])), float(self.val_loss[-1])

    @property
    def epoch_acc(self) -> tuple[float, float]:
        cur, last = self.epoch_steps[-1], self.epoch_steps[-2]
        return float(np.mean(self.train_acc[last:cur])), float(self.val_acc[-1])

    @property
    def best_val_acc(self) -> float:
        return float(np.max(self.val_acc)) if self.val_acc else 0.0

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_acc": self.train_acc,
            "val_acc": self.val_acc,
            "epoch_steps": self.epoch_steps,
            "best_val_acc": self.best_val_acc,
            "training_time_seconds": self.training_time_seconds,
            "peak_cuda_memory_bytes": self.peak_cuda_memory_bytes,
            "total_parameters": self.total_parameters,
            "trainable_parameters": self.trainable_parameters,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "TrainingMetrics":
        metrics = cls(title=payload.get("title", ""))
        metrics.train_loss = list(payload["train_loss"])
        metrics.val_loss = list(payload["val_loss"])
        metrics.train_acc = list(payload["train_acc"])
        metrics.val_acc = list(payload["val_acc"])
        metrics.epoch_steps = list(payload["epoch_steps"])
        metrics.step = len(metrics.train_loss)
        metrics.epoch = max(len(metrics.val_acc), 0)
        metrics.training_time_seconds = payload.get("training_time_seconds")
        metrics.peak_cuda_memory_bytes = payload.get("peak_cuda_memory_bytes")
        metrics.total_parameters = payload.get("total_parameters")
        metrics.trainable_parameters = payload.get("trainable_parameters")
        return metrics

    def plot(
        self,
        title: str | None = None,
        baseline_accs: dict[str, float] | None = None,
        baseline_metrics: "TrainingMetrics | None" = None,
        alpha: float = 0.1,
        max_step: int = 0,
        save_path: Path | None = None,
        show: bool = False,
    ):
        if self.fig is None:
            self.fig, self.plot_axes = plt.subplots(1, 2, figsize=(10, 4))
        loss_ax, acc_ax = self.plot_axes
        x_step = np.arange(0, self.step)

        smooth_train_loss = _ewm_mean(self.train_loss, alpha) if self.train_loss else []
        smooth_train_acc = _ewm_mean(self.train_acc, alpha) if self.train_acc else []

        loss_ax.clear()
        if len(x_step):
            loss_ax.plot(x_step, smooth_train_loss, c="tab:orange", linestyle=":", label="train loss")
        if len(self.val_loss):
            loss_ax.plot(self.epoch_steps[1:], self.val_loss, c="tab:orange", linestyle="-", label="val loss")

        num_epochs = len(self.epoch_steps)
        steps_per_epoch = self.epoch_steps[1] if num_epochs > 1 else 1
        num_expected_epochs = max(max_step // max(steps_per_epoch, 1), num_epochs, 1)
        tick_spacing = max(num_expected_epochs // 10, 1)
        loss_ax.set_xticks(self.epoch_steps[::tick_spacing], range(0, num_epochs, tick_spacing))
        loss_ax.legend()
        loss_ax.set_xlabel("Epoch")
        loss_ax.set_ylabel("Loss")

        acc_ax.clear()
        if len(x_step):
            acc_ax.plot(x_step, smooth_train_acc, c="tab:blue", linestyle=":", label="train acc")
        if len(self.val_acc):
            acc_ax.plot(self.epoch_steps[1:], self.val_acc, c="tab:blue", linestyle="-", label="val acc")
        acc_ax.set_xticks(self.epoch_steps[::tick_spacing], range(0, num_epochs, tick_spacing))
        acc_ax.legend()
        acc_ax.set_xlabel("Epoch")
        acc_ax.set_ylabel("Accuracy (%)")

        baseline_end_step = 0
        if baseline_metrics is not None and baseline_metrics.train_loss:
            b_x = np.arange(len(baseline_metrics.train_loss))
            b_train_loss = _ewm_mean(baseline_metrics.train_loss, alpha)
            b_train_acc = _ewm_mean(baseline_metrics.train_acc, alpha)
            loss_ax.plot(b_x, b_train_loss, c="lightgray", linestyle=":", zorder=-1)
            acc_ax.plot(b_x, b_train_acc, c="lightgray", linestyle=":", zorder=-1)
            if baseline_metrics.val_loss:
                loss_ax.plot(baseline_metrics.epoch_steps[1:], baseline_metrics.val_loss, c="lightgray", linestyle="-", zorder=-1)
                acc_ax.plot(baseline_metrics.epoch_steps[1:], baseline_metrics.val_acc, c="lightgray", linestyle="-", zorder=-1)
            baseline_end_step = len(baseline_metrics.train_loss)

        acc_ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1.0))
        acc_ax.yaxis.tick_right()
        acc_ax.yaxis.set_label_position("right")

        if baseline_accs:
            for name, acc in baseline_accs.items():
                acc_ax.axhline(acc, c=[0.8] * 3, linestyle=":")
                acc_ax.text(0, acc + 0.003, name, c=[0.6] * 3, size=8)

        end_step = max(max_step, self.step, baseline_end_step, 1)
        loss_ax.set_xlim([0 - (end_step * 0.05), end_step * 1.05])
        acc_ax.set_xlim([0 - (end_step * 0.05), end_step * 1.05])
        loss_ax.set_ylim([0, None])
        acc_ax.set_ylim([None, 1])

        self.fig.suptitle(title or self.title)
        plt.tight_layout()
        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            self.fig.savefig(save_path, dpi=150)
        if show:
            plt.show()
        else:
            plt.close(self.fig)
            self.fig, self.plot_axes = None, None
        return save_path


def plot_comparison_bars(
    scores: dict[str, float],
    title: str,
    save_path: Path,
    ylabel: str = "Accuracy",
) -> Path:
    names = list(scores.keys())
    values = [scores[n] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, values, color="tab:blue", alpha=0.85)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylabel.lower().startswith("acc"):
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1.0))
        ax.set_ylim(0, 1)
    for i, value in enumerate(values):
        ax.text(i, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path
