"""Training loop for both classification and CTC sequence modes.

Usage::

    from src.training.trainer import Trainer
    Trainer(cfg, model_name="lstm").run()

The trainer writes:
  * ``checkpoints/{model_name}_best.pt`` whenever val accuracy improves
  * ``checkpoints/{model_name}_last.pt`` every epoch
  * ``logs/{model_name}_history.json`` with per-epoch loss/accuracy
  * ``figures/{model_name}_training_curves.png`` after the run
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import LandmarkSequenceDataset, SequenceAugmenter, pad_collate
from src.models import build_model, count_parameters
from src.training.metrics import RunningAverage, topk_accuracy
from src.utils import get_logger, seed_everything

log = get_logger(__name__)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _make_loader(ds, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=pad_collate,
        pin_memory=False,
        drop_last=False,
    )


def _build_scheduler(opt, kind: str, total_epochs: int, warmup: int):
    """Cosine schedule with linear warmup, or ReduceLROnPlateau, or none."""
    if kind == "none":
        return None
    if kind == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=4,
        )
    if kind == "cosine":
        def lr_lambda(epoch: int) -> float:
            if epoch < warmup:
                return float(epoch + 1) / float(max(1, warmup))
            progress = (epoch - warmup) / max(1, total_epochs - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    raise ValueError(f"Unknown scheduler: {kind}")


class Trainer:
    def __init__(self, cfg, model_name: str = "lstm") -> None:
        self.cfg = cfg
        self.model_name = model_name
        self.device = _device()
        log.info("Using device: %s", self.device)

        seed_everything(int(cfg.training.seed))

        # ---------- data ----------
        manifest = Path(cfg.paths.processed) / "manifest.csv"
        landmarks = Path(cfg.paths.landmarks)
        max_T = int(cfg.dataset.max_seq_len)
        min_T = int(cfg.dataset.min_seq_len)

        augmenter = SequenceAugmenter(
            cfg.augmentation, num_hands=int(cfg.preprocessing.num_hands),
            seed=int(cfg.training.seed),
        )

        self.train_ds = LandmarkSequenceDataset(
            manifest, landmarks, "train", max_T, min_T, augmenter=augmenter,
        )
        self.val_ds = LandmarkSequenceDataset(
            manifest, landmarks, "val", max_T, min_T, augmenter=None,
        )

        bs = int(cfg.training.batch_size)
        nw = int(cfg.training.num_workers)
        self.train_loader = _make_loader(self.train_ds, bs, True, nw)
        self.val_loader = _make_loader(self.val_ds, bs, False, nw)

        # ---------- model ----------
        sample_seq, _, _ = self.train_ds[0]
        input_dim = int(sample_seq.shape[1])
        label_map = json.load((Path(cfg.paths.processed) / "label_map.json").open())
        num_classes = len(label_map)

        self.model = build_model(model_name, input_dim, num_classes, cfg).to(self.device)
        log.info(
            "Built %s model with %d parameters; input_dim=%d, num_classes=%d.",
            model_name, count_parameters(self.model), input_dim, num_classes,
        )

        # ---------- optim / loss ----------
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(cfg.training.learning_rate),
            weight_decay=float(cfg.training.weight_decay),
        )
        self.scheduler = _build_scheduler(
            self.optimizer,
            str(cfg.training.scheduler).lower(),
            int(cfg.training.epochs),
            int(cfg.training.warmup_epochs),
        )

        self.ctc_mode = bool(cfg.training.ctc_mode)
        if self.ctc_mode:
            # CTC reserves index 0 as the blank, so models are expected to have
            # been built with num_classes that already includes a blank at idx 0.
            # We don't require the user to do that; we shift labels by +1 at
            # train time and report top-k by ignoring the blank slot.
            self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        else:
            self.criterion = nn.CrossEntropyLoss(
                label_smoothing=float(cfg.training.label_smoothing),
            )

        self.amp = bool(cfg.training.mixed_precision) and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler() if self.amp else None

        self.ckpt_dir = Path(cfg.paths.checkpoints); self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(cfg.paths.logs); self.log_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir = Path(cfg.paths.figures); self.fig_dir.mkdir(parents=True, exist_ok=True)

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.label_map = label_map

    # ------------------------------------------------------------------
    def _forward_loss(self, x, lengths, labels):
        x = x.to(self.device, non_blocking=True)
        lengths = lengths.to(self.device)
        labels = labels.to(self.device)
        if self.ctc_mode:
            log_probs, lens = self.model.forward_logits_per_step(x, lengths)
            # CTC expects (T, B, C) log-probs, plus target lengths.
            B = labels.size(0)
            target = (labels + 1).unsqueeze(1)        # shift past blank
            target_lengths = torch.ones(B, dtype=torch.long, device=self.device)
            input_lengths = lens.to(self.device)
            loss = self.criterion(
                log_probs, target, input_lengths, target_lengths,
            )
            # Approximate clip-level logits for accuracy reporting:
            # average per-step logits over true length, drop blank class.
            T_max = log_probs.size(0)
            ar = torch.arange(T_max, device=self.device).unsqueeze(1)
            mask = (ar < input_lengths.unsqueeze(0)).float().unsqueeze(-1)
            avg_logits = (log_probs * mask).sum(dim=0) / mask.sum(dim=0).clamp_min(1.0)
            cls_logits = avg_logits[:, 1:]  # drop blank
            return loss, cls_logits, labels
        logits = self.model(x, lengths)
        loss = self.criterion(logits, labels)
        return loss, logits, labels

    def _train_epoch(self) -> tuple[float, dict[int, float]]:
        self.model.train()
        loss_avg = RunningAverage()
        all_logits, all_labels = [], []

        for x, lengths, labels, _ in self.train_loader:
            self.optimizer.zero_grad(set_to_none=True)
            if self.amp:
                with torch.cuda.amp.autocast():
                    loss, logits, lbl = self._forward_loss(x, lengths, labels)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), float(self.cfg.training.grad_clip),
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss, logits, lbl = self._forward_loss(x, lengths, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), float(self.cfg.training.grad_clip),
                )
                self.optimizer.step()

            loss_avg.update(loss.item(), n=lbl.size(0))
            all_logits.append(logits.detach().cpu())
            all_labels.append(lbl.detach().cpu())

        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        accs = topk_accuracy(logits, labels, ks=(1, 5))
        return loss_avg.value, accs

    @torch.no_grad()
    def _eval_epoch(self) -> tuple[float, dict[int, float]]:
        self.model.eval()
        loss_avg = RunningAverage()
        all_logits, all_labels = [], []
        for x, lengths, labels, _ in self.val_loader:
            loss, logits, lbl = self._forward_loss(x, lengths, labels)
            loss_avg.update(loss.item(), n=lbl.size(0))
            all_logits.append(logits.cpu())
            all_labels.append(lbl.cpu())
        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        accs = topk_accuracy(logits, labels, ks=(1, 5))
        return loss_avg.value, accs

    # ------------------------------------------------------------------
    def _save_checkpoint(self, path: Path, epoch: int, val_acc: float) -> None:
        torch.save({
            "model_name": self.model_name,
            "state_dict": self.model.state_dict(),
            "epoch": epoch,
            "val_top1": val_acc,
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "label_map": self.label_map,
            "config": self.cfg.to_dict(),
        }, path)

    def _plot_curves(self, history: list[dict]) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            log.warning("matplotlib not installed; skipping training curves.")
            return
        epochs = [h["epoch"] for h in history]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
        axes[0].plot(epochs, [h["val_loss"] for h in history], label="val")
        axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
        axes[1].plot(epochs, [h["train_top1"] for h in history], label="train top-1")
        axes[1].plot(epochs, [h["val_top1"] for h in history], label="val top-1")
        axes[1].plot(epochs, [h["val_top5"] for h in history], label="val top-5", linestyle="--")
        axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend()
        fig.suptitle(f"{self.model_name.upper()} training curves")
        fig.tight_layout()
        out = self.fig_dir / f"{self.model_name}_training_curves.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        log.info("Saved training curves to %s", out)

    # ------------------------------------------------------------------
    def run(self) -> dict:
        epochs = int(self.cfg.training.epochs)
        patience = int(self.cfg.training.early_stopping_patience)

        best_val = -1.0
        epochs_since_improve = 0
        history: list[dict] = []

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = self._train_epoch()
            val_loss, val_acc = self._eval_epoch()
            dt = time.time() - t0

            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            elif self.scheduler is not None:
                self.scheduler.step()

            history.append({
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_top1": tr_acc[1],
                "train_top5": tr_acc[5],
                "val_loss": val_loss,
                "val_top1": val_acc[1],
                "val_top5": val_acc[5],
                "time_sec": dt,
            })

            log.info(
                "Epoch %3d/%d | train loss %.4f top1 %.3f | val loss %.4f top1 %.3f top5 %.3f | %.1fs",
                epoch, epochs, tr_loss, tr_acc[1],
                val_loss, val_acc[1], val_acc[5], dt,
            )

            self._save_checkpoint(self.ckpt_dir / f"{self.model_name}_last.pt",
                                  epoch, val_acc[1])

            if val_acc[1] > best_val:
                best_val = val_acc[1]
                epochs_since_improve = 0
                self._save_checkpoint(
                    self.ckpt_dir / f"{self.model_name}_best.pt",
                    epoch, val_acc[1],
                )
                log.info("New best val top-1: %.4f", best_val)
            else:
                epochs_since_improve += 1
                if epochs_since_improve >= patience:
                    log.info("Early stopping at epoch %d (no improvement for %d).",
                             epoch, patience)
                    break

        hist_path = self.log_dir / f"{self.model_name}_history.json"
        with hist_path.open("w") as f:
            json.dump(history, f, indent=2)
        log.info("Wrote training history to %s", hist_path)

        self._plot_curves(history)
        return {"best_val_top1": best_val, "history": history}
