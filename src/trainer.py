import copy
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from .config import CFG, DEVICE
from .model import build_model, unfreeze_last_block
from .checkpoint import save_checkpoint



class EarlyStopping:

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience    = patience
        self.min_delta   = min_delta
        self.counter     = 0
        self.best_loss   = float("inf")
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
 
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
) -> tuple[float, float]:

    model.train()
    running_loss = 0.0
    correct      = 0
    total        = 0

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
 
        labels = labels.float().to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        logits = model(images).squeeze(1) 
        loss   = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = (torch.sigmoid(logits) >= 0.5).long()
            correct      += (preds == labels.long()).sum().item()
            total        += labels.size(0)
            running_loss += loss.item() * labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def validate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct      = 0
    total        = 0

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.float().to(DEVICE, non_blocking=True)

        logits = model(images).squeeze(1)
        loss   = criterion(logits, labels)

        preds = (torch.sigmoid(logits) >= 0.5).long()
        correct      += (preds == labels.long()).sum().item()
        total        += labels.size(0)
        running_loss += loss.item() * labels.size(0)

    return running_loss / total, correct / total

def _run_phase(
    phase_name:      str,
    model:           nn.Module,
    train_loader:    DataLoader,
    val_loader:      DataLoader,
    criterion:       nn.Module,
    optimizer:       optim.Optimizer,
    scheduler:       object,
    n_epochs:        int,
    checkpoint_path: Path,
    early_stopper:   EarlyStopping,
    history:         list,
) -> tuple[nn.Module, float]:

    best_val_acc   = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    print(f"\n{'─' * 70}")
    print(f"  {phase_name}  |  device: {DEVICE}  |  epochs: {n_epochs}")
    print(f"{'─' * 70}")

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss,   val_acc   = validate(model, val_loader, criterion)

        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:>3}/{n_epochs}  "
            f"| train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"| val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"| {elapsed:.1f}s"
        )

        history.append({
            "phase":      phase_name,
            "epoch":      epoch,
            "train_loss": train_loss,
            "train_acc":  train_acc,
            "val_loss":   val_loss,
            "val_acc":    val_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            save_checkpoint(model, checkpoint_path, {
                "phase":    phase_name,
                "epoch":    epoch,
                "val_acc":  val_acc,
                "val_loss": val_loss,
            })

        if early_stopper.step(val_loss):
            print(
                f"  [early stop] No improvement for "
                f"{early_stopper.patience} epochs. Stopping."
            )
            break

    model.load_state_dict(best_model_wts)
    print(f"  ✔ Best val accuracy ({phase_name}): {best_val_acc:.4f}")
    return model, best_val_acc


def train_model(
    train_loader: DataLoader,
    val_loader:   DataLoader,
    cfg:          dict = CFG,
) -> tuple[nn.Module, list]:

    ckpt_dir  = cfg["checkpoint_dir"]
    criterion = nn.BCEWithLogitsLoss()
    history: list[dict] = []

    print("\n══════════════════════════════════════════")
    print("  PHASE 1 – Training classification head")
    print("══════════════════════════════════════════")

    model        = build_model(freeze_backbone=True)
    optimizer_p1 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["phase1_lr"],
    )
    scheduler_p1 = lr_scheduler.ReduceLROnPlateau(
        optimizer_p1,
        mode="min",
        factor=cfg["lr_factor"],
        patience=cfg["lr_patience"],
    )

    model, _ = _run_phase(
        phase_name      = "Phase1",
        model           = model,
        train_loader    = train_loader,
        val_loader      = val_loader,
        criterion       = criterion,
        optimizer       = optimizer_p1,
        scheduler       = scheduler_p1,
        n_epochs        = cfg["phase1_epochs"],
        checkpoint_path = ckpt_dir / "best_phase1.pth",
        early_stopper   = EarlyStopping(patience=cfg["early_stop_patience"]),
        history         = history,
    )

    print("\n══════════════════════════════════════════")
    print("  PHASE 2 – Fine-tuning (layer4 + head)")
    print("══════════════════════════════════════════")

    model = unfreeze_last_block(model)

    head_params    = list(model.fc.parameters())
    head_ids       = {id(p) for p in head_params}
    backbone_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in head_ids
    ]

    optimizer_p2 = optim.Adam([
        {"params": backbone_params, "lr": cfg["phase2_lr_backbone"]},
        {"params": head_params,     "lr": cfg["phase2_lr_head"]},
    ])
    scheduler_p2 = lr_scheduler.ReduceLROnPlateau(
        optimizer_p2,
        mode="min",
        factor=cfg["lr_factor"],
        patience=cfg["lr_patience"],
    )

    model, best_acc = _run_phase(
        phase_name      = "Phase2",
        model           = model,
        train_loader    = train_loader,
        val_loader      = val_loader,
        criterion       = criterion,
        optimizer       = optimizer_p2,
        scheduler       = scheduler_p2,
        n_epochs        = cfg["phase2_epochs"],
        checkpoint_path = ckpt_dir / "best_phase2.pth",
        early_stopper   = EarlyStopping(patience=cfg["early_stop_patience"]),
        history         = history,
    )

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history_path = ckpt_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[trainer] History saved → {history_path}")
    print(f"[trainer] Final best val accuracy: {best_acc:.4f}")

    return model, history
