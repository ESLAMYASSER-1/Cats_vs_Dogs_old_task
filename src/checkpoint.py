from pathlib import Path

import torch
import torch.nn as nn

from .config import DEVICE


def save_checkpoint(model: nn.Module, path: Path, meta: dict) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **meta}, path)
    print(f"  [ckpt] Saved → {path}")


def load_checkpoint(model: nn.Module, path: Path) -> dict:

    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    
    return {k: v for k, v in ckpt.items() if k != "state_dict"}
