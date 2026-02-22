from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

from .config import CFG, DEVICE


def predict_image(
    model:       nn.Module,
    image_path:  str | Path,
    class_names: list[str],
) -> dict:

    tf = _get_inference_transforms()

    img    = Image.open(image_path).convert("RGB")
    tensor = tf(img).unsqueeze(0).to(DEVICE)   # (1, 3, H, W)

    model.eval()
    with torch.no_grad():
        logit = model(tensor).squeeze().item()
        prob  = torch.sigmoid(torch.tensor(logit)).item()

    pred_idx   = int(prob >= 0.5)
    pred_class = class_names[pred_idx]
    confidence = prob if pred_idx == 1 else 1.0 - prob

    return {
        "class_name": pred_class,
        "confidence": round(confidence, 4),
        "logit":      round(logit, 4),
    }


def _get_inference_transforms() -> transforms.Compose:
    size = CFG["image_size"]
    return transforms.Compose([
        transforms.Resize(size + 32),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=CFG["mean"], std=CFG["std"]),
    ])
