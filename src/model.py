
import torch.nn as nn
from torchvision import models

from .config import CFG, DEVICE


def build_model(freeze_backbone: bool = True) -> nn.Module:

    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model   = models.resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features   
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 1),      
    )

    model = model.to(DEVICE)
    _print_param_summary(model, freeze_backbone)
    return model


def unfreeze_last_block(model: nn.Module) -> nn.Module:

    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if name.startswith(("layer4", "fc")):
            param.requires_grad = True

    _print_param_summary(model, freeze_backbone=False)
    return model


def _print_param_summary(model: nn.Module, freeze_backbone: bool) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    state     = "backbone frozen" if freeze_backbone else "layer4 + fc unfrozen"
    print(f"[model] ResNet-18 | trainable: {trainable:,} / {total:,}  ({state})")
