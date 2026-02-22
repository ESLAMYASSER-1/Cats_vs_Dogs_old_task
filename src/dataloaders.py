
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import CFG, DEVICE


def create_dataloaders(
    train_dir:   Path = CFG["train_dir"],
    val_dir:     Path = CFG["val_dir"],
    batch_size:  int  = CFG["batch_size"],
    num_workers: int  = CFG["num_workers"],
    image_size:  int  = CFG["image_size"],
    mean:        list = CFG["mean"],
    std:         list = CFG["std"],
) -> tuple[DataLoader, DataLoader, list[str]]:

    train_transforms = _build_train_transforms(image_size, mean, std)
    val_transforms   = _build_val_transforms(image_size, mean, std)

    train_dataset = datasets.ImageFolder(str(train_dir), transform=train_transforms)
    val_dataset   = datasets.ImageFolder(str(val_dir),   transform=val_transforms)

    pin = DEVICE.type == "cuda"  

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    class_names = train_dataset.classes
    _print_summary(train_dataset, val_dataset, train_loader, val_loader, class_names)

    return train_loader, val_loader, class_names



def _build_train_transforms(
    image_size: int,
    mean: list,
    std: list,
) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def _build_val_transforms(
    image_size: int,
    mean: list,
    std: list,
) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(image_size + 32),   
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def _print_summary(
    train_ds, val_ds,
    train_dl, val_dl,
    class_names: list[str],
) -> None:
    print(f"[data] Classes : {class_names}")
    print(f"[data] Train   : {len(train_ds):,} images | {len(train_dl)} batches")
    print(f"[data] Val     : {len(val_ds):,}  images | {len(val_dl)} batches")
