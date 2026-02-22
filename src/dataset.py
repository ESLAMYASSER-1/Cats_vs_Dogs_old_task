
import sys
import shutil
import zipfile
import subprocess
from pathlib import Path

from .config import CFG


def download_dataset(dataset_dir: Path = CFG["dataset_dir"]) -> None:

    dataset_dir.mkdir(parents=True, exist_ok=True)

    zip_path = dataset_dir / "cats-and-dogs-image-classification.zip"

    _download_zip(zip_path, dataset_dir)
    raw_dir = _extract_zip(zip_path, dataset_dir)
    _build_train_val_split(raw_dir)


def _download_zip(zip_path: Path, dataset_dir: Path) -> None:
    if zip_path.exists():
        print(f"[dataset] Archive already present at {zip_path}. Skipping download.")
        return

    print(f"[dataset] Downloading '{CFG['kaggle_dataset']}' …")
    cmd = [
        sys.executable, "-m", "kaggle",
        "datasets", "download",
        "-d", CFG["kaggle_dataset"],
        "--path", str(dataset_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("[dataset] STDOUT:", result.stdout)
        print("[dataset] STDERR:", result.stderr)
        raise RuntimeError(
            "Kaggle download failed. Ensure:\n"
            "  1. kaggle is installed        (`pip install kaggle`)\n"
            "  2. ~/.kaggle/kaggle.json exists with valid credentials\n"
            "  3. You have accepted the dataset rules on Kaggle.com"
        )
    print("[dataset] Download complete.")


def _extract_zip(zip_path: Path, dataset_dir: Path) -> Path:
    extract_dir = dataset_dir / "raw"
    if extract_dir.exists():
        print(f"[dataset] Already extracted to {extract_dir}. Skipping unzip.")
        return extract_dir

    print(f"[dataset] Extracting to {extract_dir} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    print("[dataset] Extraction complete.")
    return extract_dir


def _build_train_val_split(
    raw_dir:   Path,
    train_dir: Path  = CFG["train_dir"],
    val_dir:   Path  = CFG["val_dir"],
    val_split: float = CFG["val_split"],
    seed:      int   = CFG["seed"],
) -> None:

    import random
    random.seed(seed)

    source_root = _find_source_root(raw_dir)

    class_dirs = {
        d.name: d
        for d in source_root.iterdir()
        if d.is_dir() and any(tag in d.name.lower() for tag in ("cat", "dog"))
    }
    if not class_dirs:
        raise FileNotFoundError(f"No cat/dog class folders found under {source_root}.")

    if train_dir.exists() and val_dir.exists():
        print("[dataset] Train/val directories already exist. Skipping split.")
        return

    print(f"[dataset] Building {1 - val_split:.0%} train / {val_split:.0%} val split …")

    for cls_name, cls_src in class_dirs.items():
        images = sorted(
            cls_src.glob("*.jpg"),
            key=lambda p: p.name,
        )
        images += sorted(cls_src.glob("*.jpeg"), key=lambda p: p.name)
        images += sorted(cls_src.glob("*.png"),  key=lambda p: p.name)

        random.shuffle(images)
        n_val  = int(len(images) * val_split)
        splits = {"val": images[:n_val], "train": images[n_val:]}

        for split_name, split_images in splits.items():
            dest = (train_dir if split_name == "train" else val_dir) / cls_name
            dest.mkdir(parents=True, exist_ok=True)
            for img_path in split_images:
                shutil.copy2(img_path, dest / img_path.name)

    for split_name, split_dir in [("train", train_dir), ("val", val_dir)]:
        counts = {
            d.name: len(list(d.glob("*")))
            for d in split_dir.iterdir()
            if d.is_dir()
        }
        print(f"  {split_name:5s}: {counts}")

    print("[dataset] Split complete.")


def _find_source_root(raw_dir: Path) -> Path:
    for candidate in (raw_dir / "train", raw_dir):
        if not candidate.exists():
            continue
        subdirs = [d for d in candidate.iterdir() if d.is_dir()]
        if any("cat" in d.name.lower() or "dog" in d.name.lower() for d in subdirs):
            return candidate

    raise FileNotFoundError(
        f"Could not locate cat/dog class folders under {raw_dir}.\n"
        "Please verify the extracted archive structure."
    )
