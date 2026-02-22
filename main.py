import torch

from cats_vs_dogs import (
    CFG,
    DEVICE,
    download_dataset,
    create_dataloaders,
    train_model,
    predict_image,
    load_checkpoint,
    build_model,
)


def main() -> None:
    torch.manual_seed(CFG["seed"])
    print(f"[main] Device : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"[main] GPU    : {torch.cuda.get_device_name(0)}")

    download_dataset()

    train_loader, val_loader, class_names = create_dataloaders()


    model, history = train_model(train_loader, val_loader)


    print("\n[main] ✅ All done!")
    print(f"[main] Best checkpoint → {CFG['checkpoint_dir']}/best_phase2.pth")


if __name__ == "__main__":
    main()
