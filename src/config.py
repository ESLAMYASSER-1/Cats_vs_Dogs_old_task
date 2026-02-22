from pathlib import Path
import torch

CFG: dict = {
    "dataset_dir":    Path("data"),          
    "train_dir":      Path("data/train"),     
    "val_dir":        Path("data/val"),       
    "checkpoint_dir": Path("checkpoints"),

    "kaggle_dataset": "samuelcortinhas/cats-and-dogs-image-classification",

   
    "image_size":  224,
    "batch_size":  32,
    "num_workers": 4,   
    "val_split":   0.2, 

    "mean": [0.485, 0.456, 0.406],
    "std":  [0.229, 0.224, 0.225],

    "phase1_epochs": 5,
    "phase1_lr":     1e-3,   


    "phase2_epochs":      10,
    "phase2_lr_head":     1e-4, 
    "phase2_lr_backbone": 1e-5,  


    "lr_patience": 2,    
    "lr_factor":   0.3,  

    "early_stop_patience": 5,

    "seed": 42,

    "classes": ["cat", "dog"],
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
