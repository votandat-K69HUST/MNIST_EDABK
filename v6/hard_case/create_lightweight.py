import os
import json

cfg = {
    "name": "mlp_hard_case_lightweight",
    "seed": 42,
    "num_threads": 8,
    "data": {
        "train": [
            {"path": "data/output/hard_cases_dataset.npz", "weight": 1.0},
            {"path": "data/output/v5_new_train.npz", "weight": 0.3}
        ],
        "total_train": 0,
        "val": {"real": ["data/output/v5_new_val.npz"]}
    },
    "augment": {
        "enabled": True,
        "rotate": 8,
        "shear": 10,
        "shift": 2,
        "noise_std": 0.03,
        "thickness_p": 0.3
    },
    "model": {
        "hidden": [256, 128],
        "dropout": 0.2,
        "activation": "relu",
        "batchnorm": True,
        "input_dropout": 0.0
    },
    "train": {
        "epochs": 20,
        "batch_size": 128,
        "optimizer": "adamw",
        "lr": 0.001,
        "weight_decay": 1e-4,
        "scheduler": "cosine",
        "warmup_epochs": 1,
        "min_lr_frac": 0.01,
        "label_smoothing": 0.05,
        "grad_clip": 1.0,
        "patience": 5
    },
    "base_dir": "D:\\2025\\2025.2\LAB\\AI_ML\\Week_11\\AI_HACKATHON"
}

os.makedirs("v6/hard_case/lightweight", exist_ok=True)
with open("v6/hard_case/lightweight/config.json", "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=4)

print("Created lightweight config successfully.")
