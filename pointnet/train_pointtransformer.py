import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Subset

from point_transformer_model import PointTransformerRegressor
from pointnet_npz_dataset import NPZPointParamDataset, TARGET_COLS


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ===== Inline configuration (edit here) =====
TRAIN_SHARDS = r"C:\Users\Eladio\Documents\Intro to Deep Learning\BlendedNet Dataset - Released\train\vtk_npz"
TEST_SHARDS = r"C:\Users\Eladio\Documents\Intro to Deep Learning\BlendedNet Dataset - Released\test\vtk_npz"
TRAIN_CSV = r"C:\Users\Eladio\Documents\Intro to Deep Learning\BlendedNet Dataset - Released\clarc_blended_wing_body-main\csv_files\geom_params_train.csv"
TEST_CSV = r"C:\Users\Eladio\Documents\Intro to Deep Learning\BlendedNet Dataset - Released\clarc_blended_wing_body-main\csv_files\geom_params_test.csv"

OUT_DIR = Path(r"C:\Users\Eladio\Documents\Intro to Deep Learning\BlendedNet Dataset - Released\clarc_blended_wing_body-main\runs_pointtransformer")

NUM_POINTS = 8192
BATCH_SIZE = 48
EPOCHS = 300
LR = 5e-4
WEIGHT_DECAY = 1e-4
VAL_FRACTION = 0.1
NUM_WORKERS = 0

EMBED_DIM = 128
DEPTH = 2
HEADS = 4
TOKEN_COUNT = 256
MLP_RATIO = 4.0
DROPOUT = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def point_collate_fn(batch, num_points: int = NUM_POINTS):
    pts_list, y_list, geom_list, file_list = [], [], [], []

    for pts, y, geom, fname in batch:
        n = pts.shape[1]
        if n >= num_points:
            idx = torch.randperm(n)[:num_points]
        else:
            extra = torch.randint(0, n, (num_points - n,))
            idx = torch.cat([torch.arange(n), extra], dim=0)

        pts_list.append(pts[:, idx])
        y_list.append(y)
        geom_list.append(geom)
        file_list.append(fname)

    pts_b = torch.stack(pts_list, dim=0)
    y_b = torch.stack(y_list, dim=0)
    return pts_b, y_b, geom_list, file_list


def split_by_geom(dataset, val_frac: float = VAL_FRACTION):
    geom_to_indices = defaultdict(list)
    for i in range(len(dataset)):
        _, _, geom, _ = dataset[i]
        geom_to_indices[geom].append(i)

    geoms = list(geom_to_indices.keys())
    rng = np.random.default_rng(SEED)
    rng.shuffle(geoms)

    n_val = max(1, int(len(geoms) * val_frac))
    val_geoms = set(geoms[:n_val])

    train_idx, val_idx = [], []
    for geom, idxs in geom_to_indices.items():
        if geom in val_geoms:
            val_idx.extend(idxs)
        else:
            train_idx.extend(idxs)

    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def compute_y_stats(dataset: torch.utils.data.Dataset):
    ys = []
    for i in range(len(dataset)):
        _, y, _, _ = dataset[i]
        ys.append(y.numpy())
    ys = np.stack(ys, axis=0)
    mu = ys.mean(axis=0).astype(np.float32)
    std = (ys.std(axis=0) + 1e-8).astype(np.float32)
    return mu, std


def train_one_epoch(model, loader, optimizer, criterion, mu_t, std_t):
    model.train()
    total_loss = 0.0
    total_count = 0

    for pts, y, _, _ in loader:
        pts = pts.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        y_n = (y - mu_t) / std_t

        optimizer.zero_grad(set_to_none=True)
        pred_n = model(pts)
        loss = criterion(pred_n, y_n)
        loss.backward()
        optimizer.step()

        bs = pts.size(0)
        total_loss += loss.item() * bs
        total_count += bs

    return total_loss / max(1, total_count)


@torch.no_grad()
def eval_loss(model, loader, criterion, mu_t, std_t):
    model.eval()
    total_loss = 0.0
    total_count = 0

    for pts, y, _, _ in loader:
        pts = pts.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        y_n = (y - mu_t) / std_t
        pred_n = model(pts)
        loss = criterion(pred_n, y_n)

        bs = pts.size(0)
        total_loss += loss.item() * bs
        total_count += bs

    return total_loss / max(1, total_count)


@torch.no_grad()
def evaluate_denorm_metrics(model, loader, mu, std):
    model.eval()
    y_true_all, y_pred_all = [], []

    mu_t = torch.from_numpy(mu).to(DEVICE)
    std_t = torch.from_numpy(std).to(DEVICE)

    for pts, y, _, _ in loader:
        pts = pts.to(DEVICE, non_blocking=True)
        pred_n = model(pts)
        pred = (pred_n * std_t + mu_t).cpu().numpy()

        y_true_all.append(y.numpy())
        y_pred_all.append(pred)

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)

    per = {}
    for j, name in enumerate(TARGET_COLS):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        per[name] = {
            "RMSE": math.sqrt(mean_squared_error(yt, yp)),
            "MAE": mean_absolute_error(yt, yp),
            "R2": r2_score(yt, yp),
        }

    macro = {
        "RMSE": float(np.sqrt(((y_true - y_pred) ** 2).mean())),
        "MAE": float(np.abs(y_true - y_pred).mean()),
        "R2": float(r2_score(y_true, y_pred, multioutput="variance_weighted")),
    }

    return per, macro


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {DEVICE}")

    train_full = NPZPointParamDataset(TRAIN_SHARDS, TRAIN_CSV)
    test_ds = NPZPointParamDataset(TEST_SHARDS, TEST_CSV)

    train_ds, val_ds = split_by_geom(train_full, val_frac=VAL_FRACTION)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        collate_fn=point_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        collate_fn=point_collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        collate_fn=point_collate_fn,
    )

    mu, std = compute_y_stats(train_ds)
    mu_t = torch.from_numpy(mu).to(DEVICE)
    std_t = torch.from_numpy(std).to(DEVICE)

    np.savez(OUT_DIR / "target_norm_stats.npz", mu=mu, std=std, cols=np.array(TARGET_COLS, dtype=object))

    model = PointTransformerRegressor(
        output_dim=len(TARGET_COLS),
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        heads=HEADS,
        token_count=TOKEN_COUNT,
        mlp_ratio=MLP_RATIO,
        dropout=DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    cfg = {
        "seed": SEED,
        "train_shards": TRAIN_SHARDS,
        "test_shards": TEST_SHARDS,
        "train_csv": TRAIN_CSV,
        "test_csv": TEST_CSV,
        "target_cols": TARGET_COLS,
        "num_points": NUM_POINTS,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "val_fraction": VAL_FRACTION,
        "num_workers": NUM_WORKERS,
        "device": str(DEVICE),
        "model": {
            "embed_dim": EMBED_DIM,
            "depth": DEPTH,
            "heads": HEADS,
            "token_count": TOKEN_COUNT,
            "mlp_ratio": MLP_RATIO,
            "dropout": DROPOUT,
        },
    }
    with open(OUT_DIR / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    best_val = float("inf")
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_mse = train_one_epoch(model, train_loader, optimizer, criterion, mu_t, std_t)
        val_mse = eval_loss(model, val_loader, criterion, mu_t, std_t)

        if val_mse < best_val:
            best_val = val_mse
            torch.save(
                {
                    "model": model.state_dict(),
                    "mu": mu,
                    "std": std,
                    "target_cols": TARGET_COLS,
                    "config": cfg,
                },
                OUT_DIR / "best_pointtransformer.pt",
            )
            tag = " <-- BEST"
        else:
            tag = ""

        print(
            f"[Epoch {epoch:04d}] "
            f"train_mse={train_mse:.6e} val_mse={val_mse:.6e} "
            f"time={(time.time() - t0):.1f}s{tag}"
        )

    torch.save(model.state_dict(), OUT_DIR / "final_pointtransformer_weights.pt")

    per, macro = evaluate_denorm_metrics(model, test_loader, mu, std)
    with open(OUT_DIR / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"per_param": per, "macro": macro}, f, indent=2)

    print("\n=== Test Metrics (denormalized) ===")
    for name in TARGET_COLS:
        m = per[name]
        print(f"{name:>2}: RMSE={m['RMSE']:.6f} MAE={m['MAE']:.6f} R2={m['R2']:.4f}")
    print(f"Macro: RMSE={macro['RMSE']:.6f} MAE={macro['MAE']:.6f} R2={macro['R2']:.4f}")
    print(f"\nTraining complete in {(time.time() - start) / 60.0:.1f} minutes")


if __name__ == "__main__":
    main()
