# -- coding: utf-8 --
# @Time   : 2026/1/27 10:07
# @Author : Stephanie
# @Email  : sunc696@gmail.com
# @File   : inference.py

from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import xarray as xr
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    log_loss,
)

TEST_DATA_DIR = Path("./ckpt/flip/normalization_data_anomaly_90_10/test_data")
TEST_LABEL_PATH = Path("./ckpt/flip/label_anomaly/test_label_tensor_90_10.pth")

SST_PATH = Path(
    "E:/ERA5_data/ERA5_monthly_phyVar_1deg/"
    "ERA5_monthly_averaged_data_on_single_levels_SST_1940-2024_1deg.nc"
)

FIG_DIR = Path("./fig/paper")
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATHS = {
    "Original": Path("ckpt/model_original.pth"),
    "Forget ENSO": Path("ckpt/model_unlearned_ENSO.pth"),
    "Forget NAO": Path("ckpt/model_unlearned_NAO.pth"),
    "Forget IOD": Path("ckpt/model_unlearned_IOD.pth"),
}

BATCH_SIZE = 12
INPUT_CHANNELS = 87  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Dataset
# =========================
class CustomDataset(Dataset):
    """
    Dataset that returns (label, time_idx, coord_ij).
    Assumes label_data is a DataFrame with columns:
    ['time_idx', 'lat_idx', 'lon_idx', 'label'].
    """

    def __init__(self, label_data: pd.DataFrame):
        self.label_data = label_data.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.label_data)

    def __getitem__(self, idx: int):
        row = self.label_data.iloc[idx]
        time_idx = int(row["time_idx"])
        lat_idx = int(row["lat_idx"])
        lon_idx = int(row["lon_idx"])
        y = int(row["label"])
        coord_ij = torch.tensor((lat_idx, lon_idx), dtype=torch.long)
        return y, time_idx, coord_ij


# =========================
# Helpers
# =========================
def get_lag_index(input_channels: int) -> list[int]:
    """
    Map input channel count to lag indices.
    - 75  -> current only
    - 87  -> (t-1, t)
    - 99  -> (t-2, t-1, t)
    """
    if input_channels == 75:
        return [0]
    if input_channels == 87:
        return [1, 0]
    if input_channels == 99:
        return [2, 1, 0]
    raise ValueError(f"Unsupported input_channels={input_channels}, expected 75/87/99.")


def binary_entropy_from_prob(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Binary entropy H(p) in nats.
    """
    p = np.clip(p, eps, 1 - eps)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def set_tight_ylim(ax, ys_list, pad_ratio: float = 0.08, min_pad: float = 1e-3) -> None:
    """
    Auto-scale y-limits based on curve values to avoid overly wide default ranges.
    """
    vals = np.concatenate([np.asarray(y, float) for y in ys_list])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return
    y_min, y_max = float(vals.min()), float(vals.max())
    span = max(y_max - y_min, min_pad)
    pad = max(span * pad_ratio, min_pad)
    ax.set_ylim(y_min - pad, y_max + pad)


@torch.no_grad()
def infer_one_model(
    model,
    test_loader: DataLoader,
    test_data_dir: Path,
    input_channels: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference for one model.

    Returns
    -------
    y_true : (N,) int
    y_prob : (N,) float
    t_idx  : (N,) int  (0 corresponds to 1940)
    ij     : (N,2) int (lat_idx, lon_idx)
    """
    model.eval()

    all_labels, all_probs = [], []
    all_time, all_coords = [], []

    input_feature_cache: Dict[int, torch.Tensor] = {}
    lag_index = get_lag_index(input_channels)

    for labels, time_idxs, coords in test_loader:
        labels = labels.to(device).float().view(-1)           # (B,)
        coords = coords.clone().detach().long().to(device)    # (B,2)
        B = labels.size(0)

        inputs_list = []
        for b in range(B):
            time_idx = int(time_idxs[b])
            features_for_sample = []

            for lag in lag_index:
                cur_time_idx = time_idx - lag
                if cur_time_idx < 0:
                    cur_time_idx = time_idx

                if cur_time_idx not in input_feature_cache:
                    year = 1940 + cur_time_idx
                    fpath = test_data_dir / f"input_feature_{year}.pth"
                    input_features = torch.load(fpath, map_location=device)
                    input_feature_cache[cur_time_idx] = input_features
                else:
                    input_features = input_feature_cache[cur_time_idx]

                # For lagged features in your pipeline, only take the first 12 channels
                if lag in (2, 1):
                    features_for_sample.append(input_features[:12, :, :])
                else:
                    features_for_sample.append(input_features)

            inputs_list.append(torch.cat(features_for_sample, dim=0))

        inputs = torch.stack(inputs_list, dim=0).to(device)  # (B,C,H,W)

        lat_idx = coords[:, 0].long()
        lon_idx = coords[:, 1].long()

        # Extract lat/lon values embedded in the last 3 channels (per your data format)
        lat_value = inputs[torch.arange(B), input_channels - 3, lat_idx, lon_idx]
        lon_value = inputs[torch.arange(B), input_channels - 2, lat_idx, lon_idx]
        coords_value = torch.stack([lat_value, lon_value], dim=1)

        coords_ij = torch.stack([lat_idx, lon_idx], dim=1)  # (B,2)

        outputs = model(inputs, coords_ij, coords_value).view(-1)  # (B,)
        probs = outputs.detach().float().cpu().numpy()

        all_labels.append(labels.detach().cpu().numpy().astype(int))
        all_probs.append(probs)
        all_time.append(time_idxs.detach().cpu().numpy().astype(int))
        all_coords.append(coords_ij.detach().cpu().numpy().astype(int))

    y_true = np.concatenate(all_labels, axis=0).astype(int)
    y_prob = np.concatenate(all_probs, axis=0).astype(float)
    t_idx = np.concatenate(all_time, axis=0).astype(int)       # year index (0 -> 1940)
    ij = np.concatenate(all_coords, axis=0).astype(int)        # (N,2)

    return y_true, y_prob, t_idx, ij


@torch.no_grad()
def infer_one_model_mc_dropout(
    model,
    test_loader: DataLoader,
    test_data_dir: Path,
    input_channels: int,
    device: str,
    T: int = 30,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()

    # Enable dropout layers
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

    torch.manual_seed(seed)
    np.random.seed(seed)

    probs_runs = []
    y_true = t_idx = ij = None

    for t in range(T):
        yt, yp, ti, coords = infer_one_model(
            model=model,
            test_loader=test_loader,
            test_data_dir=test_data_dir,
            input_channels=input_channels,
            device=device,
        )
        probs_runs.append(yp.astype(float))
        if t == 0:
            y_true, t_idx, ij = yt, ti, coords

    probs_mc = np.stack(probs_runs, axis=0)  # (T, N)
    probs_mean = probs_mc.mean(axis=0)
    probs_var = probs_mc.var(axis=0)

    model.eval()
    return y_true, probs_mean, probs_var, t_idx, ij


def compute_metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> Dict[str, float]:
    """
    Compute standard binary classification metrics at a fixed threshold.
    """
    y_pred = (y_prob > thr).astype(int)
    return {
        "acc": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "logloss": log_loss(y_true, y_prob, labels=[0, 1]),
        "auc": roc_auc_score(y_true, y_prob),
    }


def plot_metrics_vs_sample_size(
    results: Dict[str, Dict[str, np.ndarray]],
    thr: float = 0.5,
    step: int = 500,
    sort_mode: str = "time",
    seed: int = 0,
    save_path: Path | None = None,
) -> None:
    """
    Plot ACC / Recall / AUC / Entropy curves vs. sample size.

    sort_mode:
      - "time": sort by t_idx ascending
      - "prob": sort by probability descending
      - "random": random permutation
    """
    color_map = {
        "Original": "#BF6BA2",
        "Forget ENSO": "#FFC182",
        "Forget IOD": "#5F97D0",
        "Forget NAO": "#2CA02C",
    }

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.grid": True,
    })

    rng = np.random.default_rng(seed)
    names = list(results.keys())

    fig = plt.figure(figsize=(14, 3.8), dpi=200)
    ax_acc = fig.add_subplot(1, 4, 1)
    ax_rec = fig.add_subplot(1, 4, 2)
    ax_auc = fig.add_subplot(1, 4, 3)
    ax_ent = fig.add_subplot(1, 4, 4)

    acc_curves, rec_curves, auc_curves, ent_curves = [], [], [], []

    for name in names:
        y_true = results[name]["y_true"].astype(int)
        y_prob = results[name]["y_prob"].astype(float)
        t_idx = results[name]["t_idx"].astype(int)

        if sort_mode == "time":
            order = np.argsort(t_idx)
        elif sort_mode == "prob":
            order = np.argsort(-y_prob)
        elif sort_mode == "random":
            order = rng.permutation(len(y_true))
        else:
            raise ValueError("sort_mode must be one of: 'time', 'prob', 'random'")

        yt = y_true[order]
        yp = y_prob[order]

        Ns, accs, recs, aucs, ents = [], [], [], [], []

        N_list = list(range(step, len(yt) + 1, step))
        if not N_list or N_list[-1] != len(yt):
            N_list.append(len(yt))

        for N in N_list:
            w_true = yt[:N]
            w_prob = yp[:N]
            w_pred = (w_prob > thr).astype(int)

            Ns.append(N)
            accs.append(accuracy_score(w_true, w_pred))
            recs.append(recall_score(w_true, w_pred, zero_division=0))

            try:
                aucs.append(roc_auc_score(w_true, w_prob))
            except ValueError:
                aucs.append(np.nan)

            ents.append(np.nanmean(binary_entropy_from_prob(w_prob)))

        acc_curves.append(np.asarray(accs, float))
        rec_curves.append(np.asarray(recs, float))
        auc_curves.append(np.asarray(aucs, float))
        ent_curves.append(np.asarray(ents, float))

        c = color_map.get(name, None)
        ax_acc.plot(Ns, accs, lw=2.2, color=c, label=name)
        ax_rec.plot(Ns, recs, lw=2.2, color=c, label=name)
        ax_auc.plot(Ns, aucs, lw=2.2, color=c, label=name)
        ax_ent.plot(Ns, ents, lw=2.2, color=c, label=name)

    for ax, ylab in [(ax_acc, "Accuracy"), (ax_rec, "Recall"), (ax_auc, "AUC"), (ax_ent, "Entropy")]:
        ax.set_xlabel("Sample size")
        ax.set_ylabel(ylab)
        ax.legend(loc="best", frameon=True)

    set_tight_ylim(ax_acc, acc_curves, pad_ratio=0.08)
    set_tight_ylim(ax_rec, rec_curves, pad_ratio=0.08)
    set_tight_ylim(ax_auc, auc_curves, pad_ratio=0.08)
    set_tight_ylim(ax_ent, ent_curves, pad_ratio=0.10)

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def evaluate_all_models(
    model_paths: Dict[str, Path],
    test_loader: DataLoader,
    test_data_dir: Path,
    input_channels: int,
    device: str,
    use_mc: bool = False,
    T: int = 30,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Evaluate all models and return a dict of inference outputs.
    """
    results: Dict[str, Dict[str, np.ndarray]] = {}

    for name, path in model_paths.items():
        model = torch.load(path, map_location=device)
        model.eval()

        if use_mc:
            y_true, y_prob_mean, _y_prob_var, t_idx, ij = infer_one_model_mc_dropout(
                model=model,
                test_loader=test_loader,
                test_data_dir=test_data_dir,
                input_channels=input_channels,
                device=device,
                T=T,
                seed=0,
            )
            results[name] = {"y_true": y_true, "y_prob": y_prob_mean, "t_idx": t_idx, "ij": ij}
        else:
            y_true, y_prob, t_idx, ij = infer_one_model(
                model=model,
                test_loader=test_loader,
                test_data_dir=test_data_dir,
                input_channels=input_channels,
                device=device,
            )
            results[name] = {"y_true": y_true, "y_prob": y_prob, "t_idx": t_idx, "ij": ij}

        ent_mean = binary_entropy_from_prob(results[name]["y_prob"]).mean()
        m = compute_metrics_at_threshold(results[name]["y_true"], results[name]["y_prob"], thr=0.5)

        print(f"\n===== {name} =====")
        print(f"Accuracy : {m['acc']:.4f}")
        print(f"Recall   : {m['recall']:.4f}")
        print(f"AUC      : {m['auc']:.4f}")
        print(f"Entropy  : {ent_mean:.4f} (mean, nats)")

    return results


def main() -> None:
    # Load label dataframe
    label_df = torch.load(TEST_LABEL_PATH)
    if not isinstance(label_df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame saved by torch.save().")

    # Load lat/lon arrays (kept only if you need them downstream; safe to remove if unused)
    _ = xr.open_dataset(SST_PATH)
    # lat_values = _["lat"].values
    # lon_values = _["lon"].values

    test_dataset = CustomDataset(label_df)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    results = evaluate_all_models(
        model_paths=MODEL_PATHS,
        test_loader=test_loader,
        test_data_dir=TEST_DATA_DIR,
        input_channels=INPUT_CHANNELS,
        device=DEVICE,
        use_mc=False,
        T=30,
    )

    fig_path = FIG_DIR / "model_performance.png"
    plot_metrics_vs_sample_size(
        results,
        thr=0.5,
        step=500,
        sort_mode="time",
        seed=0,
        save_path=fig_path,
    )


if __name__ == "__main__":
    main()
