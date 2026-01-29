# -- coding: utf-8 --
# @Time   : 2026/1/27 10:07
# @Author : Stephanie
# @Email  : sunc696@gmail.com
# @File   : inference.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    roc_curve, auc, roc_auc_score, log_loss
)
import xarray as xr
from sklearn.metrics import recall_score  # (Duplicate import; already imported above)
from torch.utils.data import Dataset, DataLoader, TensorDataset


test_data_dir = './ckpt/flip/normalization_data_anomaly_90_10/test_data'
label = torch.load('./ckpt/flip/label_anomaly/test_label_tensor_90_10.pth')

data_SST = xr.open_dataset("E:/ERA5_data/ERA5_monthly_phyVar_1deg/ERA5_monthly_averaged_data_on_single_levels_SST_1940-2024_1deg.nc")
lat_values = data_SST['lat'].values
lon_values = data_SST['lon'].values


class CustomDataset(Dataset):
    def __init__(self, label_data):
        self.label_data = label_data.reset_index(drop=True)

    def __len__(self):
        return len(self.label_data)

    def __getitem__(self, idx):
        label_info = self.label_data.iloc[idx]
        time_idx = label_info['time_idx']
        lat_idx = label_info['lat_idx']
        lon_idx = label_info['lon_idx']
        label = label_info['label']
        coord = (lat_idx, lon_idx)
        coord = torch.tensor(coord, dtype=torch.long)
        return label, time_idx, coord


batch_size = 12
test_dataset = CustomDataset(label)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
test_len = len(test_loader.dataset)
num = test_len // 2


# =========================
# 1) Configuration: paths to four models
# =========================
MODEL_PATHS = {
    "Original": "ckpt/model_original.pth",
    "Forget ENSO": "ckpt/model_unlearned_ENSO.pth",
    "Forget NAO": "ckpt/model_unlearned_NAO.pth",
    "Forget IOD": "ckpt/model_unlearned_IOD.pth",
}

device = "cuda" if torch.cuda.is_available() else "cpu"
input_channels = 87  
# NOTE: `test_loader` and `test_data_dir` are assumed to be defined above.
# =========================


def _get_lag_index(input_channels: int):
    if input_channels == 75:
        return [0]
    elif input_channels == 87:
        return [1, 0]
    elif input_channels == 99:
        return [2, 1, 0]
    else:
        raise ValueError(f"Unsupported input_channels={input_channels}, expected 75/87/99.")


def binary_entropy_from_prob(p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def _set_tight_ylim(ax, ys_list, pad_ratio=0.08, min_pad=1e-3):
    """
    Auto-scale y-limits based on curve values to avoid an overly wide 0-1 range.

    Parameters
    ----------
    ys_list : list of 1D arrays/lists
        Each item may contain NaNs.
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
def infer_one_model(model, test_loader, test_data_dir, input_channels: int, device: str):
    model.eval()

    all_labels, all_probs = [], []
    all_time, all_coords = [], []

    input_feature_cache = {}
    lag_index = _get_lag_index(input_channels)

    for (labels, time_idxs, coords) in test_loader:
        labels = labels.to(device).float().view(-1)         # (B,)
        coords = coords.clone().detach().long().to(device)  # (B,2)
        B = labels.size(0)

        inputs_list = []
        for b in range(B):
            time_idx = int(time_idxs[b].item())
            features_for_sample = []

            for lag in lag_index:
                cur_time_idx = time_idx - lag
                if cur_time_idx < 0:
                    cur_time_idx = time_idx

                if cur_time_idx not in input_feature_cache:
                    year = 1940 + cur_time_idx
                    input_feature_file = f"{test_data_dir}/input_feature_{year}.pth"
                    input_features = torch.load(input_feature_file, map_location=device)
                    input_feature_cache[cur_time_idx] = input_features
                else:
                    input_features = input_feature_cache[cur_time_idx]

                if lag in [2, 1]:
                    features_for_sample.append(input_features[:12, :, :])
                else:
                    features_for_sample.append(input_features)

            inputs_list.append(torch.cat(features_for_sample, dim=0))

        inputs = torch.stack(inputs_list, dim=0).to(device)  # (B,C,H,W)

        lat_idx = coords[:, 0].long()
        lon_idx = coords[:, 1].long()

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
    t_idx = np.concatenate(all_time, axis=0).astype(int)    # Year index (0 corresponds to 1940)
    ij = np.concatenate(all_coords, axis=0).astype(int)     # (N,2)

    return y_true, y_prob, t_idx, ij


@torch.no_grad()
def infer_one_model_mc_dropout(
    model,
    test_loader,
    test_data_dir,
    input_channels: int,
    device: str,
    T=30,
    seed=0
):
    """
    MC-dropout inference.

    Returns
    -------
    mean probability (and t_idx / ij / y_true).

    Notes
    -----
    - Dropout layers are set to train() during inference.
    - Gradients are disabled via @torch.no_grad().
    """
    # Preserve current mode
    model.eval()

    # Enable dropout layers
    drop_layers = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    for d in drop_layers:
        d.train()

    # Optional: set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    probs_runs = []
    y_true = None
    t_idx = None
    ij = None

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
            y_true = yt
            t_idx = ti
            ij = coords

    probs_mc = np.stack(probs_runs, axis=0)   # (T, N)
    probs_mean = probs_mc.mean(axis=0)        # (N,)
    probs_var = probs_mc.var(axis=0)          # (N,)

    # Restore eval mode (for cleanliness)
    model.eval()
    return y_true, probs_mean, probs_var, t_idx, ij


def compute_metrics_at_threshold(y_true, y_prob, thr: float):
    y_pred = (y_prob > thr).astype(int)
    return {
        "acc": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        # log_loss expects probabilities; labels=[0,1] makes it robust
        "logloss": log_loss(y_true, y_prob, labels=[0, 1]),
        "auc": roc_auc_score(y_true, y_prob),
    }


def plot_acc_recall_auc_entropy_vs_sample_size(
    results_dict,
    thr=0.5,
    step=500,
    sort_mode="time",
    seed=0,
):
    color_map = {
        "Original": "#BF6BA2",
        "Forget ENSO": "#FFC182",
        "Forget IOD": "#5F97D0",
        "Forget NAO": "#2CA02C",
    }

    plt.rcParams.update({
        "axes.facecolor": "#EAEAF2",   # light-gray background (seaborn-like)
        "figure.facecolor": "white",
        "axes.edgecolor": "white",
        "grid.color": "white",
        "grid.linestyle": "-",
        "grid.linewidth": 1.2,
        "axes.grid": True,
    })

    names = list(results_dict.keys())
    rng = np.random.default_rng(seed)

    fig = plt.figure(figsize=(14, 3.8), dpi=200)
    ax_acc = fig.add_subplot(1, 4, 1)
    ax_rec = fig.add_subplot(1, 4, 2)
    ax_auc = fig.add_subplot(1, 4, 3)
    ax_ent = fig.add_subplot(1, 4, 4)

    # Collect curves for auto y-limit scaling
    acc_curves, rec_curves, auc_curves, ent_curves = [], [], [], []

    for name in names:
        y_true = results_dict[name]["y_true"].astype(int)
        y_prob = results_dict[name]["y_prob"].astype(float)
        t_idx = results_dict[name]["t_idx"].astype(int)

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

        Ns = []
        accs, recs, aucs, ents = [], [], [], []

        N_list = list(range(step, len(yt) + 1, step))
        if len(N_list) == 0 or N_list[-1] != len(yt):
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

        # Store for auto y-limit scaling
        acc_curves.append(np.asarray(accs, float))
        rec_curves.append(np.asarray(recs, float))
        auc_curves.append(np.asarray(aucs, float))
        ent_curves.append(np.asarray(ents, float))

        # Use a fixed color if provided; otherwise let Matplotlib decide
        c = color_map.get(name, None)
        ax_acc.plot(Ns, accs, lw=2.2, color=c, label=name)
        ax_rec.plot(Ns, recs, lw=2.2, color=c, label=name)
        ax_auc.plot(Ns, aucs, lw=2.2, color=c, label=name)
        ax_ent.plot(Ns, ents, lw=2.2, color=c, label=name)

    # Titles and axis labels
    for ax, ylab in [(ax_acc, "Accuracy"), (ax_rec, "Recall"), (ax_auc, "AUC")]:
        ax.set_xlabel("Sample size")
        ax.set_ylabel(ylab)
        ax.grid(True, linestyle="--", linewidth=1.2)
        ax.legend(loc="best", frameon=True)

    ax_ent.set_xlabel("Sample size")
    ax_ent.set_ylabel("Entropy")
    ax_ent.grid(True, linestyle="--", linewidth=0.5)
    ax_ent.legend(loc="best", frameon=True)

    # Auto y-limit scaling (ACC/Recall/AUC/Entropy)
    _set_tight_ylim(ax_acc, acc_curves, pad_ratio=0.08)
    _set_tight_ylim(ax_rec, rec_curves, pad_ratio=0.08)
    _set_tight_ylim(ax_auc, auc_curves, pad_ratio=0.08)
    _set_tight_ylim(ax_ent, ent_curves, pad_ratio=0.10)

    plt.tight_layout()
    plt.savefig('./fig/paper/model performance.png', dpi=300, bbox_inches='tight')
    plt.show()


def evaluate_all_models(use_mc=False, T=30):
    results = {}

    for name, path in MODEL_PATHS.items():
        model = torch.load(path, map_location=device)
        model.eval()

        if use_mc:
            y_true, y_prob_mean, y_prob_var, t_idx, ij = infer_one_model_mc_dropout(
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

        # Print overall mean entropy (optional)
        ent_mean = binary_entropy_from_prob(results[name]["y_prob"]).mean()
        m = compute_metrics_at_threshold(results[name]["y_true"], results[name]["y_prob"], thr=0.5)
        print(f"\n===== {name} =====")
        print(f"Accuracy : {m['acc']:.4f}")
        print(f"Recall   : {m['recall']:.4f}")
        print(f"AUC      : {m['auc']:.4f}")
        print(f"Entropy  : {ent_mean:.4f} (mean, nats)")

    # Plot: x = sample size, y = ACC / Recall / AUC / Entropy
    plot_acc_recall_auc_entropy_vs_sample_size(results, thr=0.5, step=500, sort_mode="time")

    return results


# Run
all_results = evaluate_all_models()
