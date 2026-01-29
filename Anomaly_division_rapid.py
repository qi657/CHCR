import numpy as np
import pandas as pd
import xarray as xr
import torch
from pathlib import Path


# =========================
# Configuration
# =========================
SST_PATH = Path("E:/ERA5_data/ERA5_monthly_phyVar_1deg/ERA5_monthly_averaged_data_on_single_levels_SST_1940-2024_1deg.nc")
T2M_PATH = Path("E:/ERA5_data/ERA5_daily_t2m_1deg/ERA5_Daily_t2m_combined/ERA5_daily_1deg_combined_detrend.nc")

OUT_DIR = Path("./ckpt/flip")
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "label_anomaly").mkdir(parents=True, exist_ok=True)

# Whiplash detection hyper-parameters
SMOOTH_K = 5        # k-day centered moving mean
CLIM_WIN = 31       # seasonal climatology window (DOY ± 15)
TAU = 5             # max gap days between adjacent extreme segments
MINLEN = 5          # minimum duration of warm/cold segment
USE_QUANT = True    # True: percentile thresholds; False: z-score thresholds
Q_WARM = 0.85       # warm-side quantile
Q_COLD = 0.15       # cold-side quantile
Z_THR = 1.0         # z-score threshold if USE_QUANT=False

# Use a lat band to exclude polar regions when building land mask from SST
POLAR_LAT_ABS = 60


# =========================
# Utilities: masks & loading
# =========================
def build_land_mask_from_sst(
    sst_path: Path,
    sst_var: str = "sst",
    lat_name: str = "lat",
    lon_name: str = "lon",
    polar_lat_abs: float = 60.0,
    n_time: int | None = 1008,
) -> xr.DataArray:
    """
    Build a land mask (1=land, 0=non-land) using SST NaNs:
    - SST is typically NaN over land.
    - To avoid polar missing values contaminating the land mask,
      we force polar-band values to non-NaN before detecting NaNs.

    Returns
    -------
    land_mask : xr.DataArray, dims (lat, lon), dtype int8, values in {0,1}
    """
    ds = xr.open_dataset(sst_path)
    sst = ds[sst_var]
    if n_time is not None:
        sst = sst.isel(time=slice(0, n_time)) if "time" in sst.dims else sst.isel({sst.dims[0]: slice(0, n_time)})

    lat = sst[lat_name]
    polar_mask_1d = (lat < -polar_lat_abs) | (lat > polar_lat_abs)

    # Work on one snapshot to infer NaN pattern; fix polar band first
    sst0 = sst.isel({sst.dims[0]: 0}).copy()
    sst0 = sst0.where(~polar_mask_1d, other=0.0)

    land_mask = xr.where(np.isnan(sst0), 1, 0).astype("int8")
    land_mask.name = "land_mask"
    return land_mask


def load_t2m_celsius(ds_path: Path, var: str = "t2m") -> xr.DataArray:
    """
    Load daily 2m temperature and convert to Celsius.
    """
    ds = xr.open_dataset(ds_path)
    t2m = ds[var] - 273.15
    t2m.name = "t2m_c"
    return t2m


def canonicalize_year_day_dims(t2m: xr.DataArray) -> xr.DataArray:
    """
    Ensure t2m has dims ('year','day','lat','lon').

    Supported cases:
    - Already has ('year','day',...)
    - Has ('year','valid_time',...)  -> rename valid_time to day
    - Other layouts are not handled here (you can extend if needed).
    """
    dims = list(t2m.dims)

    if "year" in dims and "day" in dims:
        day_dim = "day"
    elif "year" in dims and "valid_time" in dims:
        t2m = t2m.rename({"valid_time": "day"})
        day_dim = "day"
    else:
        raise ValueError(
            f"Unsupported t2m dims: {t2m.dims}. "
            "Expected ('year','day',lat,lon) or ('year','valid_time',lat,lon)."
        )

    # Try to infer lat/lon dim names
    lat_dim = "lat" if "lat" in t2m.dims else ("latitude" if "latitude" in t2m.dims else None)
    lon_dim = "lon" if "lon" in t2m.dims else ("longitude" if "longitude" in t2m.dims else None)
    if lat_dim is None or lon_dim is None:
        raise ValueError(f"Cannot infer lat/lon dim names from: {t2m.dims}")

    t2m = t2m.transpose("year", day_dim, lat_dim, lon_dim)
    t2m = t2m.rename({day_dim: "day", lat_dim: "lat", lon_dim: "lon"})
    return t2m


def apply_land_mask(t2m_yday: xr.DataArray, land_mask: xr.DataArray) -> xr.DataArray:
    """
    Apply land mask to t2m, returning land-only values (NaN over non-land).
    """
    # Align coords if needed
    land_mask = land_mask.rename({d: d for d in land_mask.dims})  # no-op, keeps explicit
    land_mask = land_mask.sel(lat=t2m_yday["lat"], lon=t2m_yday["lon"], method="nearest")
    return t2m_yday.where(land_mask == 1)


# =========================
# Core algorithm
# =========================
def moving_mean_wrap_ignore_nan(arr_yd: np.ndarray, k: int) -> np.ndarray:
    """
    Centered k-day moving mean along day axis with wrap-around, ignoring NaNs.
    Input shape: (Y, D)
    """
    if k <= 1:
        return arr_yd.astype(np.float32)

    y, d = arr_yd.shape
    p = k // 2
    a = arr_yd.astype(np.float32)

    valid = np.isfinite(a).astype(np.float32)
    a = np.nan_to_num(a, nan=0.0)

    num = np.zeros_like(a, dtype=np.float32)
    den = np.zeros_like(valid, dtype=np.float32)
    for off in range(-p, p + 1):
        num += np.roll(a, shift=off, axis=1)
        den += np.roll(valid, shift=off, axis=1)

    out = num / np.maximum(den, 1e-6)
    out[den < 1] = np.nan
    return out


def runs_1d(flags: np.ndarray) -> np.ndarray:
    """
    Return contiguous True runs as inclusive intervals [start, end], shape (K, 2).
    """
    x = np.asarray(flags, dtype=bool)
    if x.size == 0:
        return np.empty((0, 2), dtype=np.int32)

    pad = np.concatenate(([False], x, [False]))
    d = np.diff(pad.view(np.int8))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0] - 1

    if starts.size == 0:
        return np.empty((0, 2), dtype=np.int32)

    return np.stack([starts, ends], axis=1).astype(np.int32)


def select_runs_minlen(runs: np.ndarray, minlen: int) -> np.ndarray:
    """
    Keep runs whose length >= minlen.
    """
    if runs.size == 0:
        return runs
    keep = (runs[:, 1] - runs[:, 0] + 1) >= minlen
    return runs[keep]


def pair_adjacent_whiplash_mark(
    year_warm: np.ndarray,
    year_cold: np.ndarray,
    tau: int,
    minlen: int,
    d: int,
) -> tuple[int, np.ndarray, np.ndarray]:
    """
    Check whiplash within a single year:
    - Consider only adjacent extreme segments (warm/cold).
    - Adjacent segments must have opposite sign and gap <= tau.
    - Each segment length must be >= minlen.

    Returns
    -------
    has_event : int (0/1)
    hot_mask  : (D,) boolean mask of warm-segment days involved in whiplash
    cold_mask : (D,) boolean mask of cold-segment days involved in whiplash
    """
    warm_runs = select_runs_minlen(runs_1d(year_warm), minlen)
    cold_runs = select_runs_minlen(runs_1d(year_cold), minlen)

    if warm_runs.size == 0 or cold_runs.size == 0:
        return 0, np.zeros(d, dtype=bool), np.zeros(d, dtype=bool)

    segs: list[tuple[int, int, int]] = []
    for s, e in warm_runs:
        segs.append((s, e, +1))
    for s, e in cold_runs:
        segs.append((s, e, -1))
    segs.sort(key=lambda x: x[0])

    hot_mask = np.zeros(d, dtype=bool)
    cold_mask = np.zeros(d, dtype=bool)
    has = 0

    for i in range(len(segs) - 1):
        s1, e1, sg1 = segs[i]
        s2, e2, sg2 = segs[i + 1]
        if sg1 == sg2:
            continue
        gap = s2 - e1 - 1
        if gap <= tau:
            has = 1
            if sg1 == +1 and sg2 == -1:
                hot_mask[s1 : e1 + 1] = True
                cold_mask[s2 : e2 + 1] = True
            else:  # sg1 == -1 and sg2 == +1
                cold_mask[s1 : e1 + 1] = True
                hot_mask[s2 : e2 + 1] = True

    return has, hot_mask, cold_mask


def whiplash_5day_segments_point(
    arr_yd: np.ndarray,
    smooth_k: int,
    clim_win: int,
    tau: int,
    minlen: int,
    use_quant: bool,
    q_warm: float,
    q_cold: float,
    z_thr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute whiplash for a single grid point.

    Parameters
    ----------
    arr_yd : (Y, D) array
        Daily temperature (Celsius), NaN over non-land.

    Returns
    -------
    yr_has   : (Y,) int8, 1 if year has >=1 whiplash event else 0
    hot_mask : (Y, D) int8, warm-segment days involved in whiplash
    cold_mask: (Y, D) int8, cold-segment days involved in whiplash
    """
    if not np.isfinite(arr_yd).any():
        y, d = arr_yd.shape
        return np.zeros(y, np.int8), np.zeros((y, d), np.int8), np.zeros((y, d), np.int8)

    y, d = arr_yd.shape

    # 1) k-day moving mean with wrap-around
    sm = moving_mean_wrap_ignore_nan(arr_yd, smooth_k)

    # 2) Seasonal baseline using a +/- (clim_win//2) day window, pooled across years
    half = clim_win // 2
    offsets = np.arange(-half, half + 1, dtype=int)
    samp = sm[:, (np.arange(d)[:, None] + offsets[None, :]) % d]  # (Y, D, clim_win)

    if use_quant:
        high_q = np.nanpercentile(samp, q_warm * 100.0, axis=(0, 2))  # (D,)
        low_q = np.nanpercentile(samp, q_cold * 100.0, axis=(0, 2))   # (D,)
        warm = sm > high_q[None, :]
        cold = sm < low_q[None, :]
    else:
        clim_mu = np.nanmean(samp, axis=(0, 2))
        clim_sd = np.nanstd(samp, axis=(0, 2), ddof=1)
        clim_sd[clim_sd < 1e-6] = np.nan
        z = (sm - clim_mu[None, :]) / clim_sd[None, :]
        warm = z > +z_thr
        cold = z < -z_thr

    warm &= np.isfinite(sm)
    cold &= np.isfinite(sm)

    # 3) Year-by-year whiplash marking
    yr_has = np.zeros(y, np.int8)
    hot_mask = np.zeros((y, d), np.int8)
    cold_mask = np.zeros((y, d), np.int8)

    for yi in range(y):
        has, hm, cm = pair_adjacent_whiplash_mark(warm[yi], cold[yi], tau, minlen, d)
        yr_has[yi] = has
        hot_mask[yi] = hm.astype(np.int8)
        cold_mask[yi] = cm.astype(np.int8)

    return yr_has, hot_mask, cold_mask


# =========================
# Main workflow
# =========================
def main() -> None:
    # 1) Land mask from SST NaNs
    land_mask = build_land_mask_from_sst(
        SST_PATH,
        sst_var="sst",
        lat_name="lat",
        lon_name="lon",
        polar_lat_abs=POLAR_LAT_ABS,
        n_time=1008,
    )

    # 2) Load t2m and canonicalize dims
    t2m = load_t2m_celsius(T2M_PATH, var="t2m")
    t2m = canonicalize_year_day_dims(t2m)

    # 3) Apply land mask (NaN over non-land)
    t2m_land = apply_land_mask(t2m, land_mask).astype("float32")
    t2m_land.name = "t2m_land"

    # Optional: dask chunking for large arrays
    # t2m_land = t2m_land.chunk({"year": -1, "day": -1, "lat": 30, "lon": 30})

    # 4) Vectorize across (lat, lon) with apply_ufunc
    yr_has_whiplash, hot_mask_seg, cold_mask_seg = xr.apply_ufunc(
        whiplash_5day_segments_point,
        t2m_land,
        input_core_dims=[["year", "day"]],
        output_core_dims=[["year"], ["year", "day"], ["year", "day"]],
        vectorize=True,
        dask="parallelized",
        kwargs=dict(
            smooth_k=SMOOTH_K,
            clim_win=CLIM_WIN,
            tau=TAU,
            minlen=MINLEN,
            use_quant=USE_QUANT,
            q_warm=Q_WARM,
            q_cold=Q_COLD,
            z_thr=Z_THR,
        ),
        output_dtypes=[np.int8, np.int8, np.int8],
        keep_attrs=False,
    )

    # 5) Save NetCDF outputs
    yr_has_path = OUT_DIR / "yr_has_whiplash_90_10_5dSmooth_31dClim_tau5_len5.nc"
    hot_path = OUT_DIR / "hot_mask_90_10_segInFlip_tau5_len5.nc"
    cold_path = OUT_DIR / "cold_mask_90_10_segInFlip_tau5_len5.nc"

    yr_has_whiplash.to_netcdf(yr_has_path)
    hot_mask_seg.to_netcdf(hot_path)
    cold_mask_seg.to_netcdf(cold_path)

    # 6) Build sparse index table for label==1 over land
    # Ensure dim order: (year, lat, lon)
    yr_has_yll = yr_has_whiplash.transpose("year", "lat", "lon")
    label = yr_has_yll.values.astype(np.int8)  # shape (Y, lat, lon)

    land2d = land_mask.sel(lat=yr_has_yll["lat"], lon=yr_has_yll["lon"]).values.astype(np.int8)  # (lat, lon)
    land3d = land2d[None, :, :]  # (1, lat, lon)

    y_idx, lat_idx, lon_idx = np.where((label == 1) & (land3d == 1))

    df_label_1 = pd.DataFrame(
        {
            "year_idx": y_idx.astype(np.int32),
            "lat_idx": lat_idx.astype(np.int32),
            "lon_idx": lon_idx.astype(np.int32),
            "label": np.ones_like(y_idx, dtype=np.int8),
        }
    )

    # Save as torch + (optional) CSV for readability
    torch.save(df_label_1, OUT_DIR / "label_anomaly" / "label_tensor_90_10.pth")

if __name__ == "__main__":
    main()
