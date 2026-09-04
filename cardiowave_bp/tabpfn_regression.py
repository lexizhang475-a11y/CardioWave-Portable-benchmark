from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences
from scipy.stats import entropy, kurtosis, skew

from .pipeline import zscore

META_COLUMNS = [
    "age_clean",
    "sex_clean",
    "height_clean",
    "weight_clean",
    "bmi",
    "dx_htn_clean",
    "drug_binary",
    "pre_smoke_clean",
    "pre_coffee_clean",
]


def _finite(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _signal_entropy(x: np.ndarray) -> float:
    x = _finite(x)
    if len(x) < 2:
        return math.nan
    hist, _ = np.histogram(x, bins=32, density=True)
    hist = hist[np.isfinite(hist) & (hist > 0)]
    if len(hist) == 0:
        return math.nan
    return float(entropy(hist))


def _peak_features(x: np.ndarray, fs: int, prefix: str) -> dict[str, float]:
    out = {
        f"{prefix}_peak_count": math.nan,
        f"{prefix}_peak_rate_per_min": math.nan,
        f"{prefix}_peak_amp_mean": math.nan,
        f"{prefix}_peak_amp_std": math.nan,
        f"{prefix}_peak_prom_mean": math.nan,
        f"{prefix}_peak_prom_std": math.nan,
        f"{prefix}_peak_interval_mean": math.nan,
        f"{prefix}_peak_interval_std": math.nan,
    }
    x = _finite(x)
    if len(x) == 0:
        return out
    xz = zscore(x)
    try:
        peaks, _ = find_peaks(xz, distance=max(1, int(0.30 * fs)), prominence=max(float(np.std(xz)) * 0.15, 1e-6))
    except Exception:
        peaks = np.asarray([], dtype=int)
    duration_min = len(x) / float(fs) / 60.0
    out[f"{prefix}_peak_count"] = float(len(peaks))
    out[f"{prefix}_peak_rate_per_min"] = float(len(peaks) / duration_min) if duration_min > 0 else math.nan
    if len(peaks) > 0:
        amps = xz[peaks]
        out[f"{prefix}_peak_amp_mean"] = float(np.mean(amps))
        out[f"{prefix}_peak_amp_std"] = float(np.std(amps))
        try:
            prom = peak_prominences(xz, peaks)[0]
            out[f"{prefix}_peak_prom_mean"] = float(np.mean(prom))
            out[f"{prefix}_peak_prom_std"] = float(np.std(prom))
        except Exception:
            pass
    if len(peaks) > 1:
        intervals = np.diff(peaks) / float(fs)
        out[f"{prefix}_peak_interval_mean"] = float(np.mean(intervals))
        out[f"{prefix}_peak_interval_std"] = float(np.std(intervals))
    return out


def signal_features(x: np.ndarray, fs: int, prefix: str) -> dict[str, float]:
    x = _finite(x)
    keys = [
        "len", "mean", "std", "median", "min", "max", "p2p", "q05", "q25", "q75", "q95",
        "iqr", "rms", "mad", "skew", "kurtosis", "entropy", "diff_mean", "diff_std",
        "diff_p2p", "diff_energy", "diff_abs_mean", "diff_abs_median", "zero_cross_rate", "slope",
    ]
    out = {f"{prefix}_{k}": math.nan for k in keys}
    if len(x) == 0:
        out.update(_peak_features(x, fs, prefix))
        return out
    xz = zscore(x)
    q05, q25, q75, q95 = np.quantile(x, [0.05, 0.25, 0.75, 0.95])
    diff = np.diff(x)
    out.update({
        f"{prefix}_len": float(len(x)),
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_median": float(np.median(x)),
        f"{prefix}_min": float(np.min(x)),
        f"{prefix}_max": float(np.max(x)),
        f"{prefix}_p2p": float(np.ptp(x)),
        f"{prefix}_q05": float(q05),
        f"{prefix}_q25": float(q25),
        f"{prefix}_q75": float(q75),
        f"{prefix}_q95": float(q95),
        f"{prefix}_iqr": float(q75 - q25),
        f"{prefix}_rms": float(np.sqrt(np.mean(x * x))),
        f"{prefix}_mad": float(np.median(np.abs(x - np.median(x)))),
        f"{prefix}_skew": float(skew(x, bias=False)) if np.std(x) > 0 and len(x) > 2 else math.nan,
        f"{prefix}_kurtosis": float(kurtosis(x, fisher=True, bias=False)) if np.std(x) > 0 and len(x) > 3 else math.nan,
        f"{prefix}_entropy": _signal_entropy(x),
        f"{prefix}_zero_cross_rate": float(np.mean(np.diff(np.signbit(xz)) != 0)) if len(xz) > 1 else math.nan,
    })
    if len(diff) > 0:
        out.update({
            f"{prefix}_diff_mean": float(np.mean(diff)),
            f"{prefix}_diff_std": float(np.std(diff)),
            f"{prefix}_diff_p2p": float(np.ptp(diff)),
            f"{prefix}_diff_energy": float(np.mean(diff * diff)),
            f"{prefix}_diff_abs_mean": float(np.mean(np.abs(diff))),
            f"{prefix}_diff_abs_median": float(np.median(np.abs(diff))),
        })
    if len(x) > 1:
        try:
            out[f"{prefix}_slope"] = float(np.polyfit(np.linspace(0.0, 1.0, len(x)), xz, 1)[0])
        except Exception:
            pass
    out.update(_peak_features(x, fs, prefix))
    return out
