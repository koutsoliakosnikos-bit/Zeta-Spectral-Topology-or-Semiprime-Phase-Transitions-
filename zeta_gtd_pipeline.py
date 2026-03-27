"""
RESEARCH-GRADE v2
=================

Βελτιωμένη έκδοση του pipeline με:

1. Deterministic resume / checkpointing
2. Σωστό separation:
   - feature extraction
   - dataset assembly
   - model training
   - evaluation
3. Block-level caching
4. Multi-seed evaluation
5. Multiple OOD splits
6. Classical ablation suite
7. Deep model showdown
8. Structured experiment logging
9. Error-safe multiprocessing
10. Deterministic dataset ordering

Απαιτούμενα:
pip install numpy scipy scikit-learn matplotlib ripser torch
"""

import csv
import json
import math
import os
import pickle
import time
import hashlib
import urllib.request
import warnings
import multiprocessing as mp
from dataclasses import dataclass, asdict, field
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from ripser import ripser
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    adjusted_rand_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.tree import export_text

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# =========================================================
# 0. PATHS / GLOBALS
# =========================================================
OUTPUT_DIR = "spectral_runs_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ZETA_CSV = os.path.join(OUTPUT_DIR, "zeta_zeros_100k.csv")
EXPERIMENT_LOG_CSV = os.path.join(OUTPUT_DIR, "experiment_log.csv")
RUN_SUMMARY_JSON = os.path.join(OUTPUT_DIR, "run_summary.json")
CLASSICAL_CLUSTER_PNG = os.path.join(OUTPUT_DIR, "classical_clusters.png")
CLASSICAL_RESULTS_CSV = os.path.join(OUTPUT_DIR, "classical_results.csv")
CLASSICAL_ABLATION_CSV = os.path.join(OUTPUT_DIR, "classical_ablation.csv")
DEEP_ID_RESULTS_CSV = os.path.join(OUTPUT_DIR, "deep_id_results.csv")
DEEP_OOD_RESULTS_CSV = os.path.join(OUTPUT_DIR, "deep_ood_results.csv")


# =========================================================
# 1. CONFIGURATION
# =========================================================
@dataclass
class BaseDataConfig:
    low: int = 500
    high: int = 5000
    count_per_class: int = 150
    random_state: int = 42
    use_multiprocessing: bool = True
    n_workers: Optional[int] = None
    checkpoint_every: int = 50


@dataclass
class SpectralGeoConfig(BaseDataConfig):
    base_n_terms: int = 1000
    decay_power: float = 1.0
    damping_mode: str = "fejer"
    phase_shift: float = 0.0

    include_cos: bool = True
    include_sin: bool = True
    include_log_features: bool = True
    include_prime_power_flag: bool = False

    include_persistent_topology: bool = True
    topology_window: int = 7
    topology_scale: bool = True
    topology_max_dim: int = 1

    include_graph_spectral: bool = True
    graph_use_knn: bool = True
    graph_k: int = 3

    include_migadic_time: bool = True
    migadic_time_steps: List[float] = field(default_factory=lambda: [0.1, 1.0, 5.0])

    include_dynamic_topology_proxy: bool = True
    dynamic_window_offsets: List[int] = field(default_factory=lambda: [-1, 0, 1])
    dynamic_overlap_eps: float = 0.25

    raw_feature_cap: int = 1500
    band_count: int = 16
    use_compressed_for_large_n: bool = True


@dataclass
class SpectralSequenceConfig(BaseDataConfig):
    base_n_terms: int = 1000
    decay_power: float = 1.0
    topology_window: int = 7
    graph_k: int = 3
    seq_length: int = 20
    time_max: float = 15.0
    speed_of_light_c: float = 1.0


# =========================================================
# 2. FILE / CHECKPOINT HELPERS
# =========================================================
def stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def config_fingerprint(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()[:16]


def checkpoint_path(prefix: str, fp: str) -> str:
    return os.path.join(OUTPUT_DIR, f"{prefix}_{fp}.pkl")


def save_pickle(obj: Any, filepath: str) -> None:
    tmp = filepath + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, filepath)


def load_pickle(filepath: str) -> Any:
    with open(filepath, "rb") as f:
        return pickle.load(f)


def write_rows_to_csv(filepath: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            clean = {}
            for k, v in row.items():
                if isinstance(v, (dict, list, tuple)):
                    clean[k] = json.dumps(v, ensure_ascii=False)
                else:
                    clean[k] = v
            writer.writerow(clean)


def append_experiment_log(filepath: str, row: Dict[str, Any]) -> None:
    safe_row = {}
    for k, v in row.items():
        if isinstance(v, (dict, list, tuple)):
            safe_row[k] = json.dumps(v, ensure_ascii=False)
        else:
            safe_row[k] = v

    existing_rows: List[Dict[str, str]] = []
    existing_fields: set = set()
    if os.path.exists(filepath):
        with open(filepath, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
            existing_fields = set(reader.fieldnames or [])

    fields = sorted(existing_fields.union(safe_row.keys()))
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for old in existing_rows:
            writer.writerow({k: old.get(k, "") for k in fields})
        writer.writerow({k: safe_row.get(k, "") for k in fields})


def write_summary_json(filepath: str, payload: Dict[str, Any]) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# =========================================================
# 3. ZETA ZERO DOWNLOAD / LOAD
# =========================================================
def ensure_zeros_exist(filename: str = ZETA_CSV) -> None:
    if os.path.exists(filename):
        return

    url = "http://www.dtc.umn.edu/~odlyzko/zeta_tables/zeros1"
    print(f"Downloading zeta zeros from {url} ...")

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            raw = response.read().decode("utf-8", errors="ignore")
    except Exception as e:
        raise RuntimeError(f"Download failed: {e}") from e

    zeros: List[float] = []
    for line in raw.splitlines():
        for token in line.split():
            try:
                zeros.append(float(token))
            except ValueError:
                continue

    if not zeros:
        raise RuntimeError("No zeta zeros were parsed from downloaded content.")

    zeros = sorted(zeros)[:100000]
    for i in range(1, len(zeros)):
        if zeros[i] <= zeros[i - 1]:
            raise RuntimeError("Downloaded zeta zeros are not strictly increasing.")

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for z in zeros:
            writer.writerow([z])

    print(f"Saved {len(zeros)} zeros to {filename}")


def load_zeta_zeros_from_csv(filepath: str = ZETA_CSV) -> np.ndarray:
    zeros: List[float] = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            try:
                zeros.append(float(row[0].strip()))
            except Exception:
                continue

    if not zeros:
        raise ValueError(f"No valid zeros found in {filepath}")

    zeros = sorted(zeros)
    for i in range(1, len(zeros)):
        if zeros[i] <= zeros[i - 1]:
            raise ValueError("Loaded zeta zeros are not strictly increasing.")
    return np.array(zeros, dtype=np.float64)


# =========================================================
# 4. NUMBER THEORY HELPERS
# =========================================================
def is_prime_classic(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    lim = math.isqrt(n)
    for d in range(3, lim + 1, 2):
        if n % d == 0:
            return False
    return True


def is_prime_power(n: int) -> bool:
    if n < 4:
        return False
    max_k = int(math.log2(n)) + 1
    for k in range(2, max_k + 1):
        root = round(n ** (1.0 / k))
        for cand in {max(2, root - 1), root, root + 1}:
            if cand > 1 and cand ** k == n and is_prime_classic(cand):
                return True
    return False


def smallest_prime_factor(n: int) -> int:
    if n < 2:
        return n
    if n % 2 == 0:
        return 2
    lim = math.isqrt(n)
    for d in range(3, lim + 1, 2):
        if n % d == 0:
            return d
    return n


def factor_semiprime_if_semiprime(n: int) -> Optional[Tuple[int, int]]:
    if n < 4:
        return None
    spf = smallest_prime_factor(n)
    if spf == n:
        return None
    q = n // spf
    if spf * q != n:
        return None
    if is_prime_classic(spf) and is_prime_classic(q):
        return (min(spf, q), max(spf, q))
    return None


def semiprime_balance_ratio(n: int) -> Optional[float]:
    factors = factor_semiprime_if_semiprime(n)
    if factors is None:
        return None
    p, q = factors
    return float(p / q) if q > 0 else None


def trial_division_cost_proxy(n: int) -> float:
    if n < 2:
        return 0.0
    return float(math.log(max(2, smallest_prime_factor(n))))


def classify_hardness_proxy(
    n: int,
    small_factor_threshold: int = 11,
    balance_threshold: float = 0.4,
) -> Optional[int]:
    """
    0 -> prime
    1 -> composite with small prime factor
    2 -> balanced semiprime
    """
    if is_prime_classic(n):
        return 0

    spf = smallest_prime_factor(n)
    if spf < n and spf <= small_factor_threshold:
        return 1

    semi = factor_semiprime_if_semiprime(n)
    if semi is not None:
        p, q = semi
        if p / q >= balance_threshold:
            return 2
    return None


def generate_hardness_proxy_candidates(low: int, high: int) -> Dict[int, List[int]]:
    buckets = {0: [], 1: [], 2: []}
    for n in range(max(2, low), high + 1):
        cls = classify_hardness_proxy(n)
        if cls is not None:
            buckets[cls].append(n)
    return buckets


# =========================================================
# 5. LOW-LEVEL HELPERS
# =========================================================
def damping_weights(n_terms: int, mode: str = "fejer") -> np.ndarray:
    if n_terms <= 0:
        return np.empty(0, dtype=np.float64)
    if mode == "none":
        return np.ones(n_terms, dtype=np.float64)
    if mode == "fejer":
        if n_terms == 1:
            return np.ones(1, dtype=np.float64)
        k = np.arange(1, n_terms + 1, dtype=np.float64)
        return 1.0 - (k - 1.0) / (n_terms - 1.0)
    raise ValueError(f"Unknown damping mode: {mode}")


def split_into_bands(arr: np.ndarray, band_count: int) -> List[np.ndarray]:
    if arr.size == 0:
        return [np.empty(0, dtype=np.float64) for _ in range(band_count)]
    return [chunk for chunk in np.array_split(arr, band_count)]


def summarize_array(arr: np.ndarray) -> List[float]:
    if arr.size == 0:
        return [0.0] * 6
    return [
        float(np.mean(arr)),
        float(np.std(arr)),
        float(np.sum(arr)),
        float(np.max(arr)),
        float(np.min(arr)),
        float(np.linalg.norm(arr)),
    ]


def summarize_diagram(diagram: np.ndarray) -> List[float]:
    if diagram is None or len(diagram) == 0:
        return [0.0, 0.0, 0.0]
    lifetimes = []
    for birth, death in diagram:
        if np.isfinite(death):
            life = float(death - birth)
            if life >= 0:
                lifetimes.append(life)
    if not lifetimes:
        return [0.0, 0.0, 0.0]
    arr = np.array(lifetimes, dtype=np.float64)
    return [float(np.max(arr)), float(np.mean(arr)), float(np.sum(arr))]


def print_class_counts(name: str, y_arr: np.ndarray) -> None:
    counts = {int(c): int(np.sum(y_arr == c)) for c in np.unique(y_arr)}
    print(f"{name} class counts: {counts}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =========================================================
# 6. STATIC FEATURE BLOCKS
# =========================================================
def build_raw_spectral_block(x: int, cfg: SpectralGeoConfig, zeta_zeros: np.ndarray) -> np.ndarray:
    log_x = math.log(x)
    n = min(cfg.base_n_terms, len(zeta_zeros))
    gammas = zeta_zeros[:n]
    base_w = np.ones_like(gammas) if cfg.decay_power == 0 else 1.0 / np.power(gammas, cfg.decay_power)
    weights = base_w * damping_weights(n, cfg.damping_mode)

    feats: List[float] = []
    if cfg.include_cos:
        feats.extend((weights * np.cos(gammas * log_x + cfg.phase_shift)).tolist())
    if cfg.include_sin:
        feats.extend((weights * np.sin(gammas * log_x + cfg.phase_shift)).tolist())
    if cfg.include_log_features:
        feats.extend([log_x, 1.0 / log_x, float(n)])
    if cfg.include_prime_power_flag:
        feats.append(1.0 if is_prime_power(x) else 0.0)
    return np.array(feats, dtype=np.float64)


def build_compressed_spectral_block(x: int, cfg: SpectralGeoConfig, zeta_zeros: np.ndarray) -> np.ndarray:
    log_x = math.log(x)
    n = min(cfg.base_n_terms, len(zeta_zeros))
    gammas = zeta_zeros[:n]
    base_w = np.ones_like(gammas) if cfg.decay_power == 0 else 1.0 / np.power(gammas, cfg.decay_power)
    weights = base_w * damping_weights(n, cfg.damping_mode)

    cos_vals = weights * np.cos(gammas * log_x + cfg.phase_shift)
    sin_vals = weights * np.sin(gammas * log_x + cfg.phase_shift)

    feats: List[float] = []
    if cfg.include_cos:
        feats.extend(summarize_array(cos_vals))
    if cfg.include_sin:
        feats.extend(summarize_array(sin_vals))

    real_part = float(np.sum(cos_vals))
    imag_part = float(np.sum(sin_vals))
    magnitude = math.sqrt(real_part ** 2 + imag_part ** 2)
    energy = float(np.sum(cos_vals ** 2 + sin_vals ** 2))
    feats.extend([real_part, imag_part, magnitude, energy])

    if cfg.include_cos:
        for band in split_into_bands(cos_vals, cfg.band_count):
            feats.extend([
                float(np.sum(band)) if band.size else 0.0,
                float(np.mean(band)) if band.size else 0.0,
                float(np.linalg.norm(band)) if band.size else 0.0,
            ])
    if cfg.include_sin:
        for band in split_into_bands(sin_vals, cfg.band_count):
            feats.extend([
                float(np.sum(band)) if band.size else 0.0,
                float(np.mean(band)) if band.size else 0.0,
                float(np.linalg.norm(band)) if band.size else 0.0,
            ])

    if cfg.include_log_features:
        feats.extend([log_x, 1.0 / log_x, float(n)])
    if cfg.include_prime_power_flag:
        feats.append(1.0 if is_prime_power(x) else 0.0)

    return np.array(feats, dtype=np.float64)


def build_spectral_block(x: int, cfg: SpectralGeoConfig, zeta_zeros: np.ndarray) -> np.ndarray:
    n = min(cfg.base_n_terms, len(zeta_zeros))
    if cfg.use_compressed_for_large_n and n > cfg.raw_feature_cap:
        return build_compressed_spectral_block(x, cfg, zeta_zeros)
    return build_raw_spectral_block(x, cfg, zeta_zeros)


# =========================================================
# 7. GEOMETRY / TOPOLOGY BLOCKS
# =========================================================
def build_knn_weight_matrix(point_cloud: np.ndarray, k: int = 3) -> np.ndarray:
    n = point_cloud.shape[0]
    if n <= 1:
        return np.zeros((n, n), dtype=np.float64)

    dists = squareform(pdist(point_cloud, metric="euclidean"))
    upper = dists[np.triu_indices_from(dists, k=1)]
    sigma = np.mean(upper) if upper.size and np.mean(upper) > 0 else 1.0

    W = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        nbr_idx = np.argsort(dists[i])[1:k + 1]
        for j in nbr_idx:
            w = math.exp(-(dists[i, j] ** 2) / (2.0 * sigma ** 2))
            W[i, j] = max(W[i, j], w)
            W[j, i] = max(W[j, i], w)
    return W


def build_local_point_cloud(x: int, cfg: SpectralGeoConfig, zeta_zeros: np.ndarray) -> np.ndarray:
    half_window = cfg.topology_window // 2
    cloud = []
    for step in range(-half_window, half_window + 1):
        neighbor = max(2, x + step)
        cloud.append(build_spectral_block(neighbor, cfg, zeta_zeros))
    point_cloud = np.array(cloud, dtype=np.float64)
    if cfg.topology_scale and point_cloud.shape[0] > 1:
        point_cloud = StandardScaler().fit_transform(point_cloud)
    return point_cloud


def compute_single_local_geometry(
    x: int,
    cfg: SpectralGeoConfig,
    zeta_zeros: np.ndarray
) -> Tuple[Dict[str, List[float]], np.ndarray]:
    point_cloud = build_local_point_cloud(x, cfg, zeta_zeros)

    res: Dict[str, List[float]] = {"topology": [], "graph": [], "migadic": []}

    if cfg.include_persistent_topology:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            diagrams = ripser(point_cloud, maxdim=cfg.topology_max_dim)["dgms"]

        h0 = diagrams[0] if len(diagrams) > 0 else np.empty((0, 2))
        res["topology"].extend(summarize_diagram(h0))
        if cfg.topology_max_dim >= 1:
            h1 = diagrams[1] if len(diagrams) > 1 else np.empty((0, 2))
            res["topology"].extend(summarize_diagram(h1))

    evals = np.array([], dtype=np.float64)
    if cfg.include_graph_spectral or cfg.include_migadic_time:
        if point_cloud.shape[0] > 1:
            if cfg.graph_use_knn:
                W = build_knn_weight_matrix(point_cloud, k=cfg.graph_k)
            else:
                dists = squareform(pdist(point_cloud, metric="euclidean"))
                upper = dists[np.triu_indices_from(dists, k=1)]
                sigma = np.mean(upper) if upper.size and np.mean(upper) > 0 else 1.0
                W = np.exp(-(dists ** 2) / (2.0 * sigma ** 2))
                np.fill_diagonal(W, 0.0)

            D = np.diag(np.sum(W, axis=1))
            L = D - W
            evals = np.maximum(np.sort(eigh(L, eigvals_only=True)), 0.0)
        else:
            evals = np.array([0.0], dtype=np.float64)

    if cfg.include_graph_spectral:
        if evals.size > 1:
            res["graph"].extend([
                float(np.sum(evals < 1e-6)),
                float(evals[1]),
                float(evals[-1]),
                float(evals[-1] - evals[1]),
                float(np.sum(evals)),
                float(np.linalg.norm(evals)),
            ])
        else:
            res["graph"].extend([0.0] * 6)

    if cfg.include_migadic_time:
        for t in cfg.migadic_time_steps:
            trace_val = np.sum(np.exp(-1j * evals * t))
            res["migadic"].extend([
                float(np.real(trace_val)),
                float(np.imag(trace_val)),
                float(np.abs(trace_val)),
                float(np.abs(trace_val) ** 2),
            ])

    return res, evals


def compute_dynamic_topology_proxy(
    x: int,
    cfg: SpectralGeoConfig,
    zeta_zeros: np.ndarray
) -> List[float]:
    if not cfg.include_dynamic_topology_proxy:
        return []

    states = []
    eigs = []
    for offset in cfg.dynamic_window_offsets:
        geom, evals = compute_single_local_geometry(max(2, x + offset), cfg, zeta_zeros)
        states.append(geom)
        eigs.append(evals)

    if len(states) < 2:
        return [0.0] * 8

    topo_shift_scores = []
    graph_shift_scores = []
    shift_scores = []
    overlap_scores = []

    for i in range(len(states) - 1):
        topo_a = np.array(states[i]["topology"], dtype=np.float64)
        topo_b = np.array(states[i + 1]["topology"], dtype=np.float64)
        graph_a = np.array(states[i]["graph"], dtype=np.float64)
        graph_b = np.array(states[i + 1]["graph"], dtype=np.float64)

        topo_shift = float(np.linalg.norm(topo_a - topo_b)) if topo_a.size and topo_b.size else 0.0
        graph_shift = float(np.linalg.norm(graph_a - graph_b)) if graph_a.size and graph_b.size else 0.0
        total_shift = topo_shift + graph_shift

        ea, eb = eigs[i], eigs[i + 1]
        min_len = min(len(ea), len(eb))
        overlap = float(np.mean(np.abs(ea[:min_len] - eb[:min_len]) < cfg.dynamic_overlap_eps)) if min_len > 0 else 0.0

        topo_shift_scores.append(topo_shift)
        graph_shift_scores.append(graph_shift)
        shift_scores.append(total_shift)
        overlap_scores.append(overlap)

    return [
        float(np.mean(topo_shift_scores)),
        float(np.max(topo_shift_scores)),
        float(np.mean(graph_shift_scores)),
        float(np.max(graph_shift_scores)),
        float(np.mean(shift_scores)),
        float(np.max(shift_scores)),
        float(np.mean(overlap_scores)),
        float(np.min(overlap_scores)),
    ]


def build_feature_vector_with_blocks(
    x: int,
    cfg: SpectralGeoConfig,
    zeta_zeros: np.ndarray
) -> Tuple[np.ndarray, Dict[str, Tuple[int, int]]]:
    feats: List[float] = []
    blocks: Dict[str, Tuple[int, int]] = {}

    start = len(feats)
    spectral = build_spectral_block(x, cfg, zeta_zeros)
    feats.extend(spectral.tolist())
    blocks["spectral"] = (start, len(feats))

    geom, _ = compute_single_local_geometry(x, cfg, zeta_zeros)
    if cfg.include_dynamic_topology_proxy:
        geom["dynamic_topology"] = compute_dynamic_topology_proxy(x, cfg, zeta_zeros)
    else:
        geom["dynamic_topology"] = []

    for block_name in ["topology", "graph", "migadic", "dynamic_topology"]:
        if geom.get(block_name):
            a = len(feats)
            feats.extend(geom[block_name])
            blocks[block_name] = (a, len(feats))

    return np.array(feats, dtype=np.float64), blocks


def build_feature_vector(x: int, cfg: SpectralGeoConfig, zeta_zeros: np.ndarray) -> np.ndarray:
    vec, _ = build_feature_vector_with_blocks(x, cfg, zeta_zeros)
    return vec


# =========================================================
# 8. MULTIPROCESSING WORKERS
# =========================================================
def _static_compute_record_safe(
    pair: Tuple[int, int],
    cfg: SpectralGeoConfig,
    zeta_zeros: np.ndarray
) -> Optional[Tuple[int, np.ndarray, int, Dict[str, float]]]:
    try:
        x, label = pair
        vec = build_feature_vector(x, cfg, zeta_zeros)
        balance = semiprime_balance_ratio(x)
        meta = {
            "x": float(x),
            "label": float(label),
            "smallest_factor": float(smallest_prime_factor(x) if x >= 2 else 0),
            "trial_division_cost_proxy": float(trial_division_cost_proxy(x)),
            "semiprime_balance_ratio": float(balance) if balance is not None else -1.0,
        }
        return x, vec, label, meta
    except Exception as e:
        print(f"[WorkerError][static] x={pair[0]} -> {e}")
        return None


def _sequence_compute_record_safe(
    pair: Tuple[int, int],
    cfg: SpectralSequenceConfig,
    zeta_zeros: np.ndarray
) -> Optional[Tuple[int, np.ndarray, int]]:
    try:
        x, label = pair
        seq = build_sequence_matrix(x, cfg, zeta_zeros)
        return x, seq, label
    except Exception as e:
        print(f"[WorkerError][sequence] x={pair[0]} -> {e}")
        return None


# =========================================================
# 9. DATASET BUILDERS WITH RESUME
# =========================================================
def build_or_resume_static_dataset(
    cfg: SpectralGeoConfig,
    zeta_zeros: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, float]]]:
    fp = config_fingerprint({
        "mode": "classical_dataset_v2",
        "cfg": asdict(cfg),
        "zeta_terms": int(min(cfg.base_n_terms, len(zeta_zeros))),
    })
    ckpt_file = checkpoint_path("classical_dataset", fp)

    if os.path.exists(ckpt_file):
        ckpt = load_pickle(ckpt_file)
        records = ckpt["records"]
        processed_x = set(records.keys())
        print(f"[Resume] Loaded classical checkpoint with {len(processed_x)} samples.")
    else:
        records = {}
        processed_x = set()

    rng = np.random.default_rng(cfg.random_state)
    buckets = generate_hardness_proxy_candidates(cfg.low, cfg.high)

    target_pairs: List[Tuple[int, int]] = []
    for cls in [0, 1, 2]:
        available = len(buckets[cls])
        if available == 0:
            raise ValueError(f"No candidates for class {cls} in [{cfg.low}, {cfg.high}]")
        if cfg.count_per_class > available:
            raise ValueError(f"Not enough samples for class {cls}: requested {cfg.count_per_class}, available {available}")
        chosen = rng.choice(buckets[cls], size=cfg.count_per_class, replace=False)
        for x in chosen:
            target_pairs.append((int(x), cls))

    rng.shuffle(target_pairs)
    pending = [p for p in target_pairs if p[0] not in processed_x]
    print(f"[Classical] Pending samples: {len(pending)}")

    if cfg.use_multiprocessing and pending:
        workers = cfg.n_workers or max(1, mp.cpu_count() - 1)
        print(f"[Classical] Using {workers} workers")
        worker = partial(_static_compute_record_safe, cfg=cfg, zeta_zeros=zeta_zeros)
        with mp.Pool(workers) as pool:
            done = 0
            for out in pool.imap_unordered(worker, pending, chunksize=8):
                if out is None:
                    continue
                x, vec, label, meta = out
                records[int(x)] = {
                    "x": int(x),
                    "vec": vec,
                    "label": int(label),
                    "meta": meta,
                }
                done += 1
                if done % cfg.checkpoint_every == 0:
                    save_pickle({"records": records}, ckpt_file)
                    print(f"[Checkpoint] Saved classical progress: {len(records)} samples")
    else:
        done = 0
        for pair in pending:
            out = _static_compute_record_safe(pair, cfg, zeta_zeros)
            if out is None:
                continue
            x, vec, label, meta = out
            records[int(x)] = {
                "x": int(x),
                "vec": vec,
                "label": int(label),
                "meta": meta,
            }
            done += 1
            if done % cfg.checkpoint_every == 0:
                save_pickle({"records": records}, ckpt_file)
                print(f"[Checkpoint] Saved classical progress: {len(records)} samples")

    save_pickle({"records": records}, ckpt_file)

    ordered_x = sorted(records.keys())
    X = np.vstack([records[x]["vec"] for x in ordered_x]).astype(np.float64)
    y = np.array([records[x]["label"] for x in ordered_x], dtype=np.int64)
    meta = [records[x]["meta"] for x in ordered_x]
    return np.array(ordered_x, dtype=np.int64), X, y, meta


def build_or_resume_sequence_dataset(
    cfg: SpectralSequenceConfig,
    zeta_zeros: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fp = config_fingerprint({
        "mode": "sequence_dataset_v2",
        "cfg": asdict(cfg),
        "zeta_terms": int(min(cfg.base_n_terms, len(zeta_zeros))),
    })
    ckpt_file = checkpoint_path("sequence_dataset", fp)

    if os.path.exists(ckpt_file):
        ckpt = load_pickle(ckpt_file)
        records = ckpt["records"]
        processed_x = set(records.keys())
        print(f"[Resume] Loaded sequence checkpoint with {len(processed_x)} samples.")
    else:
        records = {}
        processed_x = set()

    rng = np.random.default_rng(cfg.random_state)
    buckets = generate_hardness_proxy_candidates(cfg.low, cfg.high)

    target_pairs: List[Tuple[int, int]] = []
    for cls in [0, 1, 2]:
        available = len(buckets[cls])
        if available == 0:
            raise ValueError(f"No candidates for class {cls} in [{cfg.low}, {cfg.high}]")
        if cfg.count_per_class > available:
            raise ValueError(f"Not enough samples for class {cls}: requested {cfg.count_per_class}, available {available}")
        chosen = rng.choice(buckets[cls], size=cfg.count_per_class, replace=False)
        for x in chosen:
            target_pairs.append((int(x), cls))

    rng.shuffle(target_pairs)
    pending = [p for p in target_pairs if p[0] not in processed_x]
    print(f"[Deep] Pending sequence samples: {len(pending)}")

    if cfg.use_multiprocessing and pending:
        workers = cfg.n_workers or max(1, mp.cpu_count() - 1)
        print(f"[Deep] Using {workers} workers")
        worker = partial(_sequence_compute_record_safe, cfg=cfg, zeta_zeros=zeta_zeros)
        with mp.Pool(workers) as pool:
            done = 0
            for out in pool.imap_unordered(worker, pending, chunksize=4):
                if out is None:
                    continue
                x, seq, label = out
                records[int(x)] = {
                    "x": int(x),
                    "seq": seq,
                    "label": int(label),
                }
                done += 1
                if done % cfg.checkpoint_every == 0:
                    save_pickle({"records": records}, ckpt_file)
                    print(f"[Checkpoint] Saved sequence progress: {len(records)} samples")
    else:
        done = 0
        for pair in pending:
            out = _sequence_compute_record_safe(pair, cfg, zeta_zeros)
            if out is None:
                continue
            x, seq, label = out
            records[int(x)] = {
                "x": int(x),
                "seq": seq,
                "label": int(label),
            }
            done += 1
            if done % cfg.checkpoint_every == 0:
                save_pickle({"records": records}, ckpt_file)
                print(f"[Checkpoint] Saved sequence progress: {len(records)} samples")

    save_pickle({"records": records}, ckpt_file)

    ordered_x = sorted(records.keys())
    X = np.array([records[x]["seq"] for x in ordered_x], dtype=np.float32)
    y = np.array([records[x]["label"] for x in ordered_x], dtype=np.int64)
    return np.array(ordered_x, dtype=np.int64), X, y


# =========================================================
# 10. SEQUENCE FEATURE GENERATOR
# =========================================================
def build_sequence_matrix(
    x: int,
    cfg: SpectralSequenceConfig,
    zeta_zeros: np.ndarray
) -> np.ndarray:
    log_x = math.log(max(2, x))
    n = min(cfg.base_n_terms, len(zeta_zeros))
    gammas = zeta_zeros[:n]
    weights = (1.0 / np.power(gammas, cfg.decay_power)) * damping_weights(n, "fejer")

    cos_vals = weights * np.cos(gammas * log_x)
    sin_vals = weights * np.sin(gammas * log_x)
    base_feats = [
        float(np.mean(cos_vals)),
        float(np.std(cos_vals)),
        float(np.mean(sin_vals)),
        float(np.std(sin_vals)),
        float(log_x),
    ]

    half_window = cfg.topology_window // 2
    point_cloud = []
    for step in range(-half_window, half_window + 1):
        neighbor = max(2, x + step)
        n_log = math.log(neighbor)
        pc_cos = weights * np.cos(gammas * n_log)
        pc_sin = weights * np.sin(gammas * n_log)
        point_cloud.append([
            float(np.mean(pc_cos)),
            float(np.std(pc_cos)),
            float(np.mean(pc_sin)),
            float(np.std(pc_sin)),
            float(n_log),
        ])

    point_cloud = StandardScaler().fit_transform(np.array(point_cloud, dtype=np.float64))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dgms = ripser(point_cloud, maxdim=1)["dgms"]

    h0_max = 0.0
    h1_max = 0.0
    if len(dgms) > 0 and dgms[0].size:
        h0_life = dgms[0][:, 1] - dgms[0][:, 0]
        h0_life = h0_life[np.isfinite(h0_life)]
        if h0_life.size > 0:
            h0_max = float(np.max(h0_life))
    if len(dgms) > 1 and dgms[1].size:
        h1_life = dgms[1][:, 1] - dgms[1][:, 0]
        h1_life = h1_life[np.isfinite(h1_life)]
        if h1_life.size > 0:
            h1_max = float(np.max(h1_life))

    dists = squareform(pdist(point_cloud, metric="euclidean"))
    upper = dists[np.triu_indices_from(dists, k=1)]
    sigma = np.mean(upper) if upper.size and np.mean(upper) > 0 else 1.0
    W = np.exp(-(dists ** 2) / (2.0 * sigma ** 2))
    np.fill_diagonal(W, 0.0)
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    evals = np.maximum(np.sort(eigh(L, eigvals_only=True)), 0.0)

    fiedler = float(evals[1]) if evals.size > 1 else 0.0
    max_eig = float(evals[-1]) if evals.size > 0 else 0.0

    mass_c2 = math.log(max(2, x)) * cfg.speed_of_light_c ** 2
    rel_e = np.sqrt((cfg.speed_of_light_c ** 2) * evals + mass_c2 ** 2)

    seq = []
    for t in np.linspace(0.1, cfg.time_max, cfg.seq_length):
        m_tr = np.sum(np.exp(-1j * evals * t))
        r_tr = np.sum(np.exp(-1j * rel_e * t))
        seq.append(base_feats + [
            h0_max,
            h1_max,
            fiedler,
            max_eig,
            float(np.real(m_tr)),
            float(np.imag(m_tr)),
            float(np.abs(m_tr)),
            float(np.abs(m_tr) ** 2),
            float(np.real(r_tr)),
            float(np.imag(r_tr)),
            float(np.abs(r_tr)),
            float(np.abs(r_tr) ** 2),
            float(t),
        ])

    return np.array(seq, dtype=np.float32)


# =========================================================
# 11. CLASSICAL MODELS / EVALUATION
# =========================================================
def metrics_dict_multiclass(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def make_multiclass_models(random_state: int = 42) -> Dict[str, Any]:
    return {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=random_state,
                multi_class="auto",
            )),
        ]),
        "rf": RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=1,
        ),
        "mlp_medium": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=1200,
                random_state=random_state,
            )),
        ]),
    }


def cross_validate_models(
    X: np.ndarray,
    y: np.ndarray,
    seeds: List[int],
    n_splits: int = 5,
) -> List[Dict[str, Any]]:
    rows = []
    for seed in seeds:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        models = make_multiclass_models(random_state=seed)
        for name, model in models.items():
            scores = cross_validate(
                model,
                X,
                y,
                cv=cv,
                scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
                n_jobs=1,
            )
            rows.append({
                "seed": seed,
                "model": name,
                "accuracy_mean": float(np.mean(scores["test_accuracy"])),
                "precision_macro_mean": float(np.mean(scores["test_precision_macro"])),
                "recall_macro_mean": float(np.mean(scores["test_recall_macro"])),
                "f1_macro_mean": float(np.mean(scores["test_f1_macro"])),
            })
    return rows


def train_and_test_models_random_split_multiclass(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.25,
    random_state: int = 42
) -> Tuple[Dict[str, Dict[str, Any]], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    models = make_multiclass_models(random_state=random_state)
    results: Dict[str, Dict[str, Any]] = {}

    for name, model in models.items():
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        preds = model.predict(X_test)
        infer_time = time.perf_counter() - t1

        metrics = metrics_dict_multiclass(y_test, preds)
        metrics["train_time_sec"] = float(train_time)
        metrics["infer_time_sec"] = float(infer_time)
        metrics["model"] = model
        results[name] = metrics

    return results, (X_train, X_test, y_train, y_test)


def evaluate_given_split_multiclass(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42
) -> Dict[str, Dict[str, Any]]:
    models = make_multiclass_models(random_state=random_state)
    results: Dict[str, Dict[str, Any]] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = metrics_dict_multiclass(y_test, preds)
    return results


def extract_rf_rules(
    model: Any,
    feature_names: Optional[List[str]] = None,
    tree_index: int = 0
) -> str:
    if not isinstance(model, RandomForestClassifier):
        return "Model must be a RandomForestClassifier to extract rules."
    tree = model.estimators_[tree_index]
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(tree.n_features_in_)]
    return export_text(tree, feature_names=feature_names, max_depth=4)


# =========================================================
# 12. INTERPRETABILITY / OOD
# =========================================================
def block_permutation_importance(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    block_map: Dict[str, Tuple[int, int]],
    random_state: int = 42
) -> List[Dict[str, float]]:
    rng = np.random.default_rng(random_state)
    baseline = f1_score(y_test, model.predict(X_test), average="macro", zero_division=0)
    rows = []

    for block_name, (a, b) in block_map.items():
        X_perm = X_test.copy()
        perm = rng.permutation(len(X_perm))
        X_perm[:, a:b] = X_perm[perm, a:b]
        score = f1_score(y_test, model.predict(X_perm), average="macro", zero_division=0)
        rows.append({
            "block": block_name,
            "baseline": float(baseline),
            "permuted_score": float(score),
            "drop": float(baseline - score),
        })
    rows.sort(key=lambda r: r["drop"], reverse=True)
    return rows


def split_semiprime_balance_ood_strict(
    x_vals: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    metadata_rows: List[Dict[str, float]],
    low_ratio_max: float = 0.20,
    high_ratio_min: float = 0.80,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    meta_by_x = {int(row["x"]): row for row in metadata_rows}
    train_idx, test_idx = [], []

    for i in np.where(y == 2)[0]:
        ratio = meta_by_x[int(x_vals[i])]["semiprime_balance_ratio"]
        if 0 <= ratio <= low_ratio_max:
            train_idx.append(i)
        elif ratio >= high_ratio_min:
            test_idx.append(i)

    if not train_idx or not test_idx:
        raise ValueError("Strict semiprime OOD split has empty low/high balance side.")

    for cls in [0, 1]:
        cls_idx = np.where(y == cls)[0]
        perm = rng.permutation(cls_idx)
        half = len(perm) // 2
        train_idx.extend(perm[:half])
        test_idx.extend(perm[half:])

    train_idx = np.array(sorted(set(train_idx)))
    test_idx = np.array(sorted(set(test_idx)))
    if set(train_idx).intersection(set(test_idx)):
        raise ValueError("Train/test overlap detected in classical OOD split.")

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def split_size_threshold_ood(
    x_vals: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    threshold: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_idx = np.where(x_vals <= threshold)[0]
    test_idx = np.where(x_vals > threshold)[0]
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Empty split in size-threshold OOD.")
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def split_semiprime_balance_ood_sequence(
    x_vals: np.ndarray,
    X_seq: np.ndarray,
    y: np.ndarray,
    low_ratio_max: float = 0.20,
    high_ratio_min: float = 0.80,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    train_idx, test_idx = [], []

    for i in np.where(y == 2)[0]:
        ratio = semiprime_balance_ratio(int(x_vals[i]))
        if ratio is None:
            continue
        if ratio <= low_ratio_max:
            train_idx.append(i)
        elif ratio >= high_ratio_min:
            test_idx.append(i)

    if not train_idx or not test_idx:
        raise ValueError("Empty balance OOD split for sequence data.")

    for cls in [0, 1]:
        cls_idx = np.where(y == cls)[0]
        perm = rng.permutation(cls_idx)
        half = len(perm) // 2
        train_idx.extend(perm[:half])
        test_idx.extend(perm[half:])

    train_idx = np.array(sorted(set(train_idx)))
    test_idx = np.array(sorted(set(test_idx)))
    if set(train_idx).intersection(set(test_idx)):
        raise ValueError("Train/test overlap in sequence OOD split.")

    return X_seq[train_idx], X_seq[test_idx], y[train_idx], y[test_idx]


# =========================================================
# 13. DEEP MODELS
# =========================================================
class TransformerLSTMHybrid(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 3, num_heads: int = 4):
        super().__init__()
        self.projector = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim // 2, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        z = self.transformer(self.projector(x))
        out, _ = self.lstm(z)
        return self.classifier(out[:, -1, :])


class MaxwellsDemonGNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 3):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.demon_agent = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        b, s, _ = x.size()
        nodes = self.node_encoder(x)

        pairs = torch.cat([
            nodes.unsqueeze(2).expand(b, s, s, -1),
            nodes.unsqueeze(1).expand(b, s, s, -1),
        ], dim=-1)

        raw_msgs = self.edge_network(pairs)
        keep = self.demon_agent(pairs)
        active = raw_msgs * keep
        trash = raw_msgs * (1.0 - keep)

        main_emb = (nodes + active.sum(dim=2)).mean(dim=1)
        trash_emb = trash.sum(dim=2).mean(dim=1)
        return self.classifier(torch.cat([main_emb, trash_emb], dim=-1))


class GraphTransformerDemon(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 3, num_heads: int = 4):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.edge_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.demon_agent = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        b, s, _ = x.size()
        nodes = self.transformer(self.projector(x))

        pairs = torch.cat([
            nodes.unsqueeze(2).expand(b, s, s, -1),
            nodes.unsqueeze(1).expand(b, s, s, -1),
        ], dim=-1)

        raw_msgs = self.edge_network(pairs)
        keep = self.demon_agent(pairs)
        active = raw_msgs * keep
        trash = raw_msgs * (1.0 - keep)

        main_emb = (nodes + active.sum(dim=2)).mean(dim=1)
        trash_emb = trash.sum(dim=2).mean(dim=1)
        return self.classifier(torch.cat([main_emb, trash_emb], dim=-1))


# =========================================================
# 14. DEEP TRAIN / EVAL
# =========================================================
def train_and_evaluate_deep_model(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    lr: float = 0.001,
) -> Dict[str, float]:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    t0 = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"{model_name:>24} | Epoch {epoch + 1:02d}/{epochs} | Loss: {total_loss / max(1, len(train_loader)):.4f}")

    train_time = time.perf_counter() - t0

    model.eval()
    all_preds, all_labels = [], []
    t1 = time.perf_counter()
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            preds = torch.argmax(model(batch_x.to(device)), dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(batch_y.numpy().tolist())
    infer_time = time.perf_counter() - t1

    return {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "f1_macro": float(f1_score(all_labels, all_preds, average="macro")),
        "train_time_sec": float(train_time),
        "infer_time_sec": float(infer_time),
        "params": float(count_parameters(model)),
    }


def evaluate_deep_model_on_given_split(
    model_name: str,
    model_cls,
    model_kwargs: Dict[str, Any],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    epochs: int = 20,
    lr: float = 0.001,
) -> Dict[str, float]:
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        batch_size=16,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)),
        batch_size=16,
        shuffle=False,
    )

    model = model_cls(**model_kwargs).to(device)
    return train_and_evaluate_deep_model(model_name, model, train_loader, test_loader, device, epochs=epochs, lr=lr)


# =========================================================
# 15. CLASSICAL AUDIT
# =========================================================
def evaluate_classical_multi_seed(
    X: np.ndarray,
    y: np.ndarray,
    seeds: List[int]
) -> Dict[str, Dict[str, float]]:
    collected: Dict[str, List[float]] = {}
    for seed in seeds:
        results, _ = train_and_test_models_random_split_multiclass(X, y, random_state=seed)
        for model_name, metrics in results.items():
            collected.setdefault(model_name, [])
            collected[model_name].append(metrics["f1_macro"])

    summary = {}
    for model_name, values in collected.items():
        summary[model_name] = {
            "f1_macro_mean": float(np.mean(values)),
            "f1_macro_std": float(np.std(values)),
        }
    return summary


def run_classical_dynamic_audit(
    cfg: SpectralGeoConfig,
    zeta_zeros: np.ndarray,
    seeds: List[int]
) -> Dict[str, Any]:
    print("\n" + "=" * 90)
    print("CLASSICAL DYNAMIC SPECTRAL / TOPOLOGY AUDIT")
    print("=" * 90)

    sample_x = 997
    _, block_map = build_feature_vector_with_blocks(sample_x, cfg, zeta_zeros)

    x_vals, X, y, metadata_rows = build_or_resume_static_dataset(cfg, zeta_zeros)
    print(f"Static dataset shape: {X.shape}")
    print_class_counts("Full classical dataset", y)

    cv_rows = cross_validate_models(X, y, seeds=seeds[:3], n_splits=5)

    seed_summary = evaluate_classical_multi_seed(X, y, seeds)

    base_seed = seeds[0]
    results, split_data = train_and_test_models_random_split_multiclass(X, y, random_state=base_seed)
    _, X_test, _, y_test = split_data

    classical_rows = []
    print("\n--- IN-DISTRIBUTION RESULTS ---")
    for model_name, metrics in results.items():
        print(f"{model_name:<12} | acc={metrics['accuracy']:.4f} | f1_macro={metrics['f1_macro']:.4f}")
        classical_rows.append({
            "split": "in_distribution",
            "model": model_name,
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "train_time_sec": metrics["train_time_sec"],
            "infer_time_sec": metrics["infer_time_sec"],
            "confusion_matrix": metrics["confusion_matrix"],
            "multi_seed_mean": seed_summary[model_name]["f1_macro_mean"],
            "multi_seed_std": seed_summary[model_name]["f1_macro_std"],
        })

    rf_model = results["rf"]["model"]
    feature_names = [f"feat_{i}" for i in range(X.shape[1])]
    for block_name, (a, b) in block_map.items():
        for i in range(a, b):
            feature_names[i] = f"{block_name}_{i - a}"

    rf_rules = extract_rf_rules(rf_model, feature_names=feature_names, tree_index=0)
    print("\n--- RANDOM FOREST DECISION RULES (Tree 0) ---")
    print(rf_rules)

    print("\n--- BLOCK PERMUTATION IMPORTANCE ---")
    imp_rows = block_permutation_importance(rf_model, X_test, y_test, block_map, random_state=base_seed)
    for row in imp_rows:
        print(f"Block: {row['block']:<16} | Drop: {row['drop']:>7.4f}")

    print("\n--- UNSUPERVISED CLUSTERING ---")
    kmeans = KMeans(n_clusters=3, random_state=base_seed, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    ari = adjusted_rand_score(y, cluster_labels)
    print(f"Adjusted Rand Index: {ari:.4f}")

    pca = PCA(n_components=2, random_state=base_seed)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(10, 6))
    plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], alpha=0.65, label="Primes")
    plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], alpha=0.65, label="Small-factor composites")
    plt.scatter(X_pca[y == 2, 0], X_pca[y == 2, 1], alpha=0.65, label="Balanced semiprimes")
    plt.title("Classical Spectral-Geometric Hardness Proxy Clusters")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CLASSICAL_CLUSTER_PNG, dpi=300)
    plt.close()

    print("\n--- OOD SPLIT: BALANCE SHIFT ---")
    X_train_ood, X_test_ood, y_train_ood, y_test_ood = split_semiprime_balance_ood_strict(
        x_vals, X, y, metadata_rows, random_state=base_seed
    )
    ood_balance = evaluate_given_split_multiclass(X_train_ood, X_test_ood, y_train_ood, y_test_ood, random_state=base_seed)
    for model_name, metrics in ood_balance.items():
        classical_rows.append({
            "split": "ood_balance",
            "model": model_name,
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "train_time_sec": None,
            "infer_time_sec": None,
            "confusion_matrix": metrics["confusion_matrix"],
        })

    print("\n--- OOD SPLIT: SIZE THRESHOLD ---")
    threshold = int(np.median(x_vals))
    X_train_sz, X_test_sz, y_train_sz, y_test_sz = split_size_threshold_ood(x_vals, X, y, threshold=threshold)
    ood_size = evaluate_given_split_multiclass(X_train_sz, X_test_sz, y_train_sz, y_test_sz, random_state=base_seed)
    for model_name, metrics in ood_size.items():
        classical_rows.append({
            "split": "ood_size",
            "model": model_name,
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "train_time_sec": None,
            "infer_time_sec": None,
            "confusion_matrix": metrics["confusion_matrix"],
        })

    write_rows_to_csv(CLASSICAL_RESULTS_CSV, classical_rows)
    append_experiment_log(EXPERIMENT_LOG_CSV, {
        "stage": "classical",
        "best_id_model": max(results.keys(), key=lambda k: results[k]["f1_macro"]),
        "best_id_f1_macro": max(v["f1_macro"] for v in results.values()),
        "ari": ari,
        "timestamp_unix": time.time(),
    })

    return {
        "classical_rows": classical_rows,
        "rf_rules": rf_rules,
        "permutation_importance": imp_rows,
        "ari": ari,
        "cv_rows": cv_rows,
        "seed_summary": seed_summary,
        "dataset_shape": X.shape,
    }


# =========================================================
# 16. DEEP AUDIT
# =========================================================
def compare_deep_models(
    cfg: SpectralSequenceConfig,
    zeta_zeros: np.ndarray,
    seeds: List[int]
) -> Dict[str, Any]:
    print("\n" + "=" * 90)
    print("DEEP SHOWDOWN: TRANSFORMER vs DEMON vs GRAPH-TRANSFORMER-DEMON")
    print("=" * 90)

    x_vals, X, y = build_or_resume_sequence_dataset(cfg, zeta_zeros)
    b, s, d = X.shape
    print(f"Deep dataset shape: {X.shape}")
    print_class_counts("Deep dataset", y)

    base_seed = seeds[0]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=base_seed
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, d)).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, d)).reshape(X_test.shape)
    X_scaled_full = scaler.fit_transform(X.reshape(-1, d)).reshape(X.shape)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        batch_size=16,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)),
        batch_size=16,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device.type.upper()}...")

    model_specs = {
        "Transformer-LSTM": (TransformerLSTMHybrid, {"input_dim": d, "hidden_dim": 64, "num_classes": 3}),
        "Demon-GNN": (MaxwellsDemonGNN, {"input_dim": d, "hidden_dim": 64, "num_classes": 3}),
        "Godzilla (G-T-D)": (GraphTransformerDemon, {"input_dim": d, "hidden_dim": 64, "num_classes": 3}),
    }

    deep_id_rows = []
    in_dist_results = {}
    for name, (model_cls, kwargs) in model_specs.items():
        model = model_cls(**kwargs).to(device)
        res = train_and_evaluate_deep_model(name, model, train_loader, test_loader, device)
        in_dist_results[name] = res
        deep_id_rows.append({
            "split": "in_distribution",
            "model": name,
            "accuracy": res["accuracy"],
            "f1_macro": res["f1_macro"],
            "params": int(res["params"]),
            "train_time_sec": res["train_time_sec"],
            "infer_time_sec": res["infer_time_sec"],
        })

    print("\n--- DEEP IN-DISTRIBUTION ---")
    for name, res in in_dist_results.items():
        print(f"{name:<24} | acc={res['accuracy']:.4f} | f1_macro={res['f1_macro']:.4f} | params={int(res['params'])}")

    X_train_ood, X_test_ood, y_train_ood, y_test_ood = split_semiprime_balance_ood_sequence(
        x_vals, X_scaled_full, y, random_state=base_seed
    )

    deep_ood_rows = []
    ood_results = {}
    for name, (model_cls, kwargs) in model_specs.items():
        res = evaluate_deep_model_on_given_split(
            f"{name}[OOD]",
            model_cls,
            kwargs,
            X_train_ood,
            X_test_ood,
            y_train_ood,
            y_test_ood,
            device,
        )
        ood_results[name] = res
        deep_ood_rows.append({
            "split": "ood_balance",
            "model": name,
            "accuracy": res["accuracy"],
            "f1_macro": res["f1_macro"],
            "params": int(res["params"]),
            "train_time_sec": res["train_time_sec"],
            "infer_time_sec": res["infer_time_sec"],
            "robustness_ratio": res["f1_macro"] / max(1e-12, in_dist_results[name]["f1_macro"]),
        })

    write_rows_to_csv(DEEP_ID_RESULTS_CSV, deep_id_rows)
    write_rows_to_csv(DEEP_OOD_RESULTS_CSV, deep_ood_rows)

    winner = max(in_dist_results.items(), key=lambda x: x[1]["f1_macro"])[0]
    append_experiment_log(EXPERIMENT_LOG_CSV, {
        "stage": "deep",
        "winner_in_distribution": winner,
        "best_id_f1_macro": max(v["f1_macro"] for v in in_dist_results.values()),
        "best_ood_f1_macro": max(v["f1_macro"] for v in ood_results.values()),
        "timestamp_unix": time.time(),
    })

    return {
        "deep_id_rows": deep_id_rows,
        "deep_ood_rows": deep_ood_rows,
        "winner_in_distribution": winner,
        "dataset_shape": X.shape,
    }


# =========================================================
# 17. ABLATION SUITE
# =========================================================
def clone_geo_config(cfg: SpectralGeoConfig, **updates) -> SpectralGeoConfig:
    data = asdict(cfg)
    data.update(updates)
    return SpectralGeoConfig(**data)


def run_classical_ablation_suite(
    base_cfg: SpectralGeoConfig,
    zeta_zeros: np.ndarray,
    seeds: List[int]
) -> List[Dict[str, Any]]:
    print("\n" + "=" * 90)
    print("CLASSICAL ABLATION SUITE")
    print("=" * 90)

    variants = [
        ("full", base_cfg),
        ("no_topology", clone_geo_config(base_cfg, include_persistent_topology=False)),
        ("no_graph", clone_geo_config(base_cfg, include_graph_spectral=False)),
        ("no_migadic", clone_geo_config(base_cfg, include_migadic_time=False)),
        ("no_dynamic_proxy", clone_geo_config(base_cfg, include_dynamic_topology_proxy=False)),
        ("spectral_only", clone_geo_config(
            base_cfg,
            include_persistent_topology=False,
            include_graph_spectral=False,
            include_migadic_time=False,
            include_dynamic_topology_proxy=False,
        )),
    ]

    rows = []
    for variant_name, cfg in variants:
        print(f"\n[Ablation] {variant_name}")
        x_vals, X, y, metadata_rows = build_or_resume_static_dataset(cfg, zeta_zeros)

        seed_summary = evaluate_classical_multi_seed(X, y, seeds)
        best_model = max(seed_summary.keys(), key=lambda k: seed_summary[k]["f1_macro_mean"])
        best_id_f1 = seed_summary[best_model]["f1_macro_mean"]

        X_train_ood, X_test_ood, y_train_ood, y_test_ood = split_semiprime_balance_ood_strict(
            x_vals, X, y, metadata_rows, random_state=seeds[0]
        )
        ood_results = evaluate_given_split_multiclass(X_train_ood, X_test_ood, y_train_ood, y_test_ood, random_state=seeds[0])
        best_ood_model = max(ood_results.keys(), key=lambda k: ood_results[k]["f1_macro"])
        best_ood_f1 = ood_results[best_ood_model]["f1_macro"]

        row = {
            "variant": variant_name,
            "feature_dim": int(X.shape[1]),
            "best_id_model": best_model,
            "best_id_f1_macro_mean": best_id_f1,
            "best_id_f1_macro_std": seed_summary[best_model]["f1_macro_std"],
            "best_ood_model": best_ood_model,
            "best_ood_f1_macro": best_ood_f1,
            "best_robustness_ratio": best_ood_f1 / max(1e-12, best_id_f1),
        }
        rows.append(row)
        append_experiment_log(EXPERIMENT_LOG_CSV, {"stage": "classical_ablation", **row, "timestamp_unix": time.time()})

    write_rows_to_csv(CLASSICAL_ABLATION_CSV, rows)
    return rows


# =========================================================
# 18. MAIN
# =========================================================
if __name__ == "__main__":
    try:
        ensure_zeros_exist(ZETA_CSV)
        all_zeros = load_zeta_zeros_from_csv(ZETA_CSV)
        print(f"Loaded {len(all_zeros)} zeros.")
    except Exception as e:
        print(f"Falling back due to error: {e}")
        all_zeros = np.array(
            [14.134725, 21.022040, 25.010858, 30.424876, 32.935062],
            dtype=np.float64,
        )

    seeds = [7, 21, 42, 84, 123]

    classical_cfg = SpectralGeoConfig(
        base_n_terms=1000,
        decay_power=1.0,
        damping_mode="fejer",
        phase_shift=0.0,
        include_cos=True,
        include_sin=True,
        include_log_features=True,
        include_prime_power_flag=False,
        include_persistent_topology=True,
        topology_window=7,
        topology_scale=True,
        topology_max_dim=1,
        include_graph_spectral=True,
        graph_use_knn=True,
        graph_k=3,
        include_migadic_time=True,
        migadic_time_steps=[0.1, 1.0, 5.0],
        include_dynamic_topology_proxy=True,
        dynamic_window_offsets=[-1, 0, 1],
        dynamic_overlap_eps=0.25,
        raw_feature_cap=1500,
        band_count=16,
        use_compressed_for_large_n=True,
        use_multiprocessing=True,
        n_workers=None,
        random_state=42,
        low=500,
        high=5000,
        count_per_class=150,
        checkpoint_every=50,
    )

    sequence_cfg = SpectralSequenceConfig(
        base_n_terms=1000,
        decay_power=1.0,
        topology_window=7,
        graph_k=3,
        seq_length=20,
        time_max=15.0,
        speed_of_light_c=1.0,
        use_multiprocessing=True,
        n_workers=None,
        random_state=42,
        low=500,
        high=5000,
        count_per_class=150,
        checkpoint_every=25,
    )

    classical_summary = run_classical_dynamic_audit(classical_cfg, all_zeros, seeds=seeds)
    deep_summary = compare_deep_models(sequence_cfg, all_zeros, seeds=seeds)
    ablation_summary = run_classical_ablation_suite(classical_cfg, all_zeros, seeds=seeds)

    full_summary = {
        "classical_config": asdict(classical_cfg),
        "sequence_config": asdict(sequence_cfg),
        "seeds": seeds,
        "classical_summary": classical_summary,
        "deep_summary": deep_summary,
        "ablation_summary": ablation_summary,
        "artifacts": {
            "classical_results_csv": CLASSICAL_RESULTS_CSV,
            "deep_id_results_csv": DEEP_ID_RESULTS_CSV,
            "deep_ood_results_csv": DEEP_OOD_RESULTS_CSV,
            "classical_ablation_csv": CLASSICAL_ABLATION_CSV,
            "classical_cluster_png": CLASSICAL_CLUSTER_PNG,
            "experiment_log_csv": EXPERIMENT_LOG_CSV,
        },
    }
    write_summary_json(RUN_SUMMARY_JSON, full_summary)
    print(f"Saved run summary to: {RUN_SUMMARY_JSON}")
