"""
RESEARCH-GRADE v2.4
Leak-free deep evaluation + multi-seed analysis + sanity checks
================================================================
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
import random

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
    f1_score,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# =========================================================
# 0. PATHS / GLOBALS
# =========================================================
OUTPUT_DIR = "spectral_runs_v24"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ZETA_CSV = os.path.join(OUTPUT_DIR, "zeta_zeros_100k.csv")
EXPERIMENT_LOG_CSV = os.path.join(OUTPUT_DIR, "experiment_log.csv")
RUN_SUMMARY_JSON = os.path.join(OUTPUT_DIR, "run_summary.json")

CLASSICAL_RESULTS_CSV = os.path.join(OUTPUT_DIR, "classical_results.csv")
CLASSICAL_ABLATION_CSV = os.path.join(OUTPUT_DIR, "classical_ablation.csv")

DEEP_ID_RESULTS_CSV = os.path.join(OUTPUT_DIR, "deep_id_results.csv")
DEEP_OOD_RESULTS_CSV = os.path.join(OUTPUT_DIR, "deep_ood_results.csv")

DEEP_MULTI_SEED_ID_CSV = os.path.join(OUTPUT_DIR, "deep_multi_seed_id_results.csv")
DEEP_MULTI_SEED_OOD_CSV = os.path.join(OUTPUT_DIR, "deep_multi_seed_ood_results.csv")
DEEP_SANITY_CSV = os.path.join(OUTPUT_DIR, "deep_random_label_sanity_results.csv")
DEEP_SUMMARY_JSON = os.path.join(OUTPUT_DIR, "deep_summary_v24.json")


# =========================================================
# 1. CONFIGURATION
# =========================================================
@dataclass
class BaseDataConfig:
    low: int = 500
    high: int = 5000
    count_per_class: int = 100
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
# 2. REPRODUCIBILITY HELPERS
# =========================================================
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_torch_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# =========================================================
# 3. FILE / CHECKPOINT HELPERS
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
# 4. ZETA ZERO DOWNLOAD / LOAD
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
    zeros = sorted(zeros)
    return np.array(zeros, dtype=np.float64)


# =========================================================
# 5. NUMBER THEORY HELPERS
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
    balance_threshold: float = 0.4
) -> Optional[int]:
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
# 6. LOW-LEVEL HELPERS
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

    lifetimes = [
        float(death - birth)
        for birth, death in diagram
        if np.isfinite(death) and death - birth >= 0
    ]
    if not lifetimes:
        return [0.0, 0.0, 0.0]

    arr = np.array(lifetimes, dtype=np.float64)
    return [float(np.max(arr)), float(np.mean(arr)), float(np.sum(arr))]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =========================================================
# 7. STATIC FEATURE BLOCKS
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
    feats.extend([
        real_part,
        imag_part,
        math.sqrt(real_part ** 2 + imag_part ** 2),
        float(np.sum(cos_vals ** 2 + sin_vals ** 2)),
    ])

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
# 8. GEOMETRY / TOPOLOGY BLOCKS
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
    cloud = [
        build_spectral_block(max(2, x + step), cfg, zeta_zeros)
        for step in range(-half_window, half_window + 1)
    ]
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

        res["topology"].extend(summarize_diagram(diagrams[0] if len(diagrams) > 0 else np.empty((0, 2))))
        if cfg.topology_max_dim >= 1:
            res["topology"].extend(summarize_diagram(diagrams[1] if len(diagrams) > 1 else np.empty((0, 2))))

    evals = np.array([0.0], dtype=np.float64)
    if (cfg.include_graph_spectral or cfg.include_migadic_time) and point_cloud.shape[0] > 1:
        if cfg.graph_use_knn:
            W = build_knn_weight_matrix(point_cloud, k=cfg.graph_k)
        else:
            dists = squareform(pdist(point_cloud, metric="euclidean"))
            upper = dists[np.triu_indices_from(dists, k=1)]
            sigma = np.mean(upper) if upper.size and np.mean(upper) > 0 else 1.0
            W = np.exp(-(dists ** 2) / (2.0 * sigma ** 2))
            np.fill_diagonal(W, 0.0)

        D = np.diag(np.sum(W, axis=1))
        evals = np.maximum(np.sort(eigh(D - W, eigvals_only=True)), 0.0)

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


def compute_dynamic_topology_proxy(x: int, cfg: SpectralGeoConfig, zeta_zeros: np.ndarray) -> List[float]:
    if not cfg.include_dynamic_topology_proxy:
        return []

    states, eigs = zip(*[
        compute_single_local_geometry(max(2, x + offset), cfg, zeta_zeros)
        for offset in cfg.dynamic_window_offsets
    ])

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

        t_shift = float(np.linalg.norm(topo_a - topo_b)) if topo_a.size and topo_b.size else 0.0
        g_shift = float(np.linalg.norm(graph_a - graph_b)) if graph_a.size and graph_b.size else 0.0

        ea, eb = eigs[i], eigs[i + 1]
        min_len = min(len(ea), len(eb))
        overlap = float(np.mean(np.abs(ea[:min_len] - eb[:min_len]) < cfg.dynamic_overlap_eps)) if min_len > 0 else 0.0

        topo_shift_scores.append(t_shift)
        graph_shift_scores.append(g_shift)
        shift_scores.append(t_shift + g_shift)
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
    feats, blocks = [], {}

    start = len(feats)
    feats.extend(build_spectral_block(x, cfg, zeta_zeros).tolist())
    blocks["spectral"] = (start, len(feats))

    geom, _ = compute_single_local_geometry(x, cfg, zeta_zeros)
    geom["dynamic_topology"] = compute_dynamic_topology_proxy(x, cfg, zeta_zeros) if cfg.include_dynamic_topology_proxy else []

    for b_name in ["topology", "graph", "migadic", "dynamic_topology"]:
        if geom.get(b_name):
            a = len(feats)
            feats.extend(geom[b_name])
            blocks[b_name] = (a, len(feats))

    return np.array(feats, dtype=np.float64), blocks


def build_feature_vector(x: int, cfg: SpectralGeoConfig, zeta_zeros: np.ndarray) -> np.ndarray:
    return build_feature_vector_with_blocks(x, cfg, zeta_zeros)[0]


# =========================================================
# 9. SEQUENCE FEATURE GENERATOR
# =========================================================
def build_sequence_matrix(x: int, cfg: SpectralSequenceConfig, zeta_zeros: np.ndarray) -> np.ndarray:
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
        n_log = math.log(max(2, x + step))
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

    h0_max, h1_max = 0.0, 0.0

    if len(dgms) > 0 and dgms[0].size > 0:
        h0_life = dgms[0][:, 1] - dgms[0][:, 0]
        h0_life = h0_life[np.isfinite(h0_life)]
        if h0_life.size > 0:
            h0_max = float(np.max(h0_life))

    if len(dgms) > 1 and dgms[1].size > 0:
        h1_life = dgms[1][:, 1] - dgms[1][:, 0]
        h1_life = h1_life[np.isfinite(h1_life)]
        if h1_life.size > 0:
            h1_max = float(np.max(h1_life))

    dists = squareform(pdist(point_cloud, metric="euclidean"))
    upper = dists[np.triu_indices_from(dists, k=1)]
    sigma = float(np.mean(upper)) if upper.size and np.mean(upper) > 0 else 1.0

    W = np.exp(-(dists ** 2) / (2.0 * sigma ** 2))
    np.fill_diagonal(W, 0.0)

    evals = np.maximum(
        np.sort(eigh(np.diag(np.sum(W, axis=1)) - W, eigvals_only=True)),
        0.0
    )

    fiedler = float(evals[1]) if evals.size > 1 else 0.0
    max_eig = float(evals[-1]) if evals.size > 0 else 0.0
    rel_e = np.sqrt((cfg.speed_of_light_c ** 2) * evals + (math.log(max(2, x)) * cfg.speed_of_light_c ** 2) ** 2)

    seq = []
    for t in np.linspace(0.1, cfg.time_max, cfg.seq_length):
        m_tr = np.sum(np.exp(-1j * evals * t))
        r_tr = np.sum(np.exp(-1j * rel_e * t))

        seq.append(
            base_feats
            + [h0_max, h1_max, fiedler, max_eig]
            + [
                float(np.real(m_tr)),
                float(np.imag(m_tr)),
                float(np.abs(m_tr)),
                float(np.abs(m_tr) ** 2),
                float(np.real(r_tr)),
                float(np.imag(r_tr)),
                float(np.abs(r_tr)),
                float(np.abs(r_tr) ** 2),
                float(t),
            ]
        )

    return np.array(seq, dtype=np.float32)


# =========================================================
# 10. MULTIPROCESSING WORKERS
# =========================================================
def _static_compute_record_safe(pair: Tuple[int, int], cfg: SpectralGeoConfig, zeta_zeros: np.ndarray):
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
    except Exception:
        return None


def _sequence_compute_record_safe(pair: Tuple[int, int], cfg: SpectralSequenceConfig, zeta_zeros: np.ndarray):
    try:
        x, label = pair
        seq = build_sequence_matrix(x, cfg, zeta_zeros)
        return x, seq, label
    except Exception as e:
        print(f"Seq worker error at x={pair[0]}: {e}")
        return None


# =========================================================
# 11. DATASET BUILDERS WITH RESUME
# =========================================================
def build_or_resume_static_dataset(cfg: SpectralGeoConfig, zeta_zeros: np.ndarray):
    fp = config_fingerprint({
        "mode": "classical_dataset_v2",
        "cfg": asdict(cfg),
        "zeta_terms": int(min(cfg.base_n_terms, len(zeta_zeros))),
    })
    ckpt_file = checkpoint_path("classical_dataset", fp)

    records = {}
    processed_x = set()

    if os.path.exists(ckpt_file):
        payload = load_pickle(ckpt_file)
        records = payload["records"]
        processed_x = set(records.keys())

    rng = np.random.default_rng(cfg.random_state)
    buckets = generate_hardness_proxy_candidates(cfg.low, cfg.high)

    target_pairs = []
    for cls in [0, 1, 2]:
        chosen = rng.choice(
            buckets[cls],
            size=min(cfg.count_per_class, len(buckets[cls])),
            replace=False
        )
        target_pairs.extend([(int(x), cls) for x in chosen])

    rng.shuffle(target_pairs)
    pending = [p for p in target_pairs if p[0] not in processed_x]

    if cfg.use_multiprocessing and pending:
        workers = cfg.n_workers or max(1, mp.cpu_count() - 1)
        worker = partial(_static_compute_record_safe, cfg=cfg, zeta_zeros=zeta_zeros)

        with mp.Pool(workers) as pool:
            for done, out in enumerate(pool.imap_unordered(worker, pending, chunksize=8), 1):
                if out:
                    records[int(out[0])] = {
                        "x": int(out[0]),
                        "vec": out[1],
                        "label": int(out[2]),
                        "meta": out[3],
                    }
                if done % cfg.checkpoint_every == 0:
                    save_pickle({"records": records}, ckpt_file)
    else:
        for done, pair in enumerate(pending, 1):
            out = _static_compute_record_safe(pair, cfg, zeta_zeros)
            if out:
                records[int(out[0])] = {
                    "x": int(out[0]),
                    "vec": out[1],
                    "label": int(out[2]),
                    "meta": out[3],
                }
            if done % cfg.checkpoint_every == 0:
                save_pickle({"records": records}, ckpt_file)

    save_pickle({"records": records}, ckpt_file)

    ordered_x = sorted(records.keys())
    X = np.vstack([records[x]["vec"] for x in ordered_x]).astype(np.float64)
    y = np.array([records[x]["label"] for x in ordered_x], dtype=np.int64)
    meta = [records[x]["meta"] for x in ordered_x]

    return np.array(ordered_x, dtype=np.int64), X, y, meta


def build_or_resume_sequence_dataset(cfg: SpectralSequenceConfig, zeta_zeros: np.ndarray):
    fp = config_fingerprint({
        "mode": "sequence_dataset_v4",
        "cfg": asdict(cfg),
        "zeta_terms": int(min(cfg.base_n_terms, len(zeta_zeros))),
        "sequence_builder_version": "build_sequence_matrix_v24",
    })
    ckpt_file = checkpoint_path("sequence_dataset", fp)

    records = {}
    processed_x = set()

    if os.path.exists(ckpt_file):
        payload = load_pickle(ckpt_file)
        records = payload["records"]
        processed_x = set(records.keys())

    rng = np.random.default_rng(cfg.random_state)
    buckets = generate_hardness_proxy_candidates(cfg.low, cfg.high)

    target_pairs = []
    for cls in [0, 1, 2]:
        chosen = rng.choice(
            buckets[cls],
            size=min(cfg.count_per_class, len(buckets[cls])),
            replace=False
        )
        target_pairs.extend([(int(x), cls) for x in chosen])

    rng.shuffle(target_pairs)
    pending = [p for p in target_pairs if p[0] not in processed_x]

    if cfg.use_multiprocessing and pending:
        workers = cfg.n_workers or max(1, mp.cpu_count() - 1)
        worker = partial(_sequence_compute_record_safe, cfg=cfg, zeta_zeros=zeta_zeros)

        with mp.Pool(workers) as pool:
            for done, out in enumerate(pool.imap_unordered(worker, pending, chunksize=4), 1):
                if out:
                    records[int(out[0])] = {
                        "x": int(out[0]),
                        "seq": out[1],
                        "label": int(out[2]),
                    }
                if done % cfg.checkpoint_every == 0:
                    save_pickle({"records": records}, ckpt_file)
    else:
        for done, pair in enumerate(pending, 1):
            out = _sequence_compute_record_safe(pair, cfg, zeta_zeros)
            if out:
                records[int(out[0])] = {
                    "x": int(out[0]),
                    "seq": out[1],
                    "label": int(out[2]),
                }
            if done % cfg.checkpoint_every == 0:
                save_pickle({"records": records}, ckpt_file)

    save_pickle({"records": records}, ckpt_file)

    ordered_x = sorted(records.keys())
    if len(ordered_x) == 0:
        raise ValueError("Sequence dataset is completely empty.")

    X_seq = np.array([records[x]["seq"] for x in ordered_x], dtype=np.float32)
    y = np.array([records[x]["label"] for x in ordered_x], dtype=np.int64)

    return np.array(ordered_x, dtype=np.int64), X_seq, y


# =========================================================
# 12. CLASSICAL MODELS / EVALUATION
# =========================================================
def metrics_dict_multiclass(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def multiclass_metrics_detailed(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class_f1": [float(x) for x in per_class_f1],
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def make_multiclass_models(random_state: int = 42) -> Dict[str, Any]:
    return {
        "rf": RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=1,
        ),
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=random_state,
            ))
        ]),
    }


def cross_validate_models(X: np.ndarray, y: np.ndarray, seeds: List[int]) -> List[Dict[str, Any]]:
    rows = []
    for seed in seeds:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        for name, model in make_multiclass_models(random_state=seed).items():
            scores = cross_validate(model, X, y, cv=cv, scoring=["accuracy", "f1_macro"], n_jobs=1)
            rows.append({
                "seed": seed,
                "model": name,
                "accuracy_mean": float(np.mean(scores["test_accuracy"])),
                "f1_macro_mean": float(np.mean(scores["test_f1_macro"])),
            })
    return rows


def train_and_test_models_random_split_multiclass(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.25,
    random_state: int = 42
):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    results = {}
    for name, model in make_multiclass_models(random_state=random_state).items():
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        preds = model.predict(X_test)
        infer_time = time.perf_counter() - t1

        metrics = multiclass_metrics_detailed(y_test, preds)
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
):
    results = {}
    for name, model in make_multiclass_models(random_state=random_state).items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = multiclass_metrics_detailed(y_test, preds)
    return results


# =========================================================
# 13. INTERPRETABILITY / OOD SPLITS
# =========================================================
def block_permutation_importance(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    block_map: Dict[str, Tuple[int, int]],
    random_state: int = 42
):
    rng = np.random.default_rng(random_state)
    baseline = f1_score(y_test, model.predict(X_test), average="macro", zero_division=0)
    rows = []

    for b_name, (a, b) in block_map.items():
        X_perm = X_test.copy()
        X_perm[:, a:b] = X_perm[rng.permutation(len(X_perm)), a:b]
        score = f1_score(y_test, model.predict(X_perm), average="macro", zero_division=0)
        rows.append({"block": b_name, "drop": float(baseline - score)})

    return sorted(rows, key=lambda r: r["drop"], reverse=True)


def split_semiprime_balance_ood_strict(
    x_vals,
    X,
    y,
    metadata_rows,
    low_ratio_max=0.70,
    high_ratio_min=0.85,
    random_state=42
):
    rng = np.random.default_rng(random_state)
    meta_by_x = {int(row["x"]): row for row in metadata_rows}

    train_idx = []
    test_idx = []

    for i in np.where(y == 2)[0]:
        ratio = meta_by_x[int(x_vals[i])]["semiprime_balance_ratio"]
        if ratio <= low_ratio_max:
            train_idx.append(i)
        elif ratio >= high_ratio_min:
            test_idx.append(i)

    if not train_idx or not test_idx:
        train_idx = []
        test_idx = []
        cls2_idx = np.where(y == 2)[0]
        ratios = [meta_by_x[int(x_vals[i])]["semiprime_balance_ratio"] for i in cls2_idx]
        median_r = float(np.median(ratios))
        for i, r in zip(cls2_idx, ratios):
            if r <= median_r:
                train_idx.append(i)
            else:
                test_idx.append(i)

    for cls in [0, 1]:
        perm = rng.permutation(np.where(y == cls)[0])
        train_idx.extend(perm[:len(perm) // 2])
        test_idx.extend(perm[len(perm) // 2:])

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def split_semiprime_balance_ood_sequence(
    x_vals,
    X_seq,
    y,
    low_ratio_max=0.70,
    high_ratio_min=0.85,
    random_state=42
):
    rng = np.random.default_rng(random_state)
    train_idx = []
    test_idx = []

    for i in np.where(y == 2)[0]:
        ratio = semiprime_balance_ratio(int(x_vals[i]))
        if ratio is None:
            continue
        if ratio <= low_ratio_max:
            train_idx.append(i)
        elif ratio >= high_ratio_min:
            test_idx.append(i)

    if not train_idx or not test_idx:
        train_idx = []
        test_idx = []
        cls2_idx = np.where(y == 2)[0]
        ratios = [semiprime_balance_ratio(int(x_vals[i])) for i in cls2_idx]
        valid_ratios = [r for r in ratios if r is not None]
        median_r = float(np.median(valid_ratios))
        for i, r in zip(cls2_idx, ratios):
            if r is not None and r <= median_r:
                train_idx.append(i)
            elif r is not None:
                test_idx.append(i)

    for cls in [0, 1]:
        perm = rng.permutation(np.where(y == cls)[0])
        train_idx.extend(perm[:len(perm) // 2])
        test_idx.extend(perm[len(perm) // 2:])

    return X_seq[train_idx], X_seq[test_idx], y[train_idx], y[test_idx]


# =========================================================
# 14. DEEP HELPERS
# =========================================================
def scale_sequence_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    if X_train.ndim != 3 or X_test.ndim != 3:
        raise ValueError("Expected 3D arrays: (batch, seq, feat).")

    n_train, s_train, d_train = X_train.shape
    n_test, s_test, d_test = X_test.shape

    if s_train != s_test or d_train != d_test:
        raise ValueError("Train/test sequence shapes do not match.")

    scaler = StandardScaler()

    X_train_2d = X_train.reshape(-1, d_train)
    X_test_2d = X_test.reshape(-1, d_test)

    X_train_scaled = scaler.fit_transform(X_train_2d).reshape(n_train, s_train, d_train)
    X_test_scaled = scaler.transform(X_test_2d).reshape(n_test, s_test, d_test)

    return X_train_scaled.astype(np.float32), X_test_scaled.astype(np.float32), scaler


def build_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long)
    )
    generator = make_torch_generator(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def aggregate_metric_rows(rows: List[Dict[str, Any]], group_key: str = "model") -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row[group_key], []).append(row)

    summary = {}
    for key, subrows in grouped.items():
        f1_macros = np.array([r["f1_macro"] for r in subrows], dtype=np.float64)
        accuracies = np.array([r["accuracy"] for r in subrows], dtype=np.float64)
        f1_weighted = np.array([r["f1_weighted"] for r in subrows], dtype=np.float64)
        per_class = np.array([r["per_class_f1"] for r in subrows], dtype=np.float64)

        robustness_vals = None
        if "robustness_ratio_vs_id_same_seed" in subrows[0]:
            robustness_vals = np.array([r["robustness_ratio_vs_id_same_seed"] for r in subrows], dtype=np.float64)

        summary[key] = {
            "n_runs": len(subrows),
            "accuracy_mean": float(np.mean(accuracies)),
            "accuracy_std": float(np.std(accuracies)),
            "f1_macro_mean": float(np.mean(f1_macros)),
            "f1_macro_std": float(np.std(f1_macros)),
            "f1_weighted_mean": float(np.mean(f1_weighted)),
            "f1_weighted_std": float(np.std(f1_weighted)),
            "per_class_f1_mean": [float(x) for x in np.mean(per_class, axis=0)],
            "per_class_f1_std": [float(x) for x in np.std(per_class, axis=0)],
        }

        if robustness_vals is not None:
            summary[key]["robustness_mean"] = float(np.mean(robustness_vals))
            summary[key]["robustness_std"] = float(np.std(robustness_vals))

    return summary


def shuffled_copy_of_labels(y: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y_shuffled = y.copy()
    rng.shuffle(y_shuffled)
    return y_shuffled


# =========================================================
# 15. DEEP MODELS
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
        out = self.projector(x)
        out = self.transformer(out)
        out, _ = self.lstm(out)
        return self.classifier(out[:, -1, :])


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

        left = nodes.unsqueeze(2).expand(b, s, s, -1)
        right = nodes.unsqueeze(1).expand(b, s, s, -1)
        pairs = torch.cat([left, right], dim=-1)

        raw_msgs = self.edge_network(pairs)
        keep = self.demon_agent(pairs)

        active = raw_msgs * keep
        trash = raw_msgs * (1.0 - keep)

        pooled_active = (nodes + active.sum(dim=2)).mean(dim=1)
        pooled_trash = trash.sum(dim=2).mean(dim=1)

        return self.classifier(torch.cat([pooled_active, pooled_trash], dim=-1))


def train_and_evaluate_deep_model(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 20
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()

            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy().tolist())

    y_true = np.array(all_labels, dtype=np.int64)
    y_pred = np.array(all_preds, dtype=np.int64)

    metrics = multiclass_metrics_detailed(y_true, y_pred)
    metrics["model_name"] = model_name
    return metrics


# =========================================================
# 16. MAIN RUNNERS
# =========================================================
def run_classical_dynamic_audit(cfg: SpectralGeoConfig, zeta_zeros: np.ndarray, seeds: List[int]) -> Dict[str, Any]:
    print("\n" + "=" * 90)
    print("CLASSICAL DYNAMIC SPECTRAL / TOPOLOGY AUDIT")
    print("=" * 90)

    x_vals, X, y, metadata_rows = build_or_resume_static_dataset(cfg, zeta_zeros)

    cv_rows = cross_validate_models(X, y, seeds=seeds)
    write_rows_to_csv(CLASSICAL_ABLATION_CSV, cv_rows)

    results, _ = train_and_test_models_random_split_multiclass(X, y, random_state=seeds[0])

    print("\n--- IN-DISTRIBUTION RESULTS ---")
    classical_rows = []
    for name, metrics in results.items():
        print(
            f"{name:<12} | "
            f"acc={metrics['accuracy']:.4f} | "
            f"f1_macro={metrics['f1_macro']:.4f} | "
            f"per_class={metrics['per_class_f1']}"
        )
        classical_rows.append({
            "split": "ID",
            "model": name,
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "f1_weighted": metrics["f1_weighted"],
            "per_class_f1": metrics["per_class_f1"],
            "confusion_matrix": metrics["confusion_matrix"],
        })

    print("\n--- OOD SPLIT: BALANCE SHIFT ---")
    X_train_ood, X_test_ood, y_train_ood, y_test_ood = split_semiprime_balance_ood_strict(
        x_vals, X, y, metadata_rows, random_state=seeds[0]
    )
    ood_results = evaluate_given_split_multiclass(
        X_train_ood, X_test_ood, y_train_ood, y_test_ood, random_state=seeds[0]
    )

    for name, metrics in ood_results.items():
        print(
            f"{name:<12} [OOD] | "
            f"acc={metrics['accuracy']:.4f} | "
            f"f1_macro={metrics['f1_macro']:.4f} | "
            f"per_class={metrics['per_class_f1']}"
        )
        classical_rows.append({
            "split": "OOD",
            "model": name,
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "f1_weighted": metrics["f1_weighted"],
            "per_class_f1": metrics["per_class_f1"],
            "confusion_matrix": metrics["confusion_matrix"],
        })

    write_rows_to_csv(CLASSICAL_RESULTS_CSV, classical_rows)
    return {"status": "success", "rows": classical_rows}


def compare_deep_models_multiseed(
    cfg: SpectralSequenceConfig,
    zeta_zeros: np.ndarray,
    seeds: List[int],
    epochs: int = 20,
    batch_size: int = 16
) -> Dict[str, Any]:
    print("\n" + "=" * 90)
    print("DEEP SHOWDOWN v2.4: MULTI-SEED / LEAK-FREE / SANITY")
    print("=" * 90)

    x_vals, X_raw, y = build_or_resume_sequence_dataset(cfg, zeta_zeros)
    _, _, input_dim = X_raw.shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device.type.upper()}...")

    model_factories = {
        "Transformer-LSTM": lambda: TransformerLSTMHybrid(input_dim, 64, 3),
        "Godzilla (GTD)": lambda: GraphTransformerDemon(input_dim, 64, 3),
    }

    id_rows: List[Dict[str, Any]] = []
    ood_rows: List[Dict[str, Any]] = []
    sanity_rows: List[Dict[str, Any]] = []

    for seed in seeds:
        print(f"\n--- SEED {seed} ---")
        set_global_seed(seed)

        # -----------------------------
        # ID split
        # -----------------------------
        X_train_id_raw, X_test_id_raw, y_train_id, y_test_id = train_test_split(
            X_raw,
            y,
            test_size=0.25,
            stratify=y,
            random_state=seed
        )

        X_train_id, X_test_id, _ = scale_sequence_train_test(X_train_id_raw, X_test_id_raw)

        train_loader_id = build_dataloader(
            X_train_id, y_train_id, batch_size=batch_size, shuffle=True, seed=seed
        )
        test_loader_id = build_dataloader(
            X_test_id, y_test_id, batch_size=batch_size, shuffle=False, seed=seed
        )

        seed_id_results = {}

        for model_name, factory in model_factories.items():
            set_global_seed(seed)
            model = factory().to(device)

            metrics = train_and_evaluate_deep_model(
                model_name=model_name,
                model=model,
                train_loader=train_loader_id,
                test_loader=test_loader_id,
                device=device,
                epochs=epochs
            )

            row = {
                "seed": seed,
                "split": "ID",
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "f1_weighted": metrics["f1_weighted"],
                "per_class_f1": metrics["per_class_f1"],
                "confusion_matrix": metrics["confusion_matrix"],
            }
            id_rows.append(row)
            seed_id_results[model_name] = metrics["f1_macro"]

            print(
                f"{model_name:<24} [ID]  | "
                f"acc={metrics['accuracy']:.4f} | "
                f"f1_macro={metrics['f1_macro']:.4f} | "
                f"per_class={metrics['per_class_f1']}"
            )

        # -----------------------------
        # OOD split
        # -----------------------------
        X_train_ood_raw, X_test_ood_raw, y_train_ood, y_test_ood = split_semiprime_balance_ood_sequence(
            x_vals, X_raw, y, random_state=seed
        )

        X_train_ood, X_test_ood, _ = scale_sequence_train_test(X_train_ood_raw, X_test_ood_raw)

        train_loader_ood = build_dataloader(
            X_train_ood, y_train_ood, batch_size=batch_size, shuffle=True, seed=seed
        )
        test_loader_ood = build_dataloader(
            X_test_ood, y_test_ood, batch_size=batch_size, shuffle=False, seed=seed
        )

        for model_name, factory in model_factories.items():
            set_global_seed(seed)
            model = factory().to(device)

            metrics = train_and_evaluate_deep_model(
                model_name=model_name,
                model=model,
                train_loader=train_loader_ood,
                test_loader=test_loader_ood,
                device=device,
                epochs=epochs
            )

            robustness = metrics["f1_macro"] / max(1e-12, seed_id_results[model_name])

            row = {
                "seed": seed,
                "split": "OOD",
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "f1_weighted": metrics["f1_weighted"],
                "per_class_f1": metrics["per_class_f1"],
                "confusion_matrix": metrics["confusion_matrix"],
                "robustness_ratio_vs_id_same_seed": float(robustness),
            }
            ood_rows.append(row)

            print(
                f"{model_name:<24} [OOD] | "
                f"acc={metrics['accuracy']:.4f} | "
                f"f1_macro={metrics['f1_macro']:.4f} | "
                f"robustness={robustness:.2f} | "
                f"per_class={metrics['per_class_f1']}"
            )

        # -----------------------------
        # RANDOM-LABEL SANITY CHECK
        # -----------------------------
        y_train_sanity = shuffled_copy_of_labels(y_train_id, seed=seed + 1000)

        train_loader_sanity = build_dataloader(
            X_train_id, y_train_sanity, batch_size=batch_size, shuffle=True, seed=seed
        )
        test_loader_sanity = build_dataloader(
            X_test_id, y_test_id, batch_size=batch_size, shuffle=False, seed=seed
        )

        for model_name, factory in model_factories.items():
            set_global_seed(seed)
            model = factory().to(device)

            metrics = train_and_evaluate_deep_model(
                model_name=model_name,
                model=model,
                train_loader=train_loader_sanity,
                test_loader=test_loader_sanity,
                device=device,
                epochs=epochs
            )

            row = {
                "seed": seed,
                "split": "SANITY_RANDOM_LABELS",
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "f1_weighted": metrics["f1_weighted"],
                "per_class_f1": metrics["per_class_f1"],
                "confusion_matrix": metrics["confusion_matrix"],
            }
            sanity_rows.append(row)

            print(
                f"{model_name:<24} [SANITY] | "
                f"acc={metrics['accuracy']:.4f} | "
                f"f1_macro={metrics['f1_macro']:.4f}"
            )

    write_rows_to_csv(DEEP_MULTI_SEED_ID_CSV, id_rows)
    write_rows_to_csv(DEEP_MULTI_SEED_OOD_CSV, ood_rows)
    write_rows_to_csv(DEEP_SANITY_CSV, sanity_rows)

    id_summary = aggregate_metric_rows(id_rows, group_key="model")
    ood_summary = aggregate_metric_rows(ood_rows, group_key="model")
    sanity_summary = aggregate_metric_rows(sanity_rows, group_key="model")

    summary_payload = {
        "seeds": seeds,
        "epochs": epochs,
        "batch_size": batch_size,
        "id_summary": id_summary,
        "ood_summary": ood_summary,
        "sanity_summary": sanity_summary,
    }
    write_summary_json(DEEP_SUMMARY_JSON, summary_payload)

    print("\n" + "-" * 90)
    print("MULTI-SEED SUMMARY")
    print("-" * 90)

    for model_name in model_factories.keys():
        id_s = id_summary[model_name]
        ood_s = ood_summary[model_name]
        san_s = sanity_summary[model_name]

        print(f"\nModel: {model_name}")
        print(
            f"  ID     | f1_macro = {id_s['f1_macro_mean']:.4f} ± {id_s['f1_macro_std']:.4f} | "
            f"acc = {id_s['accuracy_mean']:.4f} ± {id_s['accuracy_std']:.4f}"
        )
        print(
            f"  OOD    | f1_macro = {ood_s['f1_macro_mean']:.4f} ± {ood_s['f1_macro_std']:.4f} | "
            f"acc = {ood_s['accuracy_mean']:.4f} ± {ood_s['accuracy_std']:.4f} | "
            f"robustness = {ood_s.get('robustness_mean', float('nan')):.4f} ± {ood_s.get('robustness_std', float('nan')):.4f}"
        )
        print(
            f"  SANITY | f1_macro = {san_s['f1_macro_mean']:.4f} ± {san_s['f1_macro_std']:.4f} | "
            f"acc = {san_s['accuracy_mean']:.4f} ± {san_s['accuracy_std']:.4f}"
        )
        print(f"  ID per-class mean F1:     {id_s['per_class_f1_mean']}")
        print(f"  OOD per-class mean F1:    {ood_s['per_class_f1_mean']}")
        print(f"  SANITY per-class mean F1: {san_s['per_class_f1_mean']}")

    return summary_payload


# =========================================================
# 17. MAIN
# =========================================================
if __name__ == "__main__":
    set_global_seed(42)

    ensure_zeros_exist(ZETA_CSV)
    all_zeros = load_zeta_zeros_from_csv(ZETA_CSV)

    classical_cfg = SpectralGeoConfig(count_per_class=100)
    sequence_cfg = SpectralSequenceConfig(count_per_class=100)

    run_classical_dynamic_audit(
        classical_cfg,
        all_zeros,
        seeds=[42]
    )

    deep_summary = compare_deep_models_multiseed(
        cfg=sequence_cfg,
        zeta_zeros=all_zeros,
        seeds=[7, 11, 19, 42, 77],
        epochs=20,
        batch_size=16
    )

    write_summary_json(RUN_SUMMARY_JSON, {
        "classical_config": asdict(classical_cfg),
        "sequence_config": asdict(sequence_cfg),
        "deep_summary": deep_summary,
    })

    print("\n--- PIPELINE COMPLETE ---")
