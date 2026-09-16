# -*- coding: utf-8 -*-
"""All-TR within-DMN ISFC with 100,000 circular-shift permutations.

The observed ISFC estimator is identical to the existing all-TR DMN analysis:
for every participant, directed correlations with the unshifted leave-one-out
(LOO) mean are Fisher-z transformed and averaged across directions to form a
symmetric participant-level edge estimate. Group r is tanh(mean participant z).

Only the inferential step changes. For each Monte Carlo draw, every left-out
participant receives one independently sampled circular lag from 0..T-1. The
same lag is applied to all 20 ROIs and both directions for that participant,
while the participant-specific LOO mean stays unshifted. The group mean
symmetric Fisher-z is the permutation statistic. Two-sided add-one p-values
are Bonferroni-adjusted across the 190 unique within-DMN pairs per dataset.

Observed t-test outputs are not overwritten; all permutation outputs are saved
under versioned cshiftperm100000 directories.

Methodological reference
-----------------------
Simony, E., Honey, C. J., Chen, J., Lositsky, O., Yeshurun, Y., Wiesel, A., & Hasson, U. Dynamic reconfiguration of the default mode network during narrative comprehension. Nature Communications 7, 12141 (2016).
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import os
import platform
import re
import time
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests




# =============================================================================
# SELF-CONTAINED DATA/ATLAS/PLOTTING DEFINITIONS
# =============================================================================

ALPHA = 0.05
DMN_ROIS_0B = [
    2, 4, 5, 12, 48, 49, 84, 85, 89, 95,
    114, 133, 137, 140, 202, 221, 222, 224, 226, 238,
]
DMN_NODE_COLOR = "#6B7280"
DMN_EDGE_CMAP = "viridis"
EDGE_COLOR_ABS_R_MAX = 0.40
REGION_FULL_NAMES = {
    "BA8": "BA8", "BA10": "BA10", "BA11": "BA11", "BA19": "BA19",
    "BA23": "BA23", "BA30": "BA30", "BA31": "BA31", "BA36": "BA36",
    "BA39": "BA39", "Cerebellum": "Cerebellum",
}


def _find_project_root(start: Path) -> Path:
    start = start.resolve()
    for parent in [start, *start.parents]:
        if (parent / "data").exists() or (parent / "results").exists():
            return parent
    return start.parent


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _find_project_root(SCRIPT_DIR)
DATA_ROOT = Path(os.environ.get("VISC_DATA_DIR", PROJECT_ROOT / "data")).expanduser()
RESULTS_ROOT = Path(os.environ.get("VISC_RESULTS_DIR", PROJECT_ROOT / "results")).expanduser()
SHEN268_CSV_PATH = Path(
    os.environ.get("SHEN268_CSV_PATH", DATA_ROOT / "shen268.csv")
).expanduser()
STUDY1_TPDM_ROOT = Path(
    os.environ.get("STUDY1_TPDM_BRAIN_DIR", DATA_ROOT / "brain" / "study1_tpdm")
).expanduser()
STUDY1_BD_ROOT = Path(
    os.environ.get("STUDY1_BD_BRAIN_DIR", DATA_ROOT / "brain" / "study1_movieBD")
).expanduser()
STUDY2_ROOT = Path(
    os.environ.get("STUDY2_HORROR_BRAIN_DIR", DATA_ROOT / "brain" / "study2_horror")
).expanduser()

# Participant inclusion is supplied only through a local configuration.
# No final sample counts, participant IDs, exclusion lists, or cross-dataset
# membership/overlap patterns are distributed with this public release.
PARTICIPANT_CONFIG_PATH_ENV = "VISC_PARTICIPANT_CONFIG_PATH"
PARTICIPANT_CONFIG_JSON_ENV = "VISC_PARTICIPANT_CONFIG_JSON"


def _load_dataset_ids_config() -> dict[str, list[str]]:
    import json
    raw = os.environ.get(PARTICIPANT_CONFIG_JSON_ENV, "").strip()
    path_value = os.environ.get(PARTICIPANT_CONFIG_PATH_ENV, "").strip()
    if raw and path_value:
        raise ValueError(
            f"Set only one of {PARTICIPANT_CONFIG_JSON_ENV} or "
            f"{PARTICIPANT_CONFIG_PATH_ENV}."
        )
    if path_value:
        path = Path(path_value).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                "Participant configuration was not found. Keep it outside "
                "the public repository or list it in .gitignore."
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
    elif raw:
        payload = json.loads(raw)
    else:
        raise FileNotFoundError(
            "This analysis requires a dataset-specific inclusion list. "
            f"Set {PARTICIPANT_CONFIG_PATH_ENV} or {PARTICIPANT_CONFIG_JSON_ENV}."
        )
    section = payload.get("isfc", payload) if isinstance(payload, dict) else None
    if not isinstance(section, dict):
        raise TypeError("ISFC configuration must be a JSON object.")
    mapping = section.get("dataset_ids")
    if not isinstance(mapping, dict):
        raise KeyError(
            "ISFC configuration must contain isfc.dataset_ids (or dataset_ids)."
        )
    output: dict[str, list[str]] = {}
    for key, values in mapping.items():
        if not isinstance(values, list):
            raise TypeError(f"dataset_ids[{key!r}] must be a list.")
        normalized = [_subject_id_from_config(value) for value in values]
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"dataset_ids[{key!r}] contains duplicate IDs.")
        output[str(key)] = normalized
    return output


def _subject_id_from_config(value) -> str:
    text = str(value).strip()
    match = re.fullmatch(r"(?:sub-)?([A-Za-z0-9]+)", text, flags=re.IGNORECASE)
    if match is None:
        raise ValueError("Participant IDs must be simple alphanumeric labels.")
    suffix = match.group(1).lower()
    if suffix.isdigit():
        suffix = suffix.zfill(2)
    return f"sub-{suffix}"


def _configured_ids_for_dataset(dataset: str) -> list[str]:
    mapping = _load_dataset_ids_config()
    if dataset not in mapping:
        raise KeyError(
            f"ISFC configuration does not define an inclusion list for {dataset!r}."
        )
    ids = mapping[dataset]
    if len(ids) < 3:
        raise ValueError(f"{dataset}: at least three configured participants are required.")
    return ids


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    root: Path
    session_token: str


DATASET_SPECS = {
    "movieTP": DatasetSpec("movieTP", "TP", STUDY1_TPDM_ROOT, "movieTP"),
    "movieDM": DatasetSpec("movieDM", "DM", STUDY1_TPDM_ROOT, "movieDM"),
    "movieBD": DatasetSpec("movieBD", "BD", STUDY1_BD_ROOT, "movieBD"),
    "H1": DatasetSpec("H1", "H1", STUDY2_ROOT, "horror01"),
    "H2": DatasetSpec("H2", "H2", STUDY2_ROOT, "horror02"),
}


def fisher_z(r):
    values = np.asarray(r, dtype=np.float64)
    return np.arctanh(np.clip(values, -0.999999, 0.999999))


def validate_dmn_rois(rois: Sequence[int]) -> list[int]:
    rois = [int(x) for x in rois]
    if len(rois) != 20 or len(set(rois)) != 20:
        raise ValueError(f"Expected 20 unique DMN ROI indices; got {rois}")
    if min(rois) < 0 or max(rois) >= 268:
        raise ValueError(f"DMN ROI indices must be 0..267; got range {min(rois)}..{max(rois)}")
    return rois


def _subject_id(path: Path) -> str:
    match = re.search(r"(sub-[A-Za-z0-9]+)", str(path), flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Cannot parse subject ID from {path}")
    raw = match.group(1).lower()
    suffix = raw.split("sub-", 1)[1]
    if suffix.isdigit():
        suffix = suffix.zfill(2)
    return f"sub-{suffix}"


def _discover_files_for_key(dataset: str) -> dict[str, Path]:
    spec = DATASET_SPECS[dataset]
    if dataset in {"movieTP", "movieDM"}:
        pattern = f"sub-*/func/*_{spec.session_token}_6mmblur-regression_shen268.csv"
        files = sorted(spec.root.glob(pattern))
    elif dataset == "movieBD":
        files = sorted(spec.root.glob("sub-*_268ROI_shen268.csv"))
        if not files:
            files = sorted(spec.root.rglob("sub-*_268ROI_shen268.csv"))
    else:
        patterns = [
            f"sub-*_{spec.session_token}_6mmblur-regression_shen268.csv",
            f"sub-*{spec.session_token}*shen268.csv",
        ]
        found: list[Path] = []
        for pattern in patterns:
            found.extend(spec.root.rglob(pattern))
        files = sorted(set(found))
    files = [p for p in files if not p.name.startswith("._")]
    if len(files) < 2:
        raise FileNotFoundError(
            f"Need at least two neural files for {dataset}; found {len(files)} under {spec.root}"
        )
    mapping: dict[str, Path] = {}
    for path in files:
        sid = _subject_id(path)
        if sid in mapping:
            raise ValueError(f"Duplicate participant file detected for {dataset}; IDs/paths withheld.")
        mapping[sid] = path
    return mapping


def _final_dataset_files(dataset: str) -> list[tuple[str, Path]]:
    files = _discover_files_for_key(dataset)
    requested = _configured_ids_for_dataset(dataset)
    missing = [sid for sid in requested if sid not in files]
    if missing:
        raise FileNotFoundError(
            f"{dataset}: {len(missing)} configured participants are absent from the local input files. "
            "IDs are not printed by the public-release code."
        )
    return [(sid, files[sid]) for sid in requested]


def _load_atlas() -> pd.DataFrame:
    if not SHEN268_CSV_PATH.exists():
        raise FileNotFoundError(f"Shen268 metadata not found: {SHEN268_CSV_PATH}")
    atlas = pd.read_csv(SHEN268_CSV_PATH).copy()
    if len(atlas) < 268:
        raise ValueError(f"Expected at least 268 atlas rows; found {len(atlas)}")
    if "NTW" in atlas.columns:
        labels = atlas.iloc[DMN_ROIS_0B]["NTW"].astype(str).str.strip().str.casefold()
        if not labels.eq("dmn").all():
            bad = [DMN_ROIS_0B[i] for i, ok in enumerate(labels.eq("dmn")) if not ok]
            raise ValueError(f"Hard-coded DMN ROI indices are not all NTW==DMN in this atlas: {bad}")
    return atlas


def _resolve_roi_columns(columns: Sequence[str], rois_0b: Sequence[int]) -> list[str]:
    columns = [str(c) for c in columns]
    column_set = set(columns)
    candidates = [
        [str(i) for i in rois_0b],
        [str(i + 1) for i in rois_0b],
    ]
    # Use the global column range to break the otherwise common 0-/1-based tie.
    if "0" in column_set and "268" not in column_set:
        candidates = [candidates[0], candidates[1]]
    elif "268" in column_set and "0" not in column_set:
        candidates = [candidates[1], candidates[0]]
    best = max(candidates, key=lambda xs: sum(x in column_set for x in xs))
    missing = [x for x in best if x not in column_set]
    if missing:
        raise ValueError(
            f"Could not resolve all 20 DMN ROI columns. Missing {missing}; available examples={columns[:12]}"
        )
    return best


def load_dataset_dmn(dataset: str) -> tuple[np.ndarray, list[str], list[Path]]:
    if dataset not in DATASET_SPECS:
        raise KeyError(dataset)
    entries = _final_dataset_files(dataset)
    first = pd.read_csv(entries[0][1], nrows=5)
    first.columns = first.columns.astype(str)
    roi_columns = _resolve_roi_columns(first.columns, validate_dmn_rois(DMN_ROIS_0B))

    arrays: list[np.ndarray] = []
    subjects: list[str] = []
    paths: list[Path] = []
    n_tr: int | None = None
    for sid, path in entries:
        frame = pd.read_csv(path)
        frame.columns = frame.columns.astype(str)
        missing = [c for c in roi_columns if c not in frame.columns]
        if missing:
            raise ValueError(f"{path}: missing DMN columns {missing}")
        values = frame[roi_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        if n_tr is None:
            n_tr = values.shape[0]
        elif values.shape[0] != n_tr:
            raise ValueError(
                f"{dataset}: unequal TR counts ({entries[0][1]}={n_tr}, {path}={values.shape[0]})"
            )
        arrays.append(values)
        subjects.append(sid)
        paths.append(path)
    data = np.stack(arrays, axis=0)
    print(f"[LOAD] {dataset}: subjects={data.shape[0]}, TRs={data.shape[1]}, DMN nodes={data.shape[2]}")
    return data, subjects, paths


def _atlas_value(row: pd.Series, candidates: Sequence[str], default: str) -> str:
    for name in candidates:
        if name in row.index and pd.notna(row[name]) and str(row[name]).strip():
            return str(row[name]).strip()
    return default


def build_dmn_nodes() -> pd.DataFrame:
    atlas = _load_atlas()
    rows = []
    for roi in validate_dmn_rois(DMN_ROIS_0B):
        row = atlas.iloc[roi]
        ba = _atlas_value(row, ["BA", "Region", "region"], "DMN")
        hemi = _atlas_value(row, ["Hemisphere", "hemisphere", "Hemi", "hemi"], "")
        area = _atlas_value(row, ["Area", "area"], ba)
        pretty = f"{hemi + ' ' if hemi else ''}{ba} (ROI {roi})"
        rows.append(
            {
                "ID": int(roi),
                "Pretty": pretty,
                "Network": "DMN",
                "Network_full_name": "Default mode network",
                "color": DMN_NODE_COLOR,
                "hemi": hemi,
                "area": area,
                "BA": ba,
                "full_name": REGION_FULL_NAMES.get(ba, ba),
            }
        )
    return pd.DataFrame(rows)


def save_static_pdf(plot, path: Path) -> None:
    import holoviews as hv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hv.save(plot, str(path), backend="matplotlib", fmt="pdf")


# =============================================================================
# 0. CONFIGURATION
# =============================================================================


RUN_DATASETS = ("movieTP", "movieDM", "movieBD", "H1", "H2")

OUT_DIR = RESULTS_ROOT / "isfc_dmn_alltr_allfive_cshiftperm100000"
PDF_OUT_DIR = RESULTS_ROOT / "isfc_dmn_alltr_allfive_cshiftperm100000" / "pdf"
SOURCE_DATA_OUT_DIR = RESULTS_ROOT / "isfc_dmn_alltr_allfive_cshiftperm100000" / "source_data"

N_PERMUTATIONS = 100_000
PERMUTATION_BATCH_SIZE = 128
PERMUTATION_MASTER_SEED = 42
PERMUTATION_TAIL = "two-sided"
P_ADJUST_METHOD = "bonferroni"

# Fixed integer keys make each dataset's random stream independent of run order.
DATASET_SEED_INDEX = {
    "movieTP": 0,
    "movieDM": 1,
    "movieBD": 2,
    "H1": 3,
    "H2": 4,
}

INFERENCE_METHOD = "circular_shift_permutation"
PERMUTATION_SCHEME = (
    "BrainIAK-style LOO time shift: independently sample one uniform lag "
    "from 0..T-1 per participant and draw; apply the same lag to all ROIs "
    "and both edge directions of the left-out participant; keep that "
    "participant's LOO mean unshifted"
)
P_VALUE_FORMULA = (
    "(1 + count(abs(null_mean_z) >= abs(observed_mean_z))) / "
    "(n_permutations + 1)"
)


# =============================================================================
# 1. OBSERVED ISFC (UNCHANGED ESTIMATOR; NO PARAMETRIC TEST)
# =============================================================================


def _columnwise_correlation_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pairwise-complete corr(x[:, i], y[:, j]) for all column pairs."""
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError(
            f"Correlation inputs have incompatible shapes: {x.shape}, {y.shape}"
        )

    finite_x = np.isfinite(x)
    finite_y = np.isfinite(y)
    x_zero = np.where(finite_x, x, 0.0)
    y_zero = np.where(finite_y, y, 0.0)
    valid_x = finite_x.astype(np.float64)
    valid_y = finite_y.astype(np.float64)

    n_pair = valid_x.T @ valid_y
    sum_x = x_zero.T @ valid_y
    sum_y = valid_x.T @ y_zero
    sum_x_squared = (x_zero * x_zero).T @ valid_y
    sum_y_squared = valid_x.T @ (y_zero * y_zero)
    sum_xy = x_zero.T @ y_zero

    correction_x = np.zeros_like(sum_x)
    correction_y = np.zeros_like(sum_y)
    valid_n = n_pair >= 3
    np.divide(sum_x * sum_x, n_pair, out=correction_x, where=valid_n)
    np.divide(sum_y * sum_y, n_pair, out=correction_y, where=valid_n)
    centered_ss_x = np.maximum(sum_x_squared - correction_x, 0.0)
    centered_ss_y = np.maximum(sum_y_squared - correction_y, 0.0)
    mean_product = np.zeros_like(sum_xy)
    np.divide(sum_x * sum_y, n_pair, out=mean_product, where=valid_n)
    centered_cross_product = sum_xy - mean_product

    denominator = np.sqrt(centered_ss_x * centered_ss_y)
    output = np.full(sum_xy.shape, np.nan, dtype=np.float64)
    usable = valid_n & (denominator > 1e-12)
    np.divide(centered_cross_product, denominator, out=output, where=usable)
    return np.clip(output, -1.0, 1.0)


def _loo_mean_for_subject(
    x: np.ndarray,
    finite_x: np.ndarray,
    all_subject_sum: np.ndarray,
    all_subject_count: np.ndarray,
    subject_index: int,
) -> np.ndarray:
    subject_finite = finite_x[subject_index]
    others_sum = all_subject_sum - np.where(
        subject_finite, x[subject_index], 0.0
    )
    others_count = all_subject_count - subject_finite.astype(np.int64)
    others_mean = np.full(x.shape[1:], np.nan, dtype=np.float64)
    np.divide(
        others_sum,
        others_count,
        out=others_mean,
        where=others_count > 0,
    )
    return others_mean


def compute_observed_within_dmn_isfc(
    timeseries: np.ndarray,
    *,
    progress_label: str = "dataset",
) -> dict[str, np.ndarray]:
    """Return unchanged participant-z and group-r ISFC estimates."""
    x = np.asarray(timeseries, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f"Expected subjects x TRs x nodes, got {x.shape}")
    n_subjects, n_tr, n_nodes = x.shape
    if n_subjects < 3 or n_tr < 3 or n_nodes < 2:
        raise ValueError(f"Insufficient subjects/TRs/nodes for ISFC: {x.shape}")
    if np.isinf(x).any():
        raise ValueError("timeseries contains Inf after loading")

    finite_x = np.isfinite(x)
    all_subject_sum = np.where(finite_x, x, 0.0).sum(axis=0)
    all_subject_count = finite_x.sum(axis=0)
    subject_z = np.full(
        (n_subjects, n_nodes, n_nodes), np.nan, dtype=np.float64
    )
    start = time.time()
    progress_every = max(1, n_subjects // 10)

    for subject_index in range(n_subjects):
        others_mean = _loo_mean_for_subject(
            x,
            finite_x,
            all_subject_sum,
            all_subject_count,
            subject_index,
        )
        directed_r = _columnwise_correlation_matrix(
            x[subject_index], others_mean
        )
        directed_z = fisher_z(directed_r)

        reverse_z = directed_z.T
        both_finite = np.isfinite(directed_z) & np.isfinite(reverse_z)
        symmetric_z = np.full_like(directed_z, np.nan)
        symmetric_z[both_finite] = (
            directed_z[both_finite] + reverse_z[both_finite]
        ) / 2.0
        np.fill_diagonal(symmetric_z, np.nan)
        subject_z[subject_index] = symmetric_z

        completed = subject_index + 1
        if completed == n_subjects or completed % progress_every == 0:
            print(
                f"[OBSERVED ISFC] {progress_label}: "
                f"{completed}/{n_subjects} subjects "
                f"({time.time() - start:.1f}s)",
                flush=True,
            )

    finite = np.isfinite(subject_z)
    n_valid = finite.sum(axis=0)
    z_sum = np.where(finite, subject_z, 0.0).sum(axis=0)
    mean_z = np.full((n_nodes, n_nodes), np.nan, dtype=np.float64)
    np.divide(z_sum, n_valid, out=mean_z, where=n_valid > 0)
    group_r = np.tanh(mean_z)
    np.fill_diagonal(group_r, np.nan)

    return {
        "group_r": group_r,
        "group_mean_fisher_z": mean_z,
        "subject_z": subject_z,
        "n_valid": n_valid,
    }


# =============================================================================
# 2. FFT-EXACT CIRCULAR-SHIFT NULL
# =============================================================================


def _center_complete_or_all_nan_columns(
    values: np.ndarray,
    *,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Center columns that are complete; allow only whole-column missingness."""
    finite = np.isfinite(values)
    complete = finite.all(axis=0)
    absent = ~finite.any(axis=0)
    partial = ~(complete | absent)
    if partial.any():
        indices = np.flatnonzero(partial).tolist()
        raise ValueError(
            f"{label}: partial-TR missingness is incompatible with the exact "
            f"FFT time-shift implementation; affected columns={indices}"
        )

    centered = np.zeros_like(values, dtype=np.float64)
    if complete.any():
        centered[:, complete] = (
            values[:, complete] - values[:, complete].mean(axis=0, keepdims=True)
        )
    return centered, complete


def _all_lag_directed_correlations(
    subject: np.ndarray,
    loo_mean: np.ndarray,
) -> np.ndarray:
    """Return corr(roll(subject, lag)[:, i], loo[:, j]) for every lag."""
    n_tr, n_nodes = subject.shape
    if loo_mean.shape != subject.shape:
        raise ValueError(
            f"subject and LOO mean shapes differ: {subject.shape}, {loo_mean.shape}"
        )

    x_centered, x_complete = _center_complete_or_all_nan_columns(
        subject, label="left-out subject"
    )
    y_centered, y_complete = _center_complete_or_all_nan_columns(
        loo_mean, label="LOO mean"
    )

    fft_x = np.fft.rfft(x_centered, axis=0)
    fft_y = np.fft.rfft(y_centered, axis=0)
    cross_products = np.fft.irfft(
        np.conj(fft_x)[:, :, None] * fft_y[:, None, :],
        n=n_tr,
        axis=0,
    )
    # cross_products[lag, i, j] equals roll(x[:, i], lag) @ y[:, j].
    x_norm = np.sqrt(np.sum(x_centered * x_centered, axis=0))
    y_norm = np.sqrt(np.sum(y_centered * y_centered, axis=0))
    denominator = x_norm[:, None] * y_norm[None, :]
    usable = (
        x_complete[:, None]
        & y_complete[None, :]
        & (denominator > 1e-12)
    )

    directed_r = np.full(
        (n_tr, n_nodes, n_nodes), np.nan, dtype=np.float64
    )
    np.divide(
        cross_products,
        denominator[None, :, :],
        out=directed_r,
        where=usable[None, :, :],
    )
    return np.clip(directed_r, -1.0, 1.0)


def _validate_fft_orientation() -> None:
    """Fail fast unless FFT lags exactly match positive numpy.roll lags."""
    rng = np.random.default_rng(1701)
    x = rng.normal(size=(11, 4))
    y = rng.normal(size=(11, 4))
    observed = _all_lag_directed_correlations(x, y)
    for lag in range(x.shape[0]):
        expected = _columnwise_correlation_matrix(np.roll(x, lag, axis=0), y)
        np.testing.assert_allclose(
            observed[lag], expected, rtol=1e-11, atol=1e-12
        )


def precompute_lagged_symmetric_edge_z(
    timeseries: np.ndarray,
    observed_subject_z: np.ndarray,
    *,
    progress_label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute participant symmetric z for all T circular lags and 190 edges."""
    x = np.asarray(timeseries, dtype=np.float64)
    n_subjects, n_tr, n_nodes = x.shape
    upper_i, upper_j = np.triu_indices(n_nodes, k=1)
    n_edges = len(upper_i)

    finite_x = np.isfinite(x)
    all_subject_sum = np.where(finite_x, x, 0.0).sum(axis=0)
    all_subject_count = finite_x.sum(axis=0)
    lagged_z = np.full(
        (n_subjects, n_tr, n_edges), np.nan, dtype=np.float64
    )
    start = time.time()
    progress_every = max(1, n_subjects // 10)

    for subject_index in range(n_subjects):
        loo_mean = _loo_mean_for_subject(
            x,
            finite_x,
            all_subject_sum,
            all_subject_count,
            subject_index,
        )
        directed_r = _all_lag_directed_correlations(
            x[subject_index], loo_mean
        )
        directed_z = fisher_z(directed_r)
        forward = directed_z[:, upper_i, upper_j]
        reverse = directed_z[:, upper_j, upper_i]
        both_finite = np.isfinite(forward) & np.isfinite(reverse)
        symmetric = np.full_like(forward, np.nan)
        symmetric[both_finite] = (
            forward[both_finite] + reverse[both_finite]
        ) / 2.0

        # Missingness must be invariant across circular lags.
        expected_mask = np.isfinite(symmetric[0])
        if not np.array_equal(
            np.isfinite(symmetric),
            np.broadcast_to(expected_mask, symmetric.shape),
        ):
            raise RuntimeError(
                f"{progress_label}/subject {subject_index}: edge-validity mask "
                "changes across circular lags"
            )
        lagged_z[subject_index] = symmetric

        completed = subject_index + 1
        if completed == n_subjects or completed % progress_every == 0:
            print(
                f"[FFT LAG TABLE] {progress_label}: "
                f"{completed}/{n_subjects} subjects "
                f"({time.time() - start:.1f}s)",
                flush=True,
            )

    np.testing.assert_allclose(
        lagged_z[:, 0, :],
        observed_subject_z[:, upper_i, upper_j],
        rtol=1e-9,
        atol=1e-10,
        equal_nan=True,
        err_msg=f"{progress_label}: lag-0 FFT values changed the observed ISFC",
    )
    return lagged_z, upper_i, upper_j


def _dataset_rng(dataset: str) -> tuple[np.random.Generator, int, int]:
    dataset_index = DATASET_SEED_INDEX[dataset]
    seed_sequence = np.random.SeedSequence(
        [PERMUTATION_MASTER_SEED, dataset_index]
    )
    derived_seed = int(seed_sequence.generate_state(1, dtype=np.uint64)[0])
    return np.random.default_rng(seed_sequence), dataset_index, derived_seed


def circular_shift_permutation_test(
    lagged_z: np.ndarray,
    observed_mean_z: np.ndarray,
    *,
    dataset: str,
    n_permutations: int = N_PERMUTATIONS,
    batch_size: int = PERMUTATION_BATCH_SIZE,
) -> dict[str, np.ndarray | int | str]:
    """Two-sided Monte Carlo test on group mean symmetric Fisher-z."""
    if lagged_z.ndim != 3:
        raise ValueError(
            f"lagged_z must be subjects x lags x edges, got {lagged_z.shape}"
        )
    n_subjects, n_tr, n_edges = lagged_z.shape
    observed_mean_z = np.asarray(observed_mean_z, dtype=np.float64)
    if observed_mean_z.shape != (n_edges,):
        raise ValueError(
            f"observed_mean_z expected {(n_edges,)}, got {observed_mean_z.shape}"
        )
    if n_permutations < 1 or batch_size < 1:
        raise ValueError("n_permutations and batch_size must be positive")

    valid_subject_edge = np.isfinite(lagged_z[:, 0, :])
    n_valid = valid_subject_edge.sum(axis=0)
    if (n_valid < 3).any():
        bad = np.flatnonzero(n_valid < 3).tolist()
        raise ValueError(f"Fewer than 3 valid subjects for edge indices {bad}")
    if not np.isfinite(observed_mean_z).all():
        raise ValueError("observed_mean_z contains nonfinite values")

    rng, dataset_index, derived_seed = _dataset_rng(dataset)
    shift_dtype = np.int16 if n_tr <= np.iinfo(np.int16).max else np.int32
    shifts = rng.integers(
        0,
        n_tr,
        size=(n_permutations, n_subjects),
        dtype=shift_dtype,
    )
    null_mean_z = np.empty((n_permutations, n_edges), dtype=np.float64)
    subject_indices = np.arange(n_subjects)[None, :]
    start_time = time.time()

    for start in range(0, n_permutations, batch_size):
        stop = min(start + batch_size, n_permutations)
        sampled = lagged_z[
            subject_indices,
            shifts[start:stop],
            :,
        ]
        sums = np.where(np.isfinite(sampled), sampled, 0.0).sum(axis=1)
        null_mean_z[start:stop] = sums / n_valid[None, :]

        if stop == n_permutations or stop % max(batch_size, 1000) == 0:
            print(
                f"[CIRCULAR SHIFT] {dataset}: {stop:,}/{n_permutations:,} "
                f"draws ({time.time() - start_time:.1f}s)",
                flush=True,
            )

    if not np.isfinite(null_mean_z).all():
        raise RuntimeError(f"{dataset}: nonfinite permutation statistics")

    exceedances = np.sum(
        np.abs(null_mean_z) >= np.abs(observed_mean_z)[None, :],
        axis=0,
    ).astype(np.int64)
    p_permutation = (exceedances + 1.0) / (n_permutations + 1.0)

    return {
        "p_permutation_two_sided": p_permutation,
        "n_exceedances": exceedances,
        "null_mean_z": null_mean_z.mean(axis=0),
        "null_sd_z": null_mean_z.std(axis=0, ddof=1),
        "null_q025_z": np.quantile(null_mean_z, 0.025, axis=0),
        "null_q975_z": np.quantile(null_mean_z, 0.975, axis=0),
        "n_permutations": int(n_permutations),
        "permutation_dataset_seed_index": int(dataset_index),
        "permutation_derived_seed_uint64": int(derived_seed),
        "permutation_bit_generator": rng.bit_generator.__class__.__name__,
    }


# =============================================================================
# 3. EDGE TABLES AND SOURCE DATA
# =============================================================================


def build_permutation_edge_table(
    observed: dict[str, np.ndarray],
    permutation: dict[str, np.ndarray | int | str],
    *,
    dataset: str,
    n_subjects: int,
    n_tr: int,
    alpha: float = ALPHA,
) -> pd.DataFrame:
    dmn_rois = validate_dmn_rois(DMN_ROIS_0B)
    upper_i, upper_j = np.triu_indices(len(dmn_rois), k=1)
    n_tests = len(upper_i)
    if n_tests != 190:
        raise RuntimeError(f"Expected 190 unique DMN edges, found {n_tests}")

    group_r = observed["group_r"][upper_i, upper_j]
    group_mean_z = observed["group_mean_fisher_z"][upper_i, upper_j]
    n_valid = observed["n_valid"][upper_i, upper_j].astype(int)
    p_permutation = np.asarray(
        permutation["p_permutation_two_sided"], dtype=np.float64
    )
    if not (
        np.isfinite(group_r).all()
        and np.isfinite(group_mean_z).all()
        and np.isfinite(p_permutation).all()
    ):
        raise ValueError(f"{dataset}: all 190 observed tests must be finite")

    rejected, p_bonferroni, _, _ = multipletests(
        p_permutation,
        alpha=alpha,
        method=P_ADJUST_METHOD,
    )
    display_name = DATASET_SPECS[dataset].display_name
    n_permutations = int(permutation["n_permutations"])
    frame = pd.DataFrame(
        {
            "dataset": dataset,
            "figure_label": display_name,
            "n_subjects": int(n_subjects),
            "n_tr_all": int(n_tr),
            "source": np.asarray(dmn_rois)[upper_i],
            "target": np.asarray(dmn_rois)[upper_j],
            "r": group_r,
            "abs_r": np.abs(group_r),
            "observed_mean_symmetric_fisher_z": group_mean_z,
            "n_valid_subjects": n_valid,
            "n_permutations": n_permutations,
            "permutation_exceedance_count": np.asarray(
                permutation["n_exceedances"], dtype=np.int64
            ),
            "p_permutation_two_sided": p_permutation,
            "p_bonferroni": p_bonferroni,
            "n_tests_bonferroni": n_tests,
            "alpha": float(alpha),
            "inference_method": INFERENCE_METHOD,
            "permutation_statistic": "mean_symmetric_fisher_z",
            "permutation_scheme": PERMUTATION_SCHEME,
            "permutation_tail": PERMUTATION_TAIL,
            "permutation_shift_min_tr": 0,
            "permutation_shift_max_tr": int(n_tr - 1),
            "permutation_includes_zero_lag": True,
            "permutation_same_shift_across_rois": True,
            "permutation_loo_mean_shifted": False,
            "permutation_master_seed": PERMUTATION_MASTER_SEED,
            "permutation_dataset_seed_index": int(
                permutation["permutation_dataset_seed_index"]
            ),
            "permutation_derived_seed_uint64": int(
                permutation["permutation_derived_seed_uint64"]
            ),
            "permutation_bit_generator": str(
                permutation["permutation_bit_generator"]
            ),
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "pvalue_formula": P_VALUE_FORMULA,
            "correction_method": P_ADJUST_METHOD,
            "null_mean_z": np.asarray(permutation["null_mean_z"], dtype=float),
            "null_sd_z": np.asarray(permutation["null_sd_z"], dtype=float),
            "null_q025_z": np.asarray(permutation["null_q025_z"], dtype=float),
            "null_q975_z": np.asarray(permutation["null_q975_z"], dtype=float),
            "is_significant_after_correction": rejected.astype(bool),
        }
    )

    expected_raw = (
        frame["permutation_exceedance_count"].to_numpy(float) + 1.0
    ) / (n_permutations + 1.0)
    np.testing.assert_allclose(
        frame["p_permutation_two_sided"], expected_raw, rtol=0, atol=1e-15
    )
    np.testing.assert_allclose(
        frame["p_bonferroni"],
        np.minimum(frame["p_permutation_two_sided"] * n_tests, 1.0),
        rtol=0,
        atol=1e-14,
    )
    return frame


def build_figure_source_data(
    edge_table: pd.DataFrame,
    nodes: pd.DataFrame,
) -> pd.DataFrame:
    """One row per edge displayed in the corresponding chord figure."""
    displayed = edge_table.loc[
        edge_table["is_significant_after_correction"]
    ].copy()
    if displayed.empty:
        return displayed

    node_lookup = nodes.set_index("ID", verify_integrity=True)
    source = displayed["source"].map(node_lookup["Pretty"])
    target = displayed["target"].map(node_lookup["Pretty"])
    source_data = displayed.copy()
    source_data.insert(6, "source_display_label", source)
    source_data.insert(
        7, "source_hemisphere", displayed["source"].map(node_lookup["hemi"])
    )
    source_data.insert(
        8, "source_area", displayed["source"].map(node_lookup["area"])
    )
    source_data.insert(
        9,
        "source_region_abbreviation",
        displayed["source"].map(node_lookup["BA"]),
    )
    source_data.insert(
        10,
        "source_region_full_name",
        displayed["source"].map(node_lookup["full_name"]),
    )
    source_data.insert(12, "target_display_label", target)
    source_data.insert(
        13, "target_hemisphere", displayed["target"].map(node_lookup["hemi"])
    )
    source_data.insert(
        14, "target_area", displayed["target"].map(node_lookup["area"])
    )
    source_data.insert(
        15,
        "target_region_abbreviation",
        displayed["target"].map(node_lookup["BA"]),
    )
    source_data.insert(
        16,
        "target_region_full_name",
        displayed["target"].map(node_lookup["full_name"]),
    )
    source_data.insert(
        20,
        "chord_value_abs_r",
        source_data["abs_r"].astype(float),
    )
    source_data.insert(
        21,
        "edge_sign",
        np.where(source_data["r"].ge(0), "positive", "negative"),
    )
    source_data["edge_colour_scale_min_abs_r"] = 0.0
    source_data["edge_colour_scale_max_abs_r"] = EDGE_COLOR_ABS_R_MAX
    if source_data.isna().any().any():
        missing = source_data.columns[source_data.isna().any()].tolist()
        raise ValueError(f"Source Data contains missing values: {missing}")
    return source_data.reset_index(drop=True)


# =============================================================================
# 4. INTERACTIVE AND STATIC CHORD PLOTS
# =============================================================================


def _plot_edge_columns(edge_table: pd.DataFrame) -> pd.DataFrame:
    plotted = edge_table.loc[
        edge_table["is_significant_after_correction"]
    ].copy()
    if plotted.empty:
        return plotted
    plotted["value"] = plotted["abs_r"]
    plotted["sign"] = np.where(plotted["r"].ge(0), "positive", "negative")
    return plotted.loc[
        :,
        [
            "source",
            "target",
            "value",
            "r",
            "abs_r",
            "sign",
            "observed_mean_symmetric_fisher_z",
            "p_permutation_two_sided",
            "p_bonferroni",
            "permutation_exceedance_count",
            "n_permutations",
            "n_valid_subjects",
            "is_significant_after_correction",
        ],
    ]


def _node_dataset_for_plotted(nodes: pd.DataFrame, plotted: pd.DataFrame):
    import holoviews as hv

    used_ids = sorted(set(plotted["source"]).union(plotted["target"]))
    plotted_nodes = nodes.loc[nodes["ID"].isin(used_ids)].copy()
    return hv.Dataset(
        plotted_nodes,
        kdims="ID",
        vdims=[
            "Pretty",
            "Network",
            "Network_full_name",
            "color",
            "hemi",
            "area",
            "BA",
            "full_name",
        ],
    )


def build_interactive_chord(
    edge_table: pd.DataFrame,
    nodes: pd.DataFrame,
    *,
    dataset: str,
    n_subjects: int,
    n_tr: int,
    alpha: float,
):
    import holoviews as hv
    from bokeh.models import HoverTool
    from holoviews import dim, opts

    hv.extension("bokeh")
    plotted = _plot_edge_columns(edge_table)
    if plotted.empty:
        return None, plotted

    node_dataset = _node_dataset_for_plotted(nodes, plotted)
    display_name = DATASET_SPECS[dataset].display_name
    title = (
        f"Within-DMN ISFC | {display_name} | all TRs | "
        f"N={n_subjects}, T={n_tr} | {len(plotted)} edges: "
        f"{N_PERMUTATIONS:,} circular-shift permutations + Bonferroni "
        f"(p_Bonf <= {alpha:.2f}; no r threshold) | "
        "Edge intensity proportional to |r|"
    )
    edge_hover = HoverTool(
        tooltips=[
            ("source", "@{source}"),
            ("target", "@{target}"),
            ("r", "@{r}{0.000000}"),
            ("sign", "@{sign}"),
            (
                "fisher-z",
                "@{observed_mean_symmetric_fisher_z}{0.000000}",
            ),
            ("p_bonferroni", "@{p_bonferroni}{0.000000}"),
            ("n_valid", "@{n_valid_subjects}"),
        ],
        name="DMN edge statistics",
    )
    chord = hv.Chord((plotted, node_dataset)).opts(
        opts.Chord(
            title=title,
            labels="Pretty",
            node_color=dim("color"),
            node_line_color="#7f3b32",
            node_line_width=0.6,
            node_size=12,
            edge_color=dim("abs_r"),
            edge_cmap=DMN_EDGE_CMAP,
            clim=(0.0, EDGE_COLOR_ABS_R_MAX),
            colorbar=True,
            colorbar_position="right",
            colorbar_opts={"title": "|r|"},
            edge_line_dash="solid",
            edge_alpha=0.90,
            edge_line_width=1.5,
            edge_hover_line_width=4,
            edge_hover_line_color=DMN_NODE_COLOR,
            label_text_font_size="8pt",
            width=950,
            height=950,
            tools=[edge_hover, "tap", "save"],
            inspection_policy="edges",
            show_legend=False,
        )
    )
    return chord, plotted


def build_static_chord(
    edge_table: pd.DataFrame,
    nodes: pd.DataFrame,
    *,
    dataset: str,
    n_subjects: int,
    n_tr: int,
    alpha: float,
):
    import holoviews as hv
    from holoviews import dim

    hv.extension("matplotlib")
    plotted = _plot_edge_columns(edge_table)
    if plotted.empty:
        return None

    node_dataset = _node_dataset_for_plotted(nodes, plotted)
    display_name = DATASET_SPECS[dataset].display_name
    title = (
        f"Within-DMN ISFC | {display_name} | all TRs | "
        f"N={n_subjects}, T={n_tr}\n"
        f"{len(plotted)} edges: {N_PERMUTATIONS:,} circular-shift "
        "permutations + Bonferroni\n"
        f"(p_Bonf <= {alpha:.2f}; no r threshold)\n"
        "Edge intensity proportional to |r|"
    )
    return hv.Chord((plotted, node_dataset)).opts(
        backend="matplotlib",
        title=title,
        labels="Pretty",
        node_color=DMN_NODE_COLOR,
        node_edgecolors="#7f3b32",
        node_linewidth=0.6,
        node_size=12,
        edge_color=dim("abs_r"),
        edge_cmap=DMN_EDGE_CMAP,
        clim=(0.0, EDGE_COLOR_ABS_R_MAX),
        colorbar=False,
        edge_alpha=0.90,
        edge_linewidth=1.0,
        fig_size=300,
        aspect="equal",
        xaxis=None,
        yaxis=None,
        show_frame=False,
    )


# =============================================================================
# 5. RUN AND SAVE VERSIONED OUTPUTS
# =============================================================================


def output_stem(dataset: str, *, alpha: float) -> str:
    return (
        f"{dataset}_DMN_only_allTR_cshiftperm{N_PERMUTATIONS}_"
        f"bonf{int(round(alpha * 100)):02d}"
    )


def run_one_dataset(
    dataset: str,
    *,
    nodes: pd.DataFrame,
    out_dir: Path = OUT_DIR,
    pdf_out_dir: Path = PDF_OUT_DIR,
    source_data_out_dir: Path = SOURCE_DATA_OUT_DIR,
    alpha: float = ALPHA,
) -> dict[str, object]:
    import holoviews as hv

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_out_dir.mkdir(parents=True, exist_ok=True)
    source_data_out_dir.mkdir(parents=True, exist_ok=True)

    timeseries, subjects, input_paths = load_dataset_dmn(dataset)
    observed = compute_observed_within_dmn_isfc(
        timeseries, progress_label=dataset
    )
    lagged_z, upper_i, upper_j = precompute_lagged_symmetric_edge_z(
        timeseries,
        observed["subject_z"],
        progress_label=dataset,
    )
    observed_mean_z = observed["group_mean_fisher_z"][upper_i, upper_j]
    permutation = circular_shift_permutation_test(
        lagged_z,
        observed_mean_z,
        dataset=dataset,
    )
    edge_table = build_permutation_edge_table(
        observed,
        permutation,
        dataset=dataset,
        n_subjects=len(subjects),
        n_tr=timeseries.shape[1],
        alpha=alpha,
    )
    figure_source_data = build_figure_source_data(edge_table, nodes)
    chord, plotted_edges = build_interactive_chord(
        edge_table,
        nodes,
        dataset=dataset,
        n_subjects=len(subjects),
        n_tr=timeseries.shape[1],
        alpha=alpha,
    )

    stem = output_stem(dataset, alpha=alpha)
    stats_path = out_dir / f"isfc_edges_{stem}_allTests.csv"
    source_data_path = source_data_out_dir / (
        f"Source_Data_{dataset}_DMN_ISFC_cshiftperm{N_PERMUTATIONS}_"
        f"bonf{int(round(alpha * 100)):02d}_significantOnly.csv"
    )
    edge_table.to_csv(stats_path, index=False)
    figure_source_data.to_csv(source_data_path, index=False)
    print(f"[SAVED] {stats_path}", flush=True)
    print(f"[SAVED] {source_data_path}", flush=True)

    html_path: Path | None = None
    pdf_path: Path | None = None
    if chord is None:
        print(
            f"[NO PLOT] {dataset}: no edge passed circular-shift "
            f"permutation + Bonferroni p <= {alpha:.2f}.",
            flush=True,
        )
    else:
        html_path = out_dir / f"chord_{stem}_significantOnly.html"
        hv.save(
            chord,
            html_path,
            backend="bokeh",
            resources="inline",
            title=(
                f"Within-DMN ISFC - {DATASET_SPECS[dataset].display_name} - "
                f"{N_PERMUTATIONS:,} circular-shift permutations"
            ),
        )
        static_chord = build_static_chord(
            edge_table,
            nodes,
            dataset=dataset,
            n_subjects=len(subjects),
            n_tr=timeseries.shape[1],
            alpha=alpha,
        )
        if static_chord is None:
            raise RuntimeError(
                "Static chord unexpectedly empty after HTML chord creation"
            )
        pdf_path = pdf_out_dir / f"chord_{stem}_significantOnly.pdf"
        save_static_pdf(static_chord, pdf_path)
        print(f"[SAVED] {html_path}", flush=True)
        print(f"[SAVED] {pdf_path}", flush=True)

    n_tests = len(edge_table)
    minimum_raw_p = 1.0 / (N_PERMUTATIONS + 1.0)
    minimum_bonferroni_p = min(n_tests * minimum_raw_p, 1.0)
    max_exceedances = int(
        np.floor(alpha * (N_PERMUTATIONS + 1.0) / n_tests - 1.0)
    )
    return {
        "dataset": dataset,
        "display_name": DATASET_SPECS[dataset].display_name,
        "n_subjects": len(subjects),
        "n_tr": timeseries.shape[1],
        "n_dmn_nodes": len(DMN_ROIS_0B),
        "n_unique_tests": n_tests,
        "n_significant": int(
            edge_table["is_significant_after_correction"].sum()
        ),
        "n_plotted": len(plotted_edges),
        "min_r": float(edge_table["r"].min()),
        "max_r": float(edge_table["r"].max()),
        "max_abs_r": float(edge_table["abs_r"].max()),
        "n_permutations": N_PERMUTATIONS,
        "permutation_tail": PERMUTATION_TAIL,
        "permutation_shift_min_tr": 0,
        "permutation_shift_max_tr": int(timeseries.shape[1] - 1),
        "permutation_master_seed": PERMUTATION_MASTER_SEED,
        "permutation_dataset_seed_index": int(
            permutation["permutation_dataset_seed_index"]
        ),
        "permutation_derived_seed_uint64": int(
            permutation["permutation_derived_seed_uint64"]
        ),
        "permutation_bit_generator": str(
            permutation["permutation_bit_generator"]
        ),
        "minimum_raw_permutation_p": minimum_raw_p,
        "minimum_bonferroni_p": minimum_bonferroni_p,
        "max_exceedances_for_bonferroni_significance": max_exceedances,
        "alpha": alpha,
        "inference_method": INFERENCE_METHOD,
        "correction_method": P_ADJUST_METHOD,
        "permutation_scheme": PERMUTATION_SCHEME,
        "pvalue_formula": P_VALUE_FORMULA,
        "edge_color_abs_r_max": EDGE_COLOR_ABS_R_MAX,
        "input_root": str(DATASET_SPECS[dataset].root),
        "first_input": str(input_paths[0]),
        "last_input": str(input_paths[-1]),
        "all_tests_csv": str(stats_path),
        "source_data_csv": str(source_data_path),
        "html": str(html_path) if html_path else "",
        "pdf": str(pdf_path) if pdf_path else "",
    }


def run_all(
    datasets: Iterable[str] = RUN_DATASETS,
    *,
    out_dir: Path = OUT_DIR,
    pdf_out_dir: Path = PDF_OUT_DIR,
    source_data_out_dir: Path = SOURCE_DATA_OUT_DIR,
    alpha: float = ALPHA,
) -> pd.DataFrame:
    datasets = tuple(datasets)
    unknown = [dataset for dataset in datasets if dataset not in DATASET_SPECS]
    if unknown:
        raise KeyError(f"Unknown dataset(s): {unknown}")

    _validate_fft_orientation()
    dmn_rois = validate_dmn_rois(DMN_ROIS_0B)
    n_tests = len(dmn_rois) * (len(dmn_rois) - 1) // 2
    print(
        f"[CONFIG] datasets={list(datasets)} | all TRs | "
        f"DMN nodes={len(dmn_rois)} | tests={n_tests} | "
        f"inference={N_PERMUTATIONS:,} two-sided circular shifts | "
        f"correction=Bonferroni alpha={alpha:.2f} | no r threshold",
        flush=True,
    )

    nodes = build_dmn_nodes()
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes_path = out_dir / "DMN_nodes_0based.csv"
    nodes.to_csv(nodes_path, index=False)
    print(f"[SAVED] {nodes_path}", flush=True)

    rows: list[dict[str, object]] = []
    for dataset in datasets:
        print(f"\n===== {dataset} =====", flush=True)
        rows.append(
            run_one_dataset(
                dataset,
                nodes=nodes,
                out_dir=out_dir,
                pdf_out_dir=pdf_out_dir,
                source_data_out_dir=source_data_out_dir,
                alpha=alpha,
            )
        )

    summary = pd.DataFrame(rows)
    summary_path = out_dir / (
        f"run_summary_DMN_only_allTR_cshiftperm{N_PERMUTATIONS}_"
        f"bonf{int(round(alpha * 100)):02d}_significantOnly.csv"
    )
    summary.to_csv(summary_path, index=False)
    print(f"\n[SUMMARY SAVED] {summary_path}", flush=True)
    print(summary.to_string(index=False), flush=True)
    return summary


if __name__ == "__main__":
    run_all()
