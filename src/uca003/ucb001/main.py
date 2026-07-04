from __future__ import annotations

import csv
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Keep numerical backend threads small to avoid memory spikes when running
# multiple TruncatedSVD jobs concurrently.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


USECASE_ID = "uca003"
UCB_ID = "ucb001"

DEFAULT_STEP = 200
DEFAULT_MIN_DIM = 1000
DEFAULT_THRESHOLD = 0.95
DEFAULT_MAX_WORKERS = 4
HARD_MAX_WORKERS = 1

REPORT_HEADER = [
    "original_dim",
    "reduced_dim",
    "compression_ratio",
    "reconstruction_rate",
    "information_loss_rate",
    "threshold_ok",
    "is_min_dim_over_threshold",
    "singular_value_sum",
    "top_singular_value",
]


def _project_root() -> Path:
    # src/uca003/ucb001/main.py -> project root
    return Path(__file__).resolve().parents[3]


def _src_root() -> Path:
    return _project_root() / "src"


def _ensure_src_on_sys_path() -> None:
    src = str(_src_root())
    if src not in sys.path:
        sys.path.insert(0, src)


def _create_context():
    _ensure_src_on_sys_path()
    from project_context.project_context import ProjectContext

    return ProjectContext()


def get_chabsa_vector_relpath() -> str:
    return "/data/uca002/ucb002/chabsa_vector.npz"


def get_chabsa_vector_path() -> str:
    rel = get_chabsa_vector_relpath().lstrip("/")
    return str(_project_root() / rel)


def get_truncated_svd_report_relpath() -> str:
    return "/data/uca003/ucb001/truncated_svd_report.tsv"


def get_truncated_svd_report_path() -> str:
    rel = get_truncated_svd_report_relpath().lstrip("/")
    return str(_project_root() / rel)


def get_truncated_svd_model_relpath(reduced_dim: int) -> str:
    return f"/data/uca003/ucb001/truncated_svd_model_{reduced_dim}.pkl"


def get_truncated_svd_model_path(reduced_dim: int) -> str:
    rel = get_truncated_svd_model_relpath(reduced_dim).lstrip("/")
    return str(_project_root() / rel)


def _cleanup_outputs() -> None:
    report_path = Path(get_truncated_svd_report_path())
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.exists():
        report_path.unlink()

    model_dir = report_path.parent
    for existing_model in model_dir.glob("truncated_svd_model_*.pkl"):
        existing_model.unlink()


def _load_input_matrix() -> sparse.csr_matrix:
    vector_path = Path(get_chabsa_vector_path())
    if not vector_path.exists():
        raise FileNotFoundError(f"Input vector file not found: {vector_path}")

    matrix = sparse.load_npz(vector_path)
    if not sparse.isspmatrix_csr(matrix):
        matrix = matrix.tocsr()
    return matrix


def _max_valid_components(x: sparse.csr_matrix) -> int:
    # TruncatedSVD requires n_components to be smaller than both sample and feature counts.
    row_count, col_count = _get_shape(x)
    return max(1, min(row_count, col_count) - 1)


def _get_shape(x: sparse.csr_matrix) -> tuple[int, int]:
    shape = x.shape
    if shape is None:
        raise ValueError("Input matrix shape is unavailable")
    return int(shape[0]), int(shape[1])


def _build_reduced_dims(
    *,
    original_dim: int,
    max_components: int,
    step: int,
    min_dim: int,
) -> list[int]:
    if step <= 0:
        raise ValueError(f"step must be positive: {step}")

    start = min(original_dim - step, max_components)
    if start < 1:
        return [1]

    dims: list[int] = []
    dim = start
    while dim >= min_dim:
        dims.append(dim)
        dim -= step

    if not dims:
        dims.append(start)

    return sorted(set(dims), reverse=True)


def _fit_and_save_one(
    *,
    x: sparse.csr_matrix,
    original_dim: int,
    reduced_dim: int,
    threshold: float,
) -> dict[str, int | float | bool]:
    svd = TruncatedSVD(n_components=reduced_dim, random_state=42)
    svd.fit(x)

    reconstruction_rate = float(np.sum(svd.explained_variance_ratio_))
    singular_values = np.asarray(svd.singular_values_, dtype=np.float64)
    singular_value_sum = float(np.sum(singular_values))
    top_singular_value = float(np.max(singular_values)) if singular_values.size > 0 else 0.0

    model_path = Path(get_truncated_svd_model_path(reduced_dim))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as f:
        pickle.dump(svd, f)

    return {
        "original_dim": int(original_dim),
        "reduced_dim": int(reduced_dim),
        "compression_ratio": float(reduced_dim / original_dim),
        "reconstruction_rate": reconstruction_rate,
        "information_loss_rate": float(1.0 - reconstruction_rate),
        "threshold_ok": bool(reconstruction_rate >= threshold),
        "is_min_dim_over_threshold": False,
        "singular_value_sum": singular_value_sum,
        "top_singular_value": top_singular_value,
    }


def _format_float(value: float) -> str:
    return f"{value:.4f}"


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def _write_report(rows: list[dict[str, int | float | bool]]) -> str:
    report_path = Path(get_truncated_svd_report_path())
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(REPORT_HEADER)
        for row in rows:
            writer.writerow(
                [
                    int(row["original_dim"]),
                    int(row["reduced_dim"]),
                    _format_float(float(row["compression_ratio"])),
                    _format_float(float(row["reconstruction_rate"])),
                    _format_float(float(row["information_loss_rate"])),
                    _format_bool(bool(row["threshold_ok"])),
                    _format_bool(bool(row["is_min_dim_over_threshold"])),
                    _format_float(float(row["singular_value_sum"])),
                    _format_float(float(row["top_singular_value"])),
                ]
            )

    return str(report_path)


def run_truncated_svd_experiments(
    *,
    step: int = DEFAULT_STEP,
    min_dim: int = DEFAULT_MIN_DIM,
    threshold: float = DEFAULT_THRESHOLD,
    max_workers: int = DEFAULT_MAX_WORKERS,
    ctx=None,
) -> dict[str, str | int]:
    if ctx is None:
        ctx = _create_context()

    ctx.info(f"[{USECASE_ID}/{UCB_ID}] run_truncated_svd_experiments() start")
    _cleanup_outputs()

    x = _load_input_matrix()
    _, original_dim = _get_shape(x)
    max_components = _max_valid_components(x)
    reduced_dims = _build_reduced_dims(
        original_dim=original_dim,
        max_components=max_components,
        step=step,
        min_dim=min_dim,
    )

    if not reduced_dims:
        raise ValueError("No reduced dimensions generated for experiment")

    worker_count = max(1, min(max_workers, HARD_MAX_WORKERS, len(reduced_dims)))
    if max_workers > HARD_MAX_WORKERS:
        ctx.info(
            f"[{USECASE_ID}/{UCB_ID}] max_workers={max_workers} is capped to {HARD_MAX_WORKERS}"
        )
    rows: list[dict[str, int | float | bool]] = []

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                _fit_and_save_one,
                x=x,
                original_dim=original_dim,
                reduced_dim=reduced_dim,
                threshold=threshold,
            )
            for reduced_dim in reduced_dims
        ]
        for future in as_completed(futures):
            rows.append(future.result())

    rows.sort(key=lambda row: int(row["reduced_dim"]), reverse=True)

    threshold_dims = [int(row["reduced_dim"]) for row in rows if bool(row["threshold_ok"])]
    min_dim_over_threshold = min(threshold_dims) if threshold_dims else None
    if min_dim_over_threshold is not None:
        for row in rows:
            row["is_min_dim_over_threshold"] = int(row["reduced_dim"]) == min_dim_over_threshold

    report_path = _write_report(rows)

    ctx.info(
        f"[{USECASE_ID}/{UCB_ID}] completed patterns={len(rows)}, "
        f"min_dim_over_threshold={min_dim_over_threshold}"
    )

    return {
        "report": report_path,
        "patterns": len(rows),
        "models": len(rows),
    }


def main() -> None:
    _ensure_src_on_sys_path()
    from project_context.project_context import ProjectContext

    ctx = ProjectContext()
    option = sys.argv[1] if len(sys.argv) > 1 else ""
    ctx.info(f"[{USECASE_ID}/{UCB_ID}] start option='{option or 'default'}'")

    try:
        if option in ("", "run"):
            result = run_truncated_svd_experiments(ctx=ctx)
            ctx.info(f"[{USECASE_ID}/{UCB_ID}] result: {result}")
            return

        raise ValueError(f"Unknown option: {option}")
    finally:
        ctx.info(f"[{USECASE_ID}/{UCB_ID}] end")


if __name__ == "__main__":
    main()

"""
python src/uca003/ucb001/main.py
python src/uca003/ucb001/main.py run
"""
