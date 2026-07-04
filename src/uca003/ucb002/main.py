from __future__ import annotations

import csv
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA


USECASE_ID = "uca003"
UCB_ID = "ucb002"
DEFAULT_THRESHOLD = 0.95


def _project_root() -> Path:
    # src/uca003/ucb002/main.py -> project root
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


def get_pca_report_relpath(reduced_dim: int) -> str:
    return f"/data/uca003/ucb002/pca_report_{reduced_dim}.tsv"


def get_pca_report_path(reduced_dim: int) -> str:
    rel = get_pca_report_relpath(reduced_dim).lstrip("/")
    return str(_project_root() / rel)


def get_pca_model_relpath(reduced_dim: int) -> str:
    return f"/data/uca003/ucb002/pca_model_{reduced_dim}.pkl"


def get_pca_model_path(reduced_dim: int) -> str:
    rel = get_pca_model_relpath(reduced_dim).lstrip("/")
    return str(_project_root() / rel)


def _load_dense_input_matrix() -> np.ndarray:
    vector_path = Path(get_chabsa_vector_path())
    if not vector_path.exists():
        raise FileNotFoundError(f"Input vector file not found: {vector_path}")

    x_sparse = sparse.load_npz(vector_path)
    return x_sparse.toarray()


def _format_float(value: float) -> str:
    return f"{value:.4f}"


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def _build_metrics(*, original_dim: int, reduced_dim: int, pca: PCA, threshold: float) -> dict[str, str]:
    explained_variance_rate = float(np.sum(pca.explained_variance_ratio_))
    information_loss_rate = float(1.0 - explained_variance_rate)
    threshold_ok = explained_variance_rate >= threshold

    singular_values = np.asarray(pca.singular_values_, dtype=np.float64)
    singular_value_sum = float(np.sum(singular_values))
    top_singular_value = float(np.max(singular_values)) if singular_values.size > 0 else 0.0

    return {
        "original_dim": str(int(original_dim)),
        "reduced_dim": str(int(reduced_dim)),
        "compression_ratio": _format_float(float(reduced_dim / original_dim)),
        "explained_variance_rate": _format_float(explained_variance_rate),
        "information_loss_rate": _format_float(information_loss_rate),
        "threshold_ok": _format_bool(threshold_ok),
        "is_min_dim_over_threshold": _format_bool(threshold_ok),
        "singular_value_sum": _format_float(singular_value_sum),
        "top_singular_value": _format_float(top_singular_value),
    }


def _write_report(reduced_dim: int, metrics: dict[str, str]) -> str:
    report_path = Path(get_pca_report_path(reduced_dim))
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if report_path.exists():
        report_path.unlink()

    with report_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for key, value in metrics.items():
            writer.writerow([key, value])

    return str(report_path)


def _save_model(reduced_dim: int, pca: PCA) -> str:
    model_path = Path(get_pca_model_path(reduced_dim))
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if model_path.exists():
        model_path.unlink()

    with model_path.open("wb") as f:
        pickle.dump(pca, f)

    return str(model_path)


def run_pca_compression(*, reduced_dim: int, threshold: float = DEFAULT_THRESHOLD, ctx=None) -> dict[str, str | int]:
    if ctx is None:
        ctx = _create_context()

    ctx.info(f"[{USECASE_ID}/{UCB_ID}] run_pca_compression() start reduced_dim={reduced_dim}")

    x_dense = _load_dense_input_matrix()
    original_dim = int(x_dense.shape[1])

    pca = PCA(n_components=reduced_dim, random_state=42)
    pca.fit(x_dense)

    metrics = _build_metrics(
        original_dim=original_dim,
        reduced_dim=reduced_dim,
        pca=pca,
        threshold=threshold,
    )

    report_path = _write_report(reduced_dim, metrics)
    model_path = _save_model(reduced_dim, pca)

    result: dict[str, str | int] = {
        "report": report_path,
        "model": model_path,
        "original_dim": original_dim,
        "reduced_dim": reduced_dim,
    }
    result.update(metrics)
    return result


def main() -> None:
    _ensure_src_on_sys_path()
    from project_context.project_context import ProjectContext

    ctx = ProjectContext()
    option = sys.argv[1] if len(sys.argv) > 1 else ""
    ctx.info(f"[{USECASE_ID}/{UCB_ID}] start option='{option or 'none'}'")

    with ctx.log_span(f"UCA/UCB002 run_pca_compression() option='{option or 'none'}'"):
        if option == "":
            raise ValueError("numeric option is required")

        try:
            reduced_dim = int(option)
        except ValueError as exc:
            raise ValueError(f"option must be integer reduced_dim: {option}") from exc

        result = run_pca_compression(reduced_dim=reduced_dim, ctx=ctx)
        ctx.info(f"[{USECASE_ID}/{UCB_ID}] result: {result}")
    ctx.info(f"[{USECASE_ID}/{UCB_ID}] end")



if __name__ == "__main__":
    main()

"""
python src/uca003/ucb002/main.py 600
python src/uca003/ucb002/main.py 800
python src/uca003/ucb002/main.py 1000
python src/uca003/ucb002/main.py 2000
python src/uca003/ucb002/main.py 3000
python src/uca003/ucb002/main.py 4000
python src/uca003/ucb002/main.py 5000
"""
