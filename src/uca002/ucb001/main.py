from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


USECASE_ID = "uca002"
UCB_ID = "ucb001"

HEADER = ["reciprocal", "sqrt", "power0.25", "log"]
CALC_TITLES = {
    1: "1/d",
    2: "1/sqrt(d)",
    3: "1/(d^0.25)",
    4: "1/log(d+1)",
}


def _project_root() -> Path:
    # src/uca002/ucb001/main.py -> project root
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


def get_distance_weight_tsv_relpath() -> str:
    return "/data/uca002/ucb001/distance_weight.tsv"


def get_distance_weight_tsv_path() -> str:
    rel = get_distance_weight_tsv_relpath().lstrip("/")
    return str(_project_root() / rel)


def get_distance_weight_all_png_relpath() -> str:
    return "/data/uca002/ucb001/distance_weight_all.png"


def get_distance_weight_all_png_path() -> str:
    rel = get_distance_weight_all_png_relpath().lstrip("/")
    return str(_project_root() / rel)


def get_distance_weight_plot_png_relpath(calc_number: int) -> str:
    return f"/data/uca002/ucb001/distance_weight_{calc_number}.png"


def get_distance_weight_plot_png_path(calc_number: int) -> str:
    rel = get_distance_weight_plot_png_relpath(calc_number).lstrip("/")
    return str(_project_root() / rel)


def _distance_range() -> range:
    # Distances are defined as 1-origin: 1..100
    return range(1, 101)


def _calc_values(distance: int) -> list[float]:
    reciprocal = 1.0 / distance
    sqrt = 1.0 / math.sqrt(distance)
    power025 = 1.0 / (distance**0.25)
    logarithm = 1.0 / math.log(distance + 1.0)  # natural logarithm
    return [reciprocal, sqrt, power025, logarithm]


def _write_distance_weight_tsv(ctx) -> str:
    out_path = Path(get_distance_weight_tsv_path())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(HEADER)
        for distance in _distance_range():
            writer.writerow(_calc_values(distance))

    ctx.info(f"[{USECASE_ID}/{UCB_ID}] saved tsv: {out_path}")
    return str(out_path)


def _read_distance_weight_tsv(tsv_path: str) -> dict[str, list[float]]:
    columns: dict[str, list[float]] = {header: [] for header in HEADER}
    with Path(tsv_path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            for header in HEADER:
                columns[header].append(float(row[header]))
    return columns


def _cleanup_png_files() -> None:
    all_path = Path(get_distance_weight_all_png_path())
    if all_path.exists():
        all_path.unlink()

    for calc_number in range(1, 5):
        each_path = Path(get_distance_weight_plot_png_path(calc_number))
        if each_path.exists():
            each_path.unlink()


def _plot_distance_weights(ctx, columns: dict[str, list[float]]) -> list[str]:
    _cleanup_png_files()

    distances = list(_distance_range())
    created_paths: list[str] = []

    # Overlay chart for all formulas.
    plt.figure(figsize=(12, 7))
    for calc_number, header in enumerate(HEADER, start=1):
        plt.plot(distances, columns[header], linewidth=2.0, label=f"{calc_number}: {CALC_TITLES[calc_number]}")
    plt.title("Distance Weight Functions")
    plt.xlabel("distance d")
    plt.ylabel("weight")
    plt.grid(True, alpha=0.3)
    plt.legend()

    all_path = Path(get_distance_weight_all_png_path())
    all_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(all_path, dpi=160, bbox_inches="tight")
    plt.close()
    created_paths.append(str(all_path))

    # Individual chart per formula.
    for calc_number, header in enumerate(HEADER, start=1):
        plt.figure(figsize=(12, 7))
        plt.plot(distances, columns[header], linewidth=2.2)
        plt.title(f"Distance Weight {calc_number}: {CALC_TITLES[calc_number]}")
        plt.xlabel("distance d")
        plt.ylabel("weight")
        plt.grid(True, alpha=0.3)

        each_path = Path(get_distance_weight_plot_png_path(calc_number))
        plt.savefig(each_path, dpi=160, bbox_inches="tight")
        plt.close()
        created_paths.append(str(each_path))

    ctx.info(f"[{USECASE_ID}/{UCB_ID}] saved png files: {created_paths}")
    return created_paths


def test_distance_weight(ctx=None) -> dict[str, str | list[str]]:
    if ctx is None:
        ctx = _create_context()

    tsv_path = _write_distance_weight_tsv(ctx)
    columns = _read_distance_weight_tsv(tsv_path)
    png_paths = _plot_distance_weights(ctx, columns)

    return {
        "tsv": tsv_path,
        "png": png_paths,
    }


def main() -> None:
    _ensure_src_on_sys_path()
    from project_context.project_context import ProjectContext

    ctx = ProjectContext()
    option = sys.argv[1] if len(sys.argv) > 1 else ""
    ctx.info(f"[{USECASE_ID}/{UCB_ID}] start option='{option or 'none'}'")

    try:
        if option == "distance_weight":
            test_distance_weight(ctx=ctx)
            return

        raise ValueError(f"Unknown option: {option}")
    finally:
        ctx.info(f"[{USECASE_ID}/{UCB_ID}] end")


if __name__ == "__main__":
    main()

"""
python src/uca002/ucb001/main.py distance_weight
"""
