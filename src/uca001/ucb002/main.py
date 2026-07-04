from __future__ import annotations

import pickle
import sqlite3
import sys
from pathlib import Path


USECASE_ID = "uca001"
UCB_ID = "ucb002"
TABLE_NAME = "chabsa"


def _project_root() -> Path:
    # src/uca001/ucb002/main.py -> project root
    return Path(__file__).resolve().parents[3]


def _src_root() -> Path:
    return _project_root() / "src"


def _ensure_src_on_sys_path() -> None:
    src = str(_src_root())
    if src not in sys.path:
        sys.path.insert(0, src)


def get_chabsa_origin_db_relpath() -> str:
    return "/data/uca001/ucb002/chabsa_origin.db"


def get_chabsa_origin_db_path() -> str:
    rel = get_chabsa_origin_db_relpath().lstrip("/")
    return str(_project_root() / rel)


def _read_dataset_from_ucb001_pickle():
    _ensure_src_on_sys_path()
    from uca001.ucb001.main import get_chabsa_dataset_path

    input_path = Path(get_chabsa_dataset_path())
    with input_path.open("rb") as f:
        return pickle.load(f)


def _create_context():
    _ensure_src_on_sys_path()
    from project_context.project_context import ProjectContext

    return ProjectContext()


def store_sqlite(ctx=None) -> None:
    if ctx is None:
        ctx = _create_context()

    dataset = _read_dataset_from_ucb001_pickle()
    total_rows = dataset["train"].num_rows
    output_path = Path(get_chabsa_origin_db_path())
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        output_path.unlink()

    with sqlite3.connect(output_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"CREATE TABLE {TABLE_NAME} ("
            "row_id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "sentence TEXT, "
            "target TEXT, "
            "polarity TEXT"
            ")"
        )

        for index, row in enumerate(dataset["train"], start=1):
            cursor.execute(
                f"INSERT INTO {TABLE_NAME} (sentence, target, polarity) VALUES (?, ?, ?)",
                (row["sentence"], row["target"], row["polarity"]),
            )
            if index % 100 == 0:
                ctx.info(f"[{USECASE_ID}/{UCB_ID}] progress {index}/{total_rows}")

    ctx.info(f"[{USECASE_ID}/{UCB_ID}] inserted {total_rows} rows")


def _show_sample(sample_size: int = 5) -> None:
    db_path = Path(get_chabsa_origin_db_path())
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT row_id, sentence, target, polarity FROM {TABLE_NAME} "
            "ORDER BY RANDOM() LIMIT ?",
            (sample_size,),
        )
        rows = cursor.fetchall()

    for row in rows:
        print(row)


def main() -> None:
    ctx = _create_context()
    option = sys.argv[1] if len(sys.argv) > 1 else ""
    ctx.info(f"[{USECASE_ID}/{UCB_ID}] start option='{option or 'default'}'")

    try:
        if option == "sample":
            _show_sample(5)
            return

        if option in ("",):
            store_sqlite(ctx=ctx)
            return

        raise ValueError(f"Unknown option: {option}")
    finally:
        ctx.info(f"[{USECASE_ID}/{UCB_ID}] end")


if __name__ == "__main__":
    main()

"""
python src/uca001/ucb002/main.py
python src/uca001/ucb002/main.py sample
"""
