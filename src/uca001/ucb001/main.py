from __future__ import annotations

import pickle
import sys
from pathlib import Path

from datasets import load_dataset


USECASE_ID = "uca001"
UCB_ID = "ucb001"


def _project_root() -> Path:
    # src/uca001/ucb001/main.py -> project root
    return Path(__file__).resolve().parents[3]


def _src_root() -> Path:
    return _project_root() / "src"


def _ensure_src_on_sys_path() -> None:
    src = str(_src_root())
    if src not in sys.path:
        sys.path.insert(0, src)


def get_chabsa_dataset_relpath() -> str:
    return "/data/uca001/ucb001/chabsa_dataset.pkl"


def get_chabsa_dataset_path() -> str:
    rel = get_chabsa_dataset_relpath().lstrip("/")
    return str(_project_root() / rel)


def get_chabsa_dataset():
    return load_dataset("TheFinAI/jp-chABSA")


def _save_dataset_pickle(dataset) -> None:
    output_path = Path(get_chabsa_dataset_path())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    with output_path.open("wb") as f:
        pickle.dump(dataset, f)


def _check_keys() -> None:
    input_path = Path(get_chabsa_dataset_path())
    with input_path.open("rb") as f:
        dataset = pickle.load(f)

    print(dataset)
    print(dataset["train"])
    print(dataset["train"].features)
    print(f"Parent keys: {dataset.keys()}")
    for i in range(3):
        print(f"Example {i}:")
        print(dataset["train"][i])


def _run_default_pipeline() -> None:
    dataset = get_chabsa_dataset()
    _save_dataset_pickle(dataset)

    # UCB001のデフォルト実行でSQLite保存まで完了させる。
    _ensure_src_on_sys_path()
    from uca001.ucb002.main import store_sqlite

    store_sqlite()


def main() -> None:
    _ensure_src_on_sys_path()
    from project_context.project_context import ProjectContext

    ctx = ProjectContext()
    option = sys.argv[1] if len(sys.argv) > 1 else ""
    ctx.info(f"[{USECASE_ID}/{UCB_ID}] start option='{option or 'default'}'")

    try:
        if option == "check_keys":
            _check_keys()
            return

        if option in ("",):
            _run_default_pipeline()
            return

        raise ValueError(f"Unknown option: {option}")
    finally:
        ctx.info(f"[{USECASE_ID}/{UCB_ID}] end")


if __name__ == "__main__":
    main()

"""
python src/uca001/ucb001/main.py
python src/uca001/ucb001/main.py check_keys
"""
