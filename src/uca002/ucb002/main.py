from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
import random

import numpy as np
from scipy import sparse
import sentencepiece
from scipy.sparse import csr_matrix, save_npz


USECASE_ID = "uca002"
UCB_ID = "ucb002"
TABLE_NAME = "chabsa"

DEFAULT_WINDOW_WIDTH = 15
DEFAULT_SLIDE_WIDTH = 1


def _project_root() -> Path:
    # src/uca002/ucb002/main.py -> project root
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


def get_spm_model_relpath() -> str:
    return "/data/uca001/ucb006/spm_model.model"


def get_spm_model_path() -> str:
    rel = get_spm_model_relpath().lstrip("/")
    return str(_project_root() / rel)


def get_chabsa_index_relpath() -> str:
    return "/data/uca002/ucb002/chabsa_index.npy"


def get_chabsa_index_path() -> str:
    rel = get_chabsa_index_relpath().lstrip("/")
    return str(_project_root() / rel)


def get_chabsa_vector_relpath() -> str:
    return "/data/uca002/ucb002/chabsa_vector.npz"


def get_chabsa_vector_path() -> str:
    rel = get_chabsa_vector_relpath().lstrip("/")
    return str(_project_root() / rel)


def get_chabsa_next_word_relpath() -> str:
    return "/data/uca002/ucb002/chabsa_next_word.npy"


def get_chabsa_next_word_path() -> str:
    rel = get_chabsa_next_word_relpath().lstrip("/")
    return str(_project_root() / rel)


def _get_normalized_db_path() -> str:
    _ensure_src_on_sys_path()
    from uca001.ucb003.main import get_chabsa_normalized_db_path

    return get_chabsa_normalized_db_path()


def _cleanup_output_files() -> None:
    for output_file in (
        Path(get_chabsa_index_path()),
        Path(get_chabsa_vector_path()),
        Path(get_chabsa_next_word_path()),
    ):
        output_file.parent.mkdir(parents=True, exist_ok=True)
        if output_file.exists():
            output_file.unlink()


def _load_sentencepiece_model() -> sentencepiece.SentencePieceProcessor:
    model_path = Path(get_spm_model_path())
    if not model_path.exists():
        raise FileNotFoundError(f"SentencePiece model not found: {model_path}")

    sp = sentencepiece.SentencePieceProcessor()
    sp.Load(str(model_path))
    return sp


def _iter_documents() -> list[tuple[int, str]]:
    normalized_db_path = _get_normalized_db_path()
    with sqlite3.connect(normalized_db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(f"SELECT row_id, sentence FROM {TABLE_NAME} ORDER BY row_id")
        return [(int(row[0]), str(row[1])) for row in cursor.fetchall()]


def _distance_weight(distance: int) -> float:
    return 1.0 / (distance**0.25)


def _vectorize_document(
    *,
    row_id: int,
    sentence: str,
    sp: sentencepiece.SentencePieceProcessor,
    window_width: int,
    slide_width: int,
) -> tuple[list[int], list[int], list[dict[int, float]]]:
    token_ids = sp.EncodeAsIds(sentence)
    bos_id = sp.bos_id()
    eos_id = sp.eos_id()

    padded = [bos_id] * window_width + token_ids + [eos_id] * window_width

    doc_ids = []
    y_indexes = []
    idx_value_pairs = [] # 粗ベクトルとして保存、インデックスと値の配列の配列

    for start in range(0, len(padded) - window_width, slide_width):
        x_ids = padded[start : start + window_width]
        y_id = padded[start + window_width]
        idx_value_pair = {} # インデックスをキーとして値の加算値を保持、粗ベクトル
        for i, xid in enumerate(x_ids):
            distance = i + 1
            if xid in idx_value_pair:
                idx_value_pair[xid] += _distance_weight(distance)
            else:
                idx_value_pair[xid] = _distance_weight(distance)
        doc_ids.append(row_id)
        y_indexes.append(y_id)
        idx_value_pairs.append(idx_value_pair)

    return doc_ids, y_indexes, idx_value_pairs


def vectorize_text(window_width: int = DEFAULT_WINDOW_WIDTH, slide_width: int = DEFAULT_SLIDE_WIDTH, ctx=None) -> dict[str, str | int]:
    if ctx is None:
        ctx = _create_context()
    ctx.info(f"[{USECASE_ID}/{UCB_ID}] vectorize_text() start")

    # 単語辞書モデルの読み込み
    spm = _load_sentencepiece_model()
    token_count = spm.GetPieceSize()

    # ベクトルの初期化
    doc_ids = []
    y_indexes = []
    idx_value_pairs = []

    # SQLite から文章を取得してループ
    for row_id, sentence in _iter_documents():
        _doc_ids, _y_indexes, _idx_value_pairs = _vectorize_document(
            row_id=row_id,
            sentence=sentence,
            sp=spm,
            window_width=window_width,
            slide_width=slide_width,
        )
        doc_ids.extend(_doc_ids)
        y_indexes.extend(_y_indexes)
        idx_value_pairs.extend(_idx_value_pairs)

    # doc_ids を npy 配列として保存
    doc_ids_array = np.array(doc_ids, dtype=np.int32)
    np.save(get_chabsa_index_path(), doc_ids_array)

    # y_indexes を npy 配列として保存
    y_indexes_array = np.array(y_indexes, dtype=np.int32)
    np.save(get_chabsa_next_word_path(), y_indexes_array)

    # idx_value_pairs を scipy.sparse.csr_matrix として保存
    csr_row = len(doc_ids_array)
    csr_col = token_count
    rowX = sparse.lil_matrix((csr_row, csr_col), dtype=np.float32)
    for row_idx, idx_value_pair in enumerate(idx_value_pairs):
        for idx, value in idx_value_pair.items():
            rowX[row_idx, idx] = value
    rowX_csr = rowX.tocsr()
    save_npz(get_chabsa_vector_path(), rowX_csr)

    doc_ids_length = len(doc_ids_array)
    y_indexes_length = len(y_indexes_array)
    x_length = rowX_csr.shape[0]
    if doc_ids_length != y_indexes_length or x_length != doc_ids_length:
        raise ValueError(
            "Vectorized data is inconsistent: "
            f"len(doc_ids)={doc_ids_length}, len(y_indexes)={y_indexes_length}, x_rows={x_length}"
        )
    else:
        ctx.info(f"[{USECASE_ID}/{UCB_ID}] vectorized data is consistent: "
                 f"len(doc_ids)={doc_ids_length}, len(y_indexes)={y_indexes_length}, x_rows={x_length}")


    return {
        "index": get_chabsa_index_path(),
        "vector": get_chabsa_vector_path(),
        "next_word": get_chabsa_next_word_path(),
        "samples": len(doc_ids),
        "vocab_size": token_count,
        "window_width": window_width,
        "slide_width": slide_width,
    }


def _show_saved_samples_sub(x_row, sp: sentencepiece.SentencePieceProcessor) -> tuple[list[str], list[str]]:
    words = []
    values = []

    for index, value in zip(x_row.indices, x_row.data):
        words.append(sp.IdToPiece(int(index)))
        values.append(f"{float(value):.6f}")

    sorted_indexes = np.argsort(values)[::-1]
    words = [words[i] for i in sorted_indexes]
    values = [values[i] for i in sorted_indexes]

    return words, values

def _show_saved_samples(sample_count: int = 5, ctx=None) -> None:
    if ctx is None:
        ctx = _create_context()

    index_path = Path(get_chabsa_index_path())
    vector_path = Path(get_chabsa_vector_path())
    next_word_path = Path(get_chabsa_next_word_path())

    for required_file in (index_path, vector_path, next_word_path):
        if not required_file.exists():
            raise FileNotFoundError(f"Saved vector file not found: {required_file}")

    doc_ids = np.load(index_path)
    y_indexes = np.load(next_word_path)
    x_matrix = sparse.load_npz(vector_path)

    if len(doc_ids) != len(y_indexes) or x_matrix.shape[0] != len(doc_ids):
        raise ValueError(
            "Saved vectors are inconsistent: "
            f"len(doc_ids)={len(doc_ids)}, len(y_indexes)={len(y_indexes)}, x_rows={x_matrix.shape[0]}"
        )

    if len(doc_ids) == 0:
        print("No vector samples found.")
        return

    sp = _load_sentencepiece_model()

    count = min(sample_count, len(doc_ids))
    chosen_indices = random.sample(range(len(doc_ids)), count)

    # 総件数の表示
    ctx.info(f"[{USECASE_ID}/{UCB_ID}] total samples: {len(doc_ids)}")

    for sample_index in chosen_indices:
        row = x_matrix.getrow(sample_index)
        words, values = _show_saved_samples_sub(row, sp)
        answer = sp.IdToPiece(int(y_indexes[sample_index]))
        print("---")
        print(f"docId: {int(doc_ids[sample_index])}")
        print(f"answer: {answer}")
        print(", ".join(words))
        print(", ".join(values))

    ctx.info(f"[{USECASE_ID}/{UCB_ID}] displayed samples: {count}")


def main() -> None:
    _ensure_src_on_sys_path()
    from project_context.project_context import ProjectContext

    ctx = ProjectContext()
    option = sys.argv[1] if len(sys.argv) > 1 else ""
    ctx.info(f"[{USECASE_ID}/{UCB_ID}] start option='{option or 'default'}'")

    try:
        if option == "sample":
            _show_saved_samples(sample_count=5, ctx=ctx)
            return

        if option in ("",):
            result = vectorize_text(ctx=ctx)
            ctx.info(f"[{USECASE_ID}/{UCB_ID}] vectorize_text() result: {result}")
            return

        raise ValueError(f"Unknown option: {option}")
    finally:
        ctx.info(f"[{USECASE_ID}/{UCB_ID}] end")


if __name__ == "__main__":
    main()

"""
python src/uca002/ucb002/main.py
python src/uca002/ucb002/main.py sample
"""
