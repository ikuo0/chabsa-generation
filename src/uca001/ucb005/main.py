from __future__ import annotations

import csv
import math
import os
import re
import sqlite3
import statistics
import sys
from collections import Counter
from pathlib import Path

import sentencepiece


USECASE_ID = "uca001"
UCB_ID = "ucb005"
TABLE_NAME = "chabsa_tokenized_vocab"
TSV_FILENAME = "chabsa_tokenized_analyzed.tsv"


def _project_root() -> Path:
    # src/uca001/ucb005/main.py -> project root
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


def get_chabsa_tokenized_analyzed_tsv_relpath() -> str:
    return "/data/uca001/ucb005/chabsa_tokenized_analyzed.tsv"


def get_chabsa_tokenized_analyzed_tsv_path() -> str:
    rel = get_chabsa_tokenized_analyzed_tsv_relpath().lstrip("/")
    return str(_project_root() / rel)


def _get_tokenized_db_dir() -> Path:
    _ensure_src_on_sys_path()
    from uca001.ucb004.main import get_chabsa_tokenized_db_path

    sample_path = Path(get_chabsa_tokenized_db_path(1400))
    return sample_path.parent


def _get_normalized_db_path() -> str:
    _ensure_src_on_sys_path()
    from uca001.ucb003.main import get_chabsa_normalized_db_path

    return get_chabsa_normalized_db_path()


def _list_tokenized_db_paths() -> list[Path]:
    db_dir = _get_tokenized_db_dir()
    db_paths = sorted(db_dir.glob("chabsa_tokenized_*.db"), key=_extract_vocab_size_from_path)
    return db_paths


def _extract_vocab_size_from_path(db_path: Path) -> int:
    match = re.search(r"chabsa_tokenized_(\d+)\.db$", db_path.name)
    if match is None:
        raise ValueError(f"Invalid tokenized DB file name: {db_path}")
    return int(match.group(1))


def _load_sentences() -> list[str]:
    normalized_db_path = _get_normalized_db_path()
    with sqlite3.connect(normalized_db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT sentence FROM chabsa ORDER BY row_id")
        return [row[0] for row in cursor.fetchall()]


def _load_sentencepiece_processor(vocab_size: int) -> sentencepiece.SentencePieceProcessor:
    _ensure_src_on_sys_path()
    from uca001.ucb004.main import get_spm_model_prefix_path

    model_file = Path(f"{get_spm_model_prefix_path(vocab_size)}.model")
    if not model_file.exists():
        raise FileNotFoundError(f"SentencePiece model not found: {model_file}")

    processor = sentencepiece.SentencePieceProcessor()
    processor.load(str(model_file))
    return processor


def _count_low_frequency_tokens(piece_counter: Counter[int]) -> int:
    return sum(1 for count in piece_counter.values() if count <= 2)


def _analyze_one_vocab_size(vocab_size: int, sentences: list[str], ctx) -> list[float | int]:
    processor = _load_sentencepiece_processor(vocab_size)

    sentence_lengths: list[int] = []
    token_char_lengths: list[float] = []
    token_per_char_ratios: list[float] = []
    piece_counter: Counter[int] = Counter()

    for sentence in sentences:
        pieces = processor.encode(sentence, out_type=str)
        ids = processor.encode(sentence, out_type=int)

        sentence_token_count = len(ids)
        sentence_char_count = len(sentence)
        sentence_token_char_count = sum(len(piece) for piece in pieces)
        sentence_avg_token_char_length = (
            sentence_token_char_count / sentence_token_count if sentence_token_count > 0 else 0
        )

        sentence_lengths.append(sentence_token_count)
        token_char_lengths.append(sentence_avg_token_char_length)
        token_per_char_ratios.append(
            sentence_token_count / sentence_char_count if sentence_char_count > 0 else 0
        )
        piece_counter.update(ids)

    total_sentences = len(sentences)
    low_frequency_token_count = _count_low_frequency_tokens(piece_counter)
    average_token_char_length = statistics.fmean(token_char_lengths) if token_char_lengths else 0.0
    average_token_count = statistics.fmean(sentence_lengths) if sentence_lengths else 0.0
    low_frequency_ratio = low_frequency_token_count / vocab_size if vocab_size > 0 else 0.0
    token_count_variance = statistics.pvariance(sentence_lengths) if len(sentence_lengths) > 1 else 0.0
    token_count_stddev = statistics.pstdev(sentence_lengths) if len(sentence_lengths) > 1 else 0.0
    token_count_min = min(sentence_lengths) if sentence_lengths else 0
    token_count_max = max(sentence_lengths) if sentence_lengths else 0
    token_count_median = statistics.median(sentence_lengths) if sentence_lengths else 0
    average_token_per_char = statistics.fmean(token_per_char_ratios) if token_per_char_ratios else 0.0

    ctx.info(
        f"[{USECASE_ID}/{UCB_ID}] vocab_size={vocab_size}, "
        f"avg_token_char_len={average_token_char_length:.4f}, "
        f"avg_token_count={average_token_count:.4f}, "
        f"low_freq={low_frequency_token_count}, "
        f"low_freq_ratio={low_frequency_ratio:.4f}, "
        f"variance={token_count_variance:.4f}, "
        f"stddev={token_count_stddev:.4f}, "
        f"min={token_count_min}, "
        f"max={token_count_max}, "
        f"median={token_count_median}, "
        f"avg_token_per_char={average_token_per_char:.4f}, "
        f"sentences={total_sentences}"
    )

    return [
        vocab_size,
        average_token_char_length,
        average_token_count,
        low_frequency_token_count,
        low_frequency_ratio,
        token_count_variance,
        token_count_stddev,
        token_count_min,
        token_count_max,
        token_count_median,
        average_token_per_char,
    ]


def _write_tsv(rows: list[list[float | int]], out_path: str) -> None:
    header = [
        "vocab_size",
        "avg_token_char_length",
        "avg_token_count_per_sentence",
        "low_frequency_token_count",
        "low_frequency_token_ratio",
        "token_count_variance",
        "token_count_stddev",
        "token_count_min",
        "token_count_max",
        "token_count_median",
        "avg_token_per_char",
    ]
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists():
        out_file.unlink()

    with out_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def _read_tsv_rows(tsv_path: str) -> list[list[str]]:
    with Path(tsv_path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        return list(reader)


def _print_table(rows: list[list[str]]) -> None:
    if not rows:
        return

    widths = [max(len(str(row[index])) for row in rows) for index in range(len(rows[0]))]
    for row_index, row in enumerate(rows):
        line = " | ".join(str(value).ljust(widths[index]) for index, value in enumerate(row))
        print(line)
        if row_index == 0:
            print("-+-".join("-" * width for width in widths))


def analyze_tokenized_data(ctx=None) -> str:
    if ctx is None:
        ctx = _create_context()

    sentences = _load_sentences()
    db_paths = _list_tokenized_db_paths()
    if not db_paths:
        raise FileNotFoundError(f"No tokenized DB files found under: {_get_tokenized_db_dir()}")

    rows: list[list[float | int]] = []
    for db_path in db_paths:
        vocab_size = _extract_vocab_size_from_path(db_path)
        rows.append(_analyze_one_vocab_size(vocab_size, sentences, ctx))

    out_path = get_chabsa_tokenized_analyzed_tsv_path()
    _write_tsv(rows, out_path)
    ctx.info(f"[{USECASE_ID}/{UCB_ID}] saved analyzed TSV: {out_path}")
    return out_path


def main() -> None:
    _ensure_src_on_sys_path()
    from project_context.project_context import ProjectContext

    ctx = ProjectContext()
    option = sys.argv[1] if len(sys.argv) > 1 else ""
    ctx.info(f"[{USECASE_ID}/{UCB_ID}] start option='{option or 'default'}'")

    try:
        if option == "view":
            rows = _read_tsv_rows(get_chabsa_tokenized_analyzed_tsv_path())
            _print_table(rows)
            return

        if option in ("",):
            analyze_tokenized_data(ctx=ctx)
            return

        raise ValueError(f"Unknown option: {option}")
    finally:
        ctx.info(f"[{USECASE_ID}/{UCB_ID}] end")


if __name__ == "__main__":
    main()

"""
python src/uca001/ucb005/main.py
python src/uca001/ucb005/main.py view
"""
