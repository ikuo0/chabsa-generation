from __future__ import annotations

import math
import os
import random
import sqlite3
import sys
from pathlib import Path

import sentencepiece


USECASE_ID = "uca001"
UCB_ID = "ucb004"
TABLE_NAME = "chabsa_tokenized_vocab"
DEFAULT_MIN_VOCAB_SIZE = 1400
DEFAULT_MAX_VOCAB_SIZE = 10000
DEFAULT_STEP_COUNT = 10
DEFAULT_SAMPLE_COUNT = 20


def _project_root() -> Path:
    # src/uca001/ucb004/main.py -> project root
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


def get_chabsa_tokenized_db_relpath(vocab_size: int) -> str:
    return f"/data/uca001/ucb004/chabsa_tokenized_{vocab_size}.db"


def get_chabsa_tokenized_db_path(vocab_size: int) -> str:
    rel = get_chabsa_tokenized_db_relpath(vocab_size).lstrip("/")
    return str(_project_root() / rel)


def get_spm_model_prefix_relpath(vocab_size: int) -> str:
    return f"/data/uca001/ucb004/spm_model_{vocab_size}"


def get_spm_model_prefix_path(vocab_size: int) -> str:
    rel = get_spm_model_prefix_relpath(vocab_size).lstrip("/")
    return str(_project_root() / rel)


def _get_normalized_db_path() -> str:
    _ensure_src_on_sys_path()
    from uca001.ucb003.main import get_chabsa_normalized_db_path

    return get_chabsa_normalized_db_path()


def _get_sentences() -> list[str]:
    normalized_db_path = _get_normalized_db_path()
    with sqlite3.connect(normalized_db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT sentence FROM chabsa ORDER BY row_id")
        return [row[0] for row in cursor.fetchall()]


def _create_sentences_file(ctx, sentences: list[str]) -> str:
    sentences_file = _project_root() / "data" / "uca001" / "ucb004" / "sentences.txt"
    sentences_file.parent.mkdir(parents=True, exist_ok=True)
    with sentences_file.open("w", encoding="utf-8") as f:
        for sentence in sentences:
            f.write(sentence + "\n")
    ctx.info(f"[{USECASE_ID}/{UCB_ID}] saved sentences file: {sentences_file}")
    return str(sentences_file)


def get_vocab_sizes(n: int = DEFAULT_MIN_VOCAB_SIZE, m: int = DEFAULT_MAX_VOCAB_SIZE, step_count: int = DEFAULT_STEP_COUNT) -> list[int]:
    if step_count < 2:
        raise ValueError("step_count must be at least 2")
    if n <= 0 or m <= 0:
        raise ValueError("n and m must be positive integers")
    if n >= m:
        raise ValueError("n must be less than m")

    log_min = math.log(n)
    log_max = math.log(m)
    values = [round(math.exp(log_min + (log_max - log_min) * i / (step_count - 1))) for i in range(step_count)]
    values[0] = n
    values[-1] = m

    for index in range(1, len(values)):
        if values[index] <= values[index - 1]:
            values[index] = values[index - 1] + 1

    values[-1] = m
    return values


def _train_sentencepiece_model(ctx, sentences_file: str, vocab_size: int) -> sentencepiece.SentencePieceProcessor:
    ctx.info(f"[{USECASE_ID}/{UCB_ID}] training SentencePiece model with vocab size {vocab_size}")
    model_prefix = get_spm_model_prefix_path(vocab_size)

    model_file = Path(f"{model_prefix}.model")
    vocab_file = Path(f"{model_prefix}.vocab")
    for file_path in (model_file, vocab_file):
        if file_path.exists():
            file_path.unlink()

    sentencepiece.SentencePieceTrainer.train(
        input=sentences_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=0.9995,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )

    processor = sentencepiece.SentencePieceProcessor()
    processor.load(str(model_file))
    return processor


def _save_tokenized_db(ctx, sp: sentencepiece.SentencePieceProcessor, vocab_size: int) -> str:
    db_path = Path(get_chabsa_tokenized_db_path(vocab_size))
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if db_path.exists():
        db_path.unlink()

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"""
            CREATE TABLE {TABLE_NAME} (
                vocab_id INTEGER PRIMARY KEY,
                vocab TEXT NOT NULL,
                score REAL NOT NULL
            )
            """
        )

        for vocab_id in range(sp.get_piece_size()):
            cursor.execute(
                f"INSERT INTO {TABLE_NAME} (vocab_id, vocab, score) VALUES (?, ?, ?)",
                (vocab_id, sp.id_to_piece(vocab_id), sp.get_score(vocab_id)),
            )

    ctx.info(f"[{USECASE_ID}/{UCB_ID}] saved token db: {db_path}")
    return str(db_path)


def _tokenize_one_vocab_size(ctx, sentences_file: str, vocab_size: int) -> str:
    sp = _train_sentencepiece_model(ctx, sentences_file, vocab_size)
    _save_tokenized_db(ctx, sp, vocab_size)
    return get_chabsa_tokenized_db_path(vocab_size)


def tokenize_text(ctx=None) -> list[str]:
    if ctx is None:
        ctx = _create_context()

    sentences = _get_sentences()
    sentences_file = _create_sentences_file(ctx, sentences)
    vocab_sizes = get_vocab_sizes()
    ctx.info(f"[{USECASE_ID}/{UCB_ID}] vocab sizes: {vocab_sizes}")

    created_db_paths = []
    for vocab_size in vocab_sizes:
        created_db_paths.append(_tokenize_one_vocab_size(ctx, sentences_file, vocab_size))

    return created_db_paths


def _sample_token_db(token_db_path: str, sample_count: int = DEFAULT_SAMPLE_COUNT, ctx=None) -> None:
    if ctx is None:
        ctx = _create_context()

    db_path = Path(token_db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Token DB not found: {db_path}")

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
        total_count = cursor.fetchone()[0]
        ctx.info(f"[{USECASE_ID}/{UCB_ID}] token rows: {total_count}")
        cursor.execute(
            f"SELECT vocab_id, vocab, score FROM {TABLE_NAME} ORDER BY RANDOM() LIMIT ?",
            (sample_count,),
        )
        rows = cursor.fetchall()

    for row in rows:
        print(row)


def main() -> None:
    _ensure_src_on_sys_path()
    from project_context.project_context import ProjectContext

    ctx = ProjectContext()
    option = sys.argv[1] if len(sys.argv) > 1 else ""
    ctx.info(f"[{USECASE_ID}/{UCB_ID}] start option='{option or 'default'}'")

    try:
        if option == "sample":
            if len(sys.argv) < 3:
                raise ValueError("sample option requires a token DB absolute path")
            _sample_token_db(sys.argv[2], DEFAULT_SAMPLE_COUNT, ctx=ctx)
            return

        if option in ("",):
            tokenize_text(ctx=ctx)
            return

        raise ValueError(f"Unknown option: {option}")
    finally:
        ctx.info(f"[{USECASE_ID}/{UCB_ID}] end")


if __name__ == "__main__":
    main()

"""
python src/uca001/ucb004/main.py
python src/uca001/ucb004/main.py sample /workspaces/chabsa-document-classification/data/uca001/ucb004/chabsa_tokenized_1400.db
"""
