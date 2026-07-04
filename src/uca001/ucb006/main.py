from __future__ import annotations

import statistics
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import sentencepiece


USECASE_ID = "uca001"
UCB_ID = "ucb006"
TABLE_NAME = "chabsa_tokenized_vocab"
MIN_VOCAB_SIZE = 1400


def _project_root() -> Path:
    # src/uca001/ucb006/main.py -> project root
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


def get_chabsa_tokenized_db_relpath() -> str:
    return "/data/uca001/ucb006/chabsa_tokenized.db"


def get_chabsa_tokenized_db_path() -> str:
    rel = get_chabsa_tokenized_db_relpath().lstrip("/")
    return str(_project_root() / rel)


def get_spm_model_prefix_relpath() -> str:
    return "/data/uca001/ucb006/spm_model"


def get_spm_model_prefix_path() -> str:
    rel = get_spm_model_prefix_relpath().lstrip("/")
    return str(_project_root() / rel)


def get_chabsa_tokenized_info_txt_relpath() -> str:
    return "/data/uca001/ucb006/chabsa_tokenized_info.txt"


def get_chabsa_tokenized_info_txt_path() -> str:
    rel = get_chabsa_tokenized_info_txt_relpath().lstrip("/")
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
    sentences_file = _project_root() / "data" / "uca001" / "ucb006" / "sentences.txt"
    sentences_file.parent.mkdir(parents=True, exist_ok=True)
    with sentences_file.open("w", encoding="utf-8") as f:
        for sentence in sentences:
            f.write(sentence + "\n")
    ctx.info(f"[{USECASE_ID}/{UCB_ID}] saved sentences file: {sentences_file}")
    return str(sentences_file)


def _train_sentencepiece_model(ctx, sentences_file: str, vocab_size: int) -> sentencepiece.SentencePieceProcessor:
    model_prefix = get_spm_model_prefix_path()
    ctx.info(
        f"[{USECASE_ID}/{UCB_ID}] training SentencePiece model with vocab size {vocab_size}: {model_prefix}"
    )

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


def _save_tokenized_db(ctx, sp: sentencepiece.SentencePieceProcessor) -> str:
    db_path = Path(get_chabsa_tokenized_db_path())
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


def _count_low_frequency_tokens(piece_counter: Counter[int]) -> int:
    return sum(1 for count in piece_counter.values() if count <= 2)


def _analyze_tokenization(sentences: list[str], sp: sentencepiece.SentencePieceProcessor, vocab_size: int) -> dict[str, float | int]:
    sentence_lengths: list[int] = []
    token_char_lengths: list[float] = []
    token_per_char_ratios: list[float] = []
    piece_counter: Counter[int] = Counter()

    for sentence in sentences:
        pieces = sp.encode(sentence, out_type=str)
        ids = sp.encode(sentence, out_type=int)

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

    low_frequency_token_count = _count_low_frequency_tokens(piece_counter)

    return {
        "vocab_size": vocab_size,
        "avg_token_char_length": statistics.fmean(token_char_lengths) if token_char_lengths else 0.0,
        "avg_token_count_per_sentence": statistics.fmean(sentence_lengths) if sentence_lengths else 0.0,
        "low_frequency_token_count": low_frequency_token_count,
        "low_frequency_token_ratio": low_frequency_token_count / vocab_size if vocab_size > 0 else 0.0,
        "token_count_variance": statistics.pvariance(sentence_lengths) if len(sentence_lengths) > 1 else 0.0,
        "token_count_stddev": statistics.pstdev(sentence_lengths) if len(sentence_lengths) > 1 else 0.0,
        "token_count_min": min(sentence_lengths) if sentence_lengths else 0,
        "token_count_max": max(sentence_lengths) if sentence_lengths else 0,
        "token_count_median": statistics.median(sentence_lengths) if sentence_lengths else 0,
        "avg_token_per_char": statistics.fmean(token_per_char_ratios) if token_per_char_ratios else 0.0,
    }


def _write_tokenized_info(ctx, metrics: dict[str, float | int]) -> str:
    out_path = Path(get_chabsa_tokenized_info_txt_path())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

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
    values = [str(metrics[key]) for key in header]

    with out_path.open("w", encoding="utf-8") as f:
        f.write(", ".join(header) + "\n")
        f.write(", ".join(values) + "\n")

    ctx.info(f"[{USECASE_ID}/{UCB_ID}] saved tokenized info: {out_path}")
    return str(out_path)


def tokenize_text(vocab_size: int, ctx=None) -> str:
    if vocab_size < MIN_VOCAB_SIZE:
        raise ValueError(f"vocab_size must be >= {MIN_VOCAB_SIZE}: {vocab_size}")

    if ctx is None:
        ctx = _create_context()

    sentences = _get_sentences()
    sentences_file = _create_sentences_file(ctx, sentences)
    sp = _train_sentencepiece_model(ctx, sentences_file, vocab_size)
    db_path = _save_tokenized_db(ctx, sp)

    metrics = _analyze_tokenization(sentences=sentences, sp=sp, vocab_size=vocab_size)
    _write_tokenized_info(ctx, metrics)

    return db_path


def main() -> None:
    _ensure_src_on_sys_path()
    from project_context.project_context import ProjectContext

    ctx = ProjectContext()
    option = sys.argv[1] if len(sys.argv) > 1 else ""
    ctx.info(f"[{USECASE_ID}/{UCB_ID}] start option='{option or 'none'}'")

    try:
        if option == "":
            raise ValueError("numeric vocab_size argument is required")

        try:
            vocab_size = int(option)
        except ValueError as exc:
            raise ValueError(f"option must be integer vocab_size: {option}") from exc

        tokenize_text(vocab_size=vocab_size, ctx=ctx)
    finally:
        ctx.info(f"[{USECASE_ID}/{UCB_ID}] end")


if __name__ == "__main__":
    main()

"""
data/uca001/ucb005/chabsa_tokenized_analyzed.tsv の内容を確認して vocab_size を決定する
python src/uca001/ucb006/main.py 1400
python src/uca001/ucb006/main.py 6460
"""
