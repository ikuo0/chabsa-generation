from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import sentencepiece
import tensorflow as tf


USECASE_ID = "uca009"
UCB_ID = "ucb002"

WINDOW_WIDTH = 15
MAX_PREDICT_WORDS = 80
DEFAULT_TOP_K = 2
DEFAULT_TEMPERATURE = 1.0


def _project_root() -> Path:
    # src/uca009/ucb002/main.py -> project root
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


def get_chabsa_next_word_predict_relpath() -> str:
    return "/data/uca009/ucb002/chabsa_next_word_predict.txt"


def get_chabsa_next_word_predict_path() -> str:
    rel = get_chabsa_next_word_predict_relpath().lstrip("/")
    return str(_project_root() / rel)


def _load_sentencepiece_model() -> sentencepiece.SentencePieceProcessor:
    model_path = Path(get_spm_model_path())
    if not model_path.exists():
        raise FileNotFoundError(f"SentencePiece model not found: {model_path}")

    sp = sentencepiece.SentencePieceProcessor()
    sp.Load(str(model_path))
    return sp


def _load_trained_model() -> tf.keras.Model:
    _ensure_src_on_sys_path()
    from uca009.ucb001.main import (
        get_chabsa_next_word_model_h5_path,
        get_chabsa_next_word_model_keras_path,
    )

    h5_path = Path(get_chabsa_next_word_model_h5_path())
    keras_path = Path(get_chabsa_next_word_model_keras_path())

    if h5_path.exists():
        return tf.keras.models.load_model(str(h5_path))
    if keras_path.exists():
        return tf.keras.models.load_model(str(keras_path))

    raise FileNotFoundError(
        "Trained model not found: "
        f"{h5_path} or {keras_path}. Run UCA009/UCB001 first."
    )


def _distance_weight(distance: int) -> float:
    return 1.0 / (distance**0.25)


def _build_sparse_context_tensor(context_ids: list[int], vocab_size: int) -> tf.sparse.SparseTensor:
    weights: dict[int, float] = {}
    for index, token_id in enumerate(context_ids):
        if token_id < 0 or token_id >= vocab_size:
            continue
        distance = index + 1
        weights[token_id] = weights.get(token_id, 0.0) + _distance_weight(distance)

    indices = [[0, token_id] for token_id in sorted(weights)]
    values = [weights[token_id] for token_id in sorted(weights)]
    sparse_tensor = tf.sparse.SparseTensor(
        indices=indices,
        values=values,
        dense_shape=[1, vocab_size],
    )
    return tf.sparse.reorder(sparse_tensor)


def _build_dense_context_tensor(context_ids: list[int], vocab_size: int) -> tf.Tensor:
    vector = np.zeros((1, vocab_size), dtype=np.float32)
    for index, token_id in enumerate(context_ids):
        if token_id < 0 or token_id >= vocab_size:
            continue
        distance = index + 1
        vector[0, token_id] += _distance_weight(distance)
    return tf.convert_to_tensor(vector, dtype=tf.float32)


def _model_expects_sparse_input(model: tf.keras.Model) -> bool:
    if not model.inputs:
        return False
    return bool(getattr(model.inputs[0], "sparse", False))


def _sample_from_logits(
    logits_1d: np.ndarray,
    *,
    top_k: int,
    temperature: float,
    rng: np.random.Generator,
) -> int:
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0: {temperature}")
    if top_k <= 0:
        raise ValueError(f"top_k must be positive: {top_k}")

    vocab_size = int(logits_1d.shape[0])
    effective_k = min(top_k, vocab_size)

    if effective_k == vocab_size:
        candidate_indexes = np.arange(vocab_size)
        candidate_logits = logits_1d
    else:
        candidate_indexes = np.argpartition(logits_1d, -effective_k)[-effective_k:]
        candidate_logits = logits_1d[candidate_indexes]

    scaled = candidate_logits / float(temperature)
    scaled = scaled - np.max(scaled)
    probs = np.exp(scaled)
    probs_sum = float(np.sum(probs))
    if probs_sum <= 0.0:
        return int(candidate_indexes[int(np.argmax(candidate_logits))])
    probs = probs / probs_sum

    sampled_index = int(rng.choice(len(candidate_indexes), p=probs))
    return int(candidate_indexes[sampled_index])


def _predict_next_token_id(
    model: tf.keras.Model,
    context_ids: list[int],
    vocab_size: int,
    *,
    top_k: int,
    temperature: float,
    rng: np.random.Generator,
) -> int:
    if _model_expects_sparse_input(model):
        x_input = _build_sparse_context_tensor(context_ids, vocab_size)
    else:
        x_input = _build_dense_context_tensor(context_ids, vocab_size)

    logits = model(x_input, training=False)
    logits_1d = np.asarray(logits[0].numpy(), dtype=np.float64)
    next_token_id = _sample_from_logits(
        logits_1d,
        top_k=top_k,
        temperature=temperature,
        rng=rng,
    )
    return next_token_id


def _tokens_to_text(tokens: list[int], sp: sentencepiece.SentencePieceProcessor) -> str:
    bos_id = sp.bos_id()
    pieces = [sp.IdToPiece(token_id) for token_id in tokens if token_id != bos_id]
    return "".join(pieces)


def predict_sequence(
    *,
    model: tf.keras.Model,
    sp: sentencepiece.SentencePieceProcessor,
    seed_text: str,
    max_predict_words: int = MAX_PREDICT_WORDS,
    top_k: int = DEFAULT_TOP_K,
    temperature: float = DEFAULT_TEMPERATURE,
    rng: np.random.Generator | None = None,
) -> tuple[list[int], str]:
    vocab_size = int(sp.GetPieceSize())
    bos_id = int(sp.bos_id())
    eos_id = int(sp.eos_id())
    if rng is None:
        rng = np.random.default_rng()

    seed_ids = sp.EncodeAsIds(seed_text) if seed_text else []
    sequence = [bos_id] * WINDOW_WIDTH + [int(token_id) for token_id in seed_ids]

    predicted_count = 0
    while predicted_count < max_predict_words:
        context_ids = sequence[-WINDOW_WIDTH:]
        next_token_id = _predict_next_token_id(
            model=model,
            context_ids=context_ids,
            vocab_size=vocab_size,
            top_k=top_k,
            temperature=temperature,
            rng=rng,
        )
        sequence.append(next_token_id)
        predicted_count += 1
        if next_token_id == eos_id:
            break

    return sequence, _tokens_to_text(sequence, sp)


def _save_predict_result(lines: list[str]) -> str:
    out_path = Path(get_chabsa_next_word_predict_path())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return str(out_path)


def run_predict(
    option: str = "none",
    text: str = "",
    *,
    top_k: int = DEFAULT_TOP_K,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: int | None = None,
    ctx=None,
) -> dict[str, str | int | float]:
    if ctx is None:
        ctx = _create_context()

    sp = _load_sentencepiece_model()
    model = _load_trained_model()
    rng = np.random.default_rng(seed)

    lines: list[str] = []
    if option == "none":
        _, output_text = predict_sequence(
            model=model,
            sp=sp,
            seed_text="",
            top_k=top_k,
            temperature=temperature,
            rng=rng,
        )
        lines.append(output_text)
    elif option == "none10":
        for index in range(10):
            _, output_text = predict_sequence(
                model=model,
                sp=sp,
                seed_text="",
                top_k=top_k,
                temperature=temperature,
                rng=rng,
            )
            lines.append(f"[{index + 1}] {output_text}")
    elif option == "take":
        _, output_text = predict_sequence(
            model=model,
            sp=sp,
            seed_text=text,
            top_k=top_k,
            temperature=temperature,
            rng=rng,
        )
        lines.append(output_text)
    else:
        raise ValueError(f"Unknown option: {option}")

    out_path = _save_predict_result(lines)
    ctx.info(f"[{USECASE_ID}/{UCB_ID}] saved prediction: {out_path}")

    return {
        "output": out_path,
        "lines": len(lines),
        "option": option,
        "top_k": int(top_k),
        "temperature": float(temperature),
        "seed": int(seed) if seed is not None else -1,
    }


def main() -> None:
    _ensure_src_on_sys_path()
    from project_context.project_context import ProjectContext

    ctx = ProjectContext()

    option = sys.argv[1] if len(sys.argv) > 1 else "none"
    take_text = ""
    if option == "take":
        if len(sys.argv) < 3:
            raise ValueError("take option requires a text argument")
        take_text = " ".join(sys.argv[2:])

    top_k = int(os.environ.get("CHABSA_PREDICT_TOP_K", str(DEFAULT_TOP_K)))
    temperature = float(os.environ.get("CHABSA_PREDICT_TEMPERATURE", str(DEFAULT_TEMPERATURE)))
    seed_text = os.environ.get("CHABSA_PREDICT_SEED", "").strip()
    seed = int(seed_text) if seed_text else None

    ctx.info(f"[{USECASE_ID}/{UCB_ID}] start option='{option}'")
    try:
        result = run_predict(
            option=option,
            text=take_text,
            top_k=top_k,
            temperature=temperature,
            seed=seed,
            ctx=ctx,
        )
        ctx.info(f"[{USECASE_ID}/{UCB_ID}] result: {result}")
    finally:
        ctx.info(f"[{USECASE_ID}/{UCB_ID}] end")


if __name__ == "__main__":
    main()

"""
python src/uca009/ucb002/main.py none
python src/uca009/ucb002/main.py none10
python src/uca009/ucb002/main.py take 今日はとても良い決算でした

# 例: 上位2件からランダムサンプリング（デフォルト）
CHABSA_PREDICT_TOP_K=2 python src/uca009/ucb002/main.py none10

# 例: 乱数固定（再現可能）
CHABSA_PREDICT_TOP_K=2 CHABSA_PREDICT_SEED=42 python src/uca009/ucb002/main.py none10

# 例: 温度を上げてブレを増やす
CHABSA_PREDICT_TOP_K=2 CHABSA_PREDICT_TEMPERATURE=1.3 python src/uca009/ucb002/main.py take 今日はとても良い決算でした
"""
