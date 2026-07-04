from __future__ import annotations

import csv
import os
import re
import sys
from pathlib import Path

import numpy as np
import sentencepiece
import tensorflow as tf
from scipy import sparse


USECASE_ID = "uca009"
UCB_ID = "ucb001"

DEFAULT_MAX_ITER = 50
DEFAULT_BATCH_SIZE = 256

_TF_DEVICE_CONFIGURED = False
_TF_RUNTIME_DEVICE = "CPU"


def _configure_tensorflow_device(ctx) -> str:
    global _TF_DEVICE_CONFIGURED, _TF_RUNTIME_DEVICE

    if _TF_DEVICE_CONFIGURED:
        return _TF_RUNTIME_DEVICE

    physical_gpus = tf.config.list_physical_devices("GPU")
    if not physical_gpus:
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError:
            # Device configuration was already finalized by TensorFlow runtime.
            pass
        _TF_RUNTIME_DEVICE = "CPU"
        ctx.info(f"[{USECASE_ID}/{UCB_ID}] GPU not found or disabled. fallback to CPU")
        _TF_DEVICE_CONFIGURED = True
        return _TF_RUNTIME_DEVICE

    try:
        for gpu in physical_gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        _TF_RUNTIME_DEVICE = "GPU"
        ctx.info(
            f"[{USECASE_ID}/{UCB_ID}] GPU detected: count={len(physical_gpus)}. use GPU runtime"
        )
    except RuntimeError as exc:
        logical_gpus = tf.config.list_logical_devices("GPU")
        _TF_RUNTIME_DEVICE = "GPU" if logical_gpus else "CPU"
        ctx.warn(
            f"[{USECASE_ID}/{UCB_ID}] GPU config skipped (already initialized): {exc}. "
            f"runtime={_TF_RUNTIME_DEVICE}"
        )
    except Exception as exc:
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        _TF_RUNTIME_DEVICE = "CPU"
        ctx.warn(
            f"[{USECASE_ID}/{UCB_ID}] GPU initialization failed ({exc}). fallback to CPU"
        )

    _TF_DEVICE_CONFIGURED = True
    return _TF_RUNTIME_DEVICE


def _select_training_device(ctx, x_csr: sparse.csr_matrix) -> str:
    runtime_device = _configure_tensorflow_device(ctx)
    if runtime_device != "GPU":
        return "/CPU:0"

    return "/GPU:0"


def _log_tensorflow_runtime(ctx) -> None:
    tf_version = tf.__version__
    tf_git_version = getattr(tf.version, "GIT_VERSION", "unknown")
    tf_compiler_version = getattr(tf.version, "COMPILER_VERSION", "unknown")
    tf_built_with_cuda = bool(tf.test.is_built_with_cuda())
    tf_built_with_rocm = bool(tf.test.is_built_with_rocm())

    physical_cpus = tf.config.list_physical_devices("CPU")
    physical_gpus = tf.config.list_physical_devices("GPU")
    logical_cpus = tf.config.list_logical_devices("CPU")
    logical_gpus = tf.config.list_logical_devices("GPU")

    runtime_device = "GPU" if logical_gpus else "CPU"

    ctx.info(
        f"[{USECASE_ID}/{UCB_ID}] tensorflow runtime: "
        f"version={tf_version}, git={tf_git_version}, compiler={tf_compiler_version}"
    )
    ctx.info(
        f"[{USECASE_ID}/{UCB_ID}] tensorflow build: "
        f"cuda={tf_built_with_cuda}, rocm={tf_built_with_rocm}"
    )
    ctx.info(
        f"[{USECASE_ID}/{UCB_ID}] tensorflow devices: "
        f"physical_cpu={len(physical_cpus)}, physical_gpu={len(physical_gpus)}, "
        f"logical_cpu={len(logical_cpus)}, logical_gpu={len(logical_gpus)}, "
        f"runtime={runtime_device}"
    )

    if logical_gpus:
        gpu_names = ", ".join(device.name for device in logical_gpus)
        ctx.info(f"[{USECASE_ID}/{UCB_ID}] active gpu devices: {gpu_names}")
    else:
        ctx.info(f"[{USECASE_ID}/{UCB_ID}] active device: CPU")


def _project_root() -> Path:
    # src/uca009/ucb001/main.py -> project root
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


def get_chabsa_next_word_relpath() -> str:
    return "/data/uca002/ucb002/chabsa_next_word.npy"


def get_chabsa_next_word_path() -> str:
    rel = get_chabsa_next_word_relpath().lstrip("/")
    return str(_project_root() / rel)


def get_spm_model_relpath() -> str:
    return "/data/uca001/ucb006/spm_model.model"


def get_spm_model_path() -> str:
    rel = get_spm_model_relpath().lstrip("/")
    return str(_project_root() / rel)


def get_chabsa_next_word_model_h5_relpath() -> str:
    return "/data/uca009/ucb001/chabsa_next_word_model.h5"


def get_chabsa_next_word_model_h5_path() -> str:
    rel = get_chabsa_next_word_model_h5_relpath().lstrip("/")
    return str(_project_root() / rel)


def get_chabsa_next_word_model_keras_relpath() -> str:
    return "/data/uca009/ucb001/chabsa_next_word_model.keras"


def get_chabsa_next_word_model_keras_path() -> str:
    rel = get_chabsa_next_word_model_keras_relpath().lstrip("/")
    return str(_project_root() / rel)


def get_chabsa_next_word_model_epoch_h5_relpath(epoch: int) -> str:
    return f"/data/uca009/ucb001/chabsa_next_word_model_epoch_{epoch}.h5"


def get_chabsa_next_word_model_epoch_h5_path(epoch: int) -> str:
    rel = get_chabsa_next_word_model_epoch_h5_relpath(epoch).lstrip("/")
    return str(_project_root() / rel)


def get_chabsa_next_word_model_epoch_keras_relpath(epoch: int) -> str:
    return f"/data/uca009/ucb001/chabsa_next_word_model_epoch_{epoch}.keras"


def get_chabsa_next_word_model_epoch_keras_path(epoch: int) -> str:
    rel = get_chabsa_next_word_model_epoch_keras_relpath(epoch).lstrip("/")
    return str(_project_root() / rel)


def get_chabsa_next_word_train_log_tsv_relpath() -> str:
    return "/data/uca009/ucb001/chabsa_next_word_train_log.tsv"


def get_chabsa_next_word_train_log_tsv_path() -> str:
    rel = get_chabsa_next_word_train_log_tsv_relpath().lstrip("/")
    return str(_project_root() / rel)


class _PerEpochSaveAndLogCallback(tf.keras.callbacks.Callback):
    def __init__(self, ctx, *, epoch_offset: int = 0, reset_log: bool = True):
        super().__init__()
        self._ctx = ctx
        self._epoch_offset = int(epoch_offset)
        self._tsv_path = Path(get_chabsa_next_word_train_log_tsv_path())
        self._tsv_path.parent.mkdir(parents=True, exist_ok=True)
        if reset_log and self._tsv_path.exists():
            self._tsv_path.unlink()

        if reset_log or not self._tsv_path.exists():
            with self._tsv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(["epoch", "loss", "sparse_categorical_accuracy"])

    @property
    def tsv_path(self) -> str:
        return str(self._tsv_path)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch_number = self._epoch_offset + int(epoch) + 1
        loss = float(logs.get("loss", -1.0))
        acc = float(logs.get("sparse_categorical_accuracy", -1.0))

        epoch_h5_path = Path(get_chabsa_next_word_model_epoch_h5_path(epoch_number))
        epoch_keras_path = Path(get_chabsa_next_word_model_epoch_keras_path(epoch_number))
        self.model.save(str(epoch_h5_path))
        self.model.save(str(epoch_keras_path))

        with self._tsv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow([epoch_number, f"{loss:.8f}", f"{acc:.8f}"])

        self._ctx.info(
            f"[{USECASE_ID}/{UCB_ID}] epoch={epoch_number} saved: "
            f"h5={epoch_h5_path.name}, keras={epoch_keras_path.name}, "
            f"loss={loss:.6f}, sparse_categorical_accuracy={acc:.6f}"
        )


def _parse_epoch_from_checkpoint_name(filename: str) -> int | None:
    h5_match = re.fullmatch(r"chabsa_next_word_model_epoch_(\d+)\.h5", filename)
    if h5_match is not None:
        return int(h5_match.group(1))

    keras_match = re.fullmatch(r"chabsa_next_word_model_epoch_(\d+)\.keras", filename)
    if keras_match is not None:
        return int(keras_match.group(1))

    return None


def _find_latest_epoch_checkpoint() -> tuple[int, Path] | None:
    base_dir = Path(get_chabsa_next_word_model_h5_path()).parent
    if not base_dir.exists():
        return None

    latest: tuple[int, Path] | None = None
    for candidate in base_dir.glob("chabsa_next_word_model_epoch_*"):
        epoch = _parse_epoch_from_checkpoint_name(candidate.name)
        if epoch is None:
            continue
        if latest is None or epoch > latest[0]:
            latest = (epoch, candidate)

    return latest


def _read_last_epoch_from_tsv() -> int:
    tsv_path = Path(get_chabsa_next_word_train_log_tsv_path())
    if not tsv_path.exists():
        return 0

    last_epoch = 0
    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            epoch_text = (row.get("epoch") or "").strip()
            if not epoch_text:
                continue
            try:
                epoch_value = int(epoch_text)
            except ValueError:
                continue
            last_epoch = max(last_epoch, epoch_value)

    return last_epoch


def _cleanup_all_training_outputs(ctx) -> None:
    targets = [
        Path(get_chabsa_next_word_model_h5_path()),
        Path(get_chabsa_next_word_model_keras_path()),
        Path(get_chabsa_next_word_train_log_tsv_path()),
    ]
    model_dir = Path(get_chabsa_next_word_model_h5_path()).parent
    if model_dir.exists():
        targets.extend(model_dir.glob("chabsa_next_word_model_epoch_*.h5"))
        targets.extend(model_dir.glob("chabsa_next_word_model_epoch_*.keras"))

    for target in targets:
        if target.exists():
            target.unlink()

    model_dir.mkdir(parents=True, exist_ok=True)
    ctx.info(f"[{USECASE_ID}/{UCB_ID}] reset completed: cleaned old models and logs")


def _load_vocab_size_from_spm() -> int:
    spm_path = Path(get_spm_model_path())
    if not spm_path.exists():
        raise FileNotFoundError(f"SentencePiece model not found: {spm_path}")

    processor = sentencepiece.SentencePieceProcessor()
    processor.Load(str(spm_path))
    return int(processor.GetPieceSize())


def _csr_to_tf_sparse_tensor(x_csr: sparse.csr_matrix) -> tf.sparse.SparseTensor:
    x_coo = x_csr.tocoo()
    indices = np.column_stack((x_coo.row, x_coo.col)).astype(np.int64)
    values = x_coo.data.astype(np.float32)
    dense_shape = np.array(x_csr.shape, dtype=np.int64)
    sparse_tensor = tf.sparse.SparseTensor(
        indices=indices,
        values=values,
        dense_shape=dense_shape,
    )
    return tf.sparse.reorder(sparse_tensor)


def _build_model(vocab_size: int, *, sparse_input: bool = True) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(vocab_size,), sparse=sparse_input)
    x = tf.keras.layers.Dense(512, activation="relu")(inputs)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    outputs = tf.keras.layers.Dense(vocab_size)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    return model


def _make_dense_batch_dataset(
    x_csr: sparse.csr_matrix,
    y_np: np.ndarray,
    batch_size: int,
) -> tf.data.Dataset:
    shape = x_csr.shape
    if shape is None:
        raise ValueError("CSR matrix shape is undefined")
    sample_count = int(shape[0])
    feature_dim = int(shape[1])

    def _generator():
        for start in range(0, sample_count, batch_size):
            end = min(start + batch_size, sample_count)
            x_dense = x_csr[start:end].toarray().astype(np.float32, copy=False)
            y_batch = y_np[start:end].astype(np.int32, copy=False)
            yield x_dense, y_batch

    output_signature = (
        tf.TensorSpec(shape=(None, feature_dim), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
    )
    return tf.data.Dataset.from_generator(_generator, output_signature=output_signature)


def _model_expects_sparse_input(model: tf.keras.Model) -> bool:
    if not model.inputs:
        return False
    return bool(getattr(model.inputs[0], "sparse", False))


def train_next_word_model(max_iter: int = DEFAULT_MAX_ITER, *, mode: str = "reset", ctx=None) -> dict[str, str | int]:
    if ctx is None:
        ctx = _create_context()

    _configure_tensorflow_device(ctx)

    if max_iter <= 0:
        raise ValueError(f"max_iter must be positive: {max_iter}")

    if mode not in ("reset", "continue"):
        raise ValueError(f"mode must be 'reset' or 'continue': {mode}")

    x_path = Path(get_chabsa_vector_path())
    y_path = Path(get_chabsa_next_word_path())
    if not x_path.exists():
        raise FileNotFoundError(f"X data not found: {x_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Y data not found: {y_path}")

    ctx.info(f"[{USECASE_ID}/{UCB_ID}] loading input data")
    x_csr = sparse.load_npz(x_path)
    if not sparse.isspmatrix_csr(x_csr):
        x_csr = x_csr.tocsr()
    y_np = np.load(y_path)

    vocab_size = _load_vocab_size_from_spm()
    if int(x_csr.shape[1]) != vocab_size:
        raise ValueError(
            "Input dimension mismatch: "
            f"x_dim={x_csr.shape[1]}, vocab_size={vocab_size}"
        )

    if int(x_csr.shape[0]) != int(y_np.shape[0]):
        raise ValueError(
            "Input sample count mismatch: "
            f"x_rows={x_csr.shape[0]}, y_rows={y_np.shape[0]}"
        )

    x_tf = _csr_to_tf_sparse_tensor(x_csr)
    y_tf = tf.constant(y_np.astype(np.int32))
    training_device = _select_training_device(ctx, x_csr)
    ctx.info(f"[{USECASE_ID}/{UCB_ID}] training device selected: {training_device}")
    batch_size = int(os.environ.get("CHABSA_BATCH_SIZE", str(DEFAULT_BATCH_SIZE)))
    if batch_size <= 0:
        raise ValueError(f"CHABSA_BATCH_SIZE must be positive: {batch_size}")

    use_dense_batches = training_device == "/GPU:0" and sparse.isspmatrix_csr(x_csr)
    if use_dense_batches:
        ctx.info(
            f"[{USECASE_ID}/{UCB_ID}] use dense mini-batch dataset on GPU "
            f"for sparse source matrix (batch_size={batch_size})"
        )
    else:
        ctx.info(
            f"[{USECASE_ID}/{UCB_ID}] use SparseTensor training path "
            f"(batch_size={batch_size})"
        )

    if mode == "reset":
        _cleanup_all_training_outputs(ctx)
        with tf.device(training_device):
            model = _build_model(vocab_size, sparse_input=not use_dense_batches)
        initial_epoch = 0
        callback = _PerEpochSaveAndLogCallback(ctx, epoch_offset=0, reset_log=True)
    else:
        latest_checkpoint = _find_latest_epoch_checkpoint()
        if latest_checkpoint is not None:
            initial_epoch = int(latest_checkpoint[0])
            with tf.device(training_device):
                loaded_model = tf.keras.models.load_model(str(latest_checkpoint[1]))
                if use_dense_batches and _model_expects_sparse_input(loaded_model):
                    # Keep learned weights while switching sparse input model to dense input model.
                    model = _build_model(vocab_size, sparse_input=False)
                    model.set_weights(loaded_model.get_weights())
                else:
                    model = loaded_model
            ctx.info(
                f"[{USECASE_ID}/{UCB_ID}] continue from epoch checkpoint: "
                f"epoch={initial_epoch}, file={latest_checkpoint[1]}"
            )
        else:
            initial_epoch = _read_last_epoch_from_tsv()
            with tf.device(training_device):
                model = _build_model(vocab_size, sparse_input=not use_dense_batches)
            if initial_epoch > 0:
                ctx.warn(
                    f"[{USECASE_ID}/{UCB_ID}] train log exists up to epoch={initial_epoch}, "
                    "but epoch checkpoint model not found. start from fresh weights."
                )
            else:
                ctx.info(f"[{USECASE_ID}/{UCB_ID}] no previous checkpoint, start new training")

        callback = _PerEpochSaveAndLogCallback(
            ctx,
            epoch_offset=initial_epoch,
            reset_log=(initial_epoch == 0),
        )

    remaining_epochs = int(max_iter) - int(initial_epoch)
    if remaining_epochs <= 0:
        ctx.info(
            f"[{USECASE_ID}/{UCB_ID}] no remaining epochs: "
            f"max_iter={max_iter}, already_trained={initial_epoch}"
        )

        h5_path = Path(get_chabsa_next_word_model_h5_path())
        keras_path = Path(get_chabsa_next_word_model_keras_path())
        h5_path.parent.mkdir(parents=True, exist_ok=True)
        if h5_path.exists():
            h5_path.unlink()
        if keras_path.exists():
            keras_path.unlink()
        model.save(str(h5_path))
        model.save(str(keras_path))

        return {
            "mode": mode,
            "h5": str(h5_path),
            "keras": str(keras_path),
            "train_log_tsv": callback.tsv_path,
            "samples": int(x_csr.shape[0]),
            "vocab_size": int(vocab_size),
            "epochs": int(max_iter),
            "resume_from_epoch": int(initial_epoch),
            "training_device": training_device,
            "final_loss": "-1.000000",
            "final_sparse_categorical_accuracy": "-1.000000",
        }

    ctx.info(
        f"[{USECASE_ID}/{UCB_ID}] training start "
        f"samples={x_csr.shape[0]}, vocab_size={vocab_size}, "
        f"target_epochs={max_iter}, resume_from={initial_epoch}, run_epochs={remaining_epochs}"
    )
    training_steps = int(np.ceil(float(x_csr.shape[0]) / float(batch_size)))
    with tf.device(training_device):
        if use_dense_batches:
            # `from_generator` dataset can be exhausted when `steps_per_epoch` is fixed.
            # Repeat explicitly so each epoch always consumes full steps.
            train_dataset = (
                _make_dense_batch_dataset(x_csr, y_np, batch_size)
                .repeat()
                .prefetch(tf.data.AUTOTUNE)
            )
            history = model.fit(
                train_dataset,
                initial_epoch=0,
                epochs=remaining_epochs,
                steps_per_epoch=training_steps,
                verbose=1,
                callbacks=[callback],
            )
        else:
            history = model.fit(
                x_tf,
                y_tf,
                initial_epoch=0,
                epochs=remaining_epochs,
                batch_size=batch_size,
                verbose=1,
                callbacks=[callback],
            )

    h5_path = Path(get_chabsa_next_word_model_h5_path())
    keras_path = Path(get_chabsa_next_word_model_keras_path())
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    for output_path in (h5_path, keras_path):
        if output_path.exists():
            output_path.unlink()

    model.save(str(h5_path))
    model.save(str(keras_path))
    ctx.info(
        f"[{USECASE_ID}/{UCB_ID}] saved models: "
        f"h5={h5_path}, keras={keras_path}"
    )

    final_loss = float(history.history["loss"][-1]) if history.history.get("loss") else -1.0
    final_acc = (
        float(history.history["sparse_categorical_accuracy"][-1])
        if history.history.get("sparse_categorical_accuracy")
        else -1.0
    )

    return {
        "mode": mode,
        "h5": str(h5_path),
        "keras": str(keras_path),
        "train_log_tsv": callback.tsv_path,
        "samples": int(x_csr.shape[0]),
        "vocab_size": int(vocab_size),
        "epochs": int(max_iter),
        "resume_from_epoch": int(initial_epoch),
        "training_device": training_device,
        "final_loss": f"{final_loss:.6f}",
        "final_sparse_categorical_accuracy": f"{final_acc:.6f}",
    }


def main() -> None:
    _ensure_src_on_sys_path()
    from project_context.project_context import ProjectContext

    ctx = ProjectContext()
    option = sys.argv[1] if len(sys.argv) > 1 else ""
    ctx.info(f"[{USECASE_ID}/{UCB_ID}] start option='{option or 'reset'}'")
    _configure_tensorflow_device(ctx)
    _log_tensorflow_runtime(ctx)

    try:
        if option in ("", "reset"):
            result = train_next_word_model(max_iter=DEFAULT_MAX_ITER, mode="reset", ctx=ctx)
            ctx.info(f"[{USECASE_ID}/{UCB_ID}] result: {result}")
            return

        if option == "continue":
            result = train_next_word_model(max_iter=DEFAULT_MAX_ITER, mode="continue", ctx=ctx)
            ctx.info(f"[{USECASE_ID}/{UCB_ID}] result: {result}")
            return

        raise ValueError(f"Unknown option: {option}")
    finally:
        ctx.info(f"[{USECASE_ID}/{UCB_ID}] end")


if __name__ == "__main__":
    main()

"""
python src/uca009/ucb001/main.py
python src/uca009/ucb001/main.py reset
python src/uca009/ucb001/main.py continue
"""
