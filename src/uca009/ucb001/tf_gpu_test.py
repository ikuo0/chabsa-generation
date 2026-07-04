import os

# TensorFlowのC++ログを少し抑制
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import tensorflow as tf


def main() -> None:
    print("TensorFlow version:", tf.__version__)

    # GPU一覧を取得
    gpus = tf.config.list_physical_devices("GPU")
    print("Physical GPUs:", gpus)

    if not gpus:
        raise RuntimeError(
            "TensorFlowからGPUが見えていません。"
            "devcontainerの --gpus=all、Docker Desktop/WSL2/NVIDIA Driver を確認してください。"
        )

    # GPUメモリを必要分だけ確保する設定
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # 演算配置ログを出す
    # MatMul や Dense が GPU に配置されるか確認できます。
    tf.debugging.set_log_device_placement(True)

    # 明示的にGPU:0上でモデルとデータを作る
    with tf.device("/GPU:0"):
        # ダミーデータ
        # 6000次元入力、6000クラス分類
        sample_count = 4096
        input_dim = 6000
        class_count = 6000

        x_train = tf.random.normal((sample_count, input_dim), dtype=tf.float32)
        y_train = tf.random.uniform(
            (sample_count,),
            minval=0,
            maxval=class_count,
            dtype=tf.int32,
        )

        model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(input_dim,)),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(class_count),
            ]
        )

        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["sparse_categorical_accuracy"],
        )

    model.summary()

    print("\n--- start training: 1 epoch only ---")
    history = model.fit(
        x_train,
        y_train,
        epochs=1,
        batch_size=128,
        verbose=1,
    )

    print("\n--- result ---")
    print("loss:", history.history["loss"][-1])
    print(
        "sparse_categorical_accuracy:",
        history.history["sparse_categorical_accuracy"][-1],
    )

    print("\nLogical devices:", tf.config.list_logical_devices())
    print("\nGPU test finished successfully.")


if __name__ == "__main__":
    main()

"""
python src/uca009/ucb001/tf_gpu_test.py
"""