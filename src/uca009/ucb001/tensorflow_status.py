from __future__ import annotations

import tensorflow as tf


def _bool_text(value: bool) -> str:
    return "yes" if value else "no"


def _print_tensorflow_status() -> None:
    tf_version = tf.__version__
    tf_git_version = getattr(tf.version, "GIT_VERSION", "unknown")
    tf_compiler_version = getattr(tf.version, "COMPILER_VERSION", "unknown")

    is_cuda_build = bool(tf.test.is_built_with_cuda())
    is_rocm_build = bool(tf.test.is_built_with_rocm())

    physical_cpus = tf.config.list_physical_devices("CPU")
    physical_gpus = tf.config.list_physical_devices("GPU")
    logical_cpus = tf.config.list_logical_devices("CPU")
    logical_gpus = tf.config.list_logical_devices("GPU")

    runtime_device = "GPU" if logical_gpus else "CPU"

    print("=== TensorFlow Status ===")
    print(f"tensorflow.version: {tf_version}")
    print(f"tensorflow.git_version: {tf_git_version}")
    print(f"tensorflow.compiler_version: {tf_compiler_version}")
    print(f"build.cuda: {_bool_text(is_cuda_build)}")
    print(f"build.rocm: {_bool_text(is_rocm_build)}")

    print("=== Device Summary ===")
    print(f"runtime.device: {runtime_device}")
    print(f"physical.cpu.count: {len(physical_cpus)}")
    print(f"physical.gpu.count: {len(physical_gpus)}")
    print(f"logical.cpu.count: {len(logical_cpus)}")
    print(f"logical.gpu.count: {len(logical_gpus)}")
    print(f"gpu.detected: {_bool_text(len(physical_gpus) > 0)}")

    if physical_gpus:
        print("=== Physical GPU Devices ===")
        for index, device in enumerate(physical_gpus, start=1):
            print(f"gpu[{index}]: {device.name}")
    else:
        print("No physical GPU devices detected.")

    if logical_gpus:
        print("=== Logical GPU Devices ===")
        for index, device in enumerate(logical_gpus, start=1):
            print(f"logical_gpu[{index}]: {device.name}")

    build_info_getter = getattr(tf.sysconfig, "get_build_info", None)
    if callable(build_info_getter):
        build_info = build_info_getter()
        cuda_version = build_info.get("cuda_version")
        cudnn_version = build_info.get("cudnn_version")
        if cuda_version is not None:
            print(f"build.cuda_version: {cuda_version}")
        if cudnn_version is not None:
            print(f"build.cudnn_version: {cudnn_version}")


def main() -> None:
    _print_tensorflow_status()


if __name__ == "__main__":
    main()

"""
python src/uca009/ucb001/tensorflow_status.py
"""
