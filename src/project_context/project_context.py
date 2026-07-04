
import logging
import os
import time
from pathlib import Path


def setup_logger(outdir="/tmp", log_filename="experiment.log"):
    # 1. ロガーの生成とログレベルの設定
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # 既にハンドラが登録されている場合は、二重出力防止のためスキップ
    if logger.hasHandlers():
        return logger

    # 2. 共通のフォーマットを設定（時間、ログレベル、メッセージ）
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 3. 標準出力用のハンドラ設定
    stdout_handler = logging.StreamHandler()
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    # 4. ファイル出力用のハンドラ設定
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)  # ディレクトリがなければ作成
    log_file = out_path / log_filename

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


class _LogSpan:
    def __init__(self, ctx, name: str):
        self._ctx = ctx
        self._name = name
        self._start = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        self._ctx.info(f"[span] start: {self._name}")
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.perf_counter() - self._start
        if exc is None:
            self._ctx.info(f"[span] end: {self._name} elapsed={elapsed:.3f}s")
        else:
            self._ctx.error(
                f"[span] error: {self._name} elapsed={elapsed:.3f}s error={exc}"
            )
        return False


def log_span(ctx, name: str):
    return _LogSpan(ctx, name)


class ProjectContext:
    def __init__(self):
        self.logger = setup_logger(outdir="/tmp", log_filename="experiment.log")

    def info(self, msg):
        self.logger.info(msg)

    def warn(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def log_span(self, name: str):
        return log_span(self, name)
