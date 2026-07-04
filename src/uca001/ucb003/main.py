from __future__ import annotations

import os
import re
import sqlite3
import sys
import unicodedata
from pathlib import Path


USECASE_ID = "uca001"
UCB_ID = "ucb003"
TABLE_NAME = "chabsa"


def _project_root() -> Path:
    # src/uca001/ucb003/main.py -> project root
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


def get_chabsa_normalized_db_relpath() -> str:
    return "/data/uca001/ucb003/chabsa_normalized.db"


def get_chabsa_normalized_db_path() -> str:
    rel = get_chabsa_normalized_db_relpath().lstrip("/")
    return str(_project_root() / rel)


def _get_origin_db_path() -> str:
    _ensure_src_on_sys_path()
    from uca001.ucb002.main import get_chabsa_origin_db_path

    return get_chabsa_origin_db_path()


def unicode_normalize(s):
    # Unicode正規化を行う
    # 例: "ＡＢＣａｂｃ１２３" -> "ＡＢＣａｂｃ１２３" (全角英数字はそのまま、半角スペースは全角スペースに変換される)
    return unicodedata.normalize("NFKC", s)


def to_full_width(s):
    return s.translate(str.maketrans({chr(i): chr(i + 0xFEE0) for i in range(0x21, 0x7F)}))


def pre_normalize(s):
    s = unicode_normalize(s)
    s = to_full_width(s)
    return s


##############################
# 以下の処理は、全て全角統一後に行うことを想定している
##############################


def unify_brackets(s: str) -> str:
    # 事前に to_full_width(s) 済みである前提
    # 始まり括弧として扱う文字
    open_brackets = (
        "（"
        "［"
        "｛"
        "〈"
        "《"
        "「"
        "『"
        "【"
        "〔"
        "〖"
        "〘"
        "〚"
        "〝"
        "‘"
        "“"
        "｟"
        "〿"
        "⦅"
        "⦃"
        "⟨"
        "⟪"
        "⦇"
        "⟦"
        "⟬"
        "⟮"
        "⦉"
        "⦋"
        "⦍"
        "⦏"
        "⦑"
        "⦓"
        "⦕"
        "⦗"
    )

    # 終わり括弧として扱う文字
    close_brackets = (
        "）"
        "］"
        "｝"
        "〉"
        "》"
        "」"
        "』"
        "】"
        "〕"
        "〗"
        "〙"
        "〛"
        "〟"
        "’"
        "”"
        "｠"
        "⦆"
        "⦄"
        "⟩"
        "⟫"
        "⦈"
        "⟧"
        "⟭"
        "⟯"
        "⦊"
        "⦌"
        "⦎"
        "⦐"
        "⦒"
        "⦔"
        "⦖"
        "⦘"
    )

    open_pattern = "[" + re.escape(open_brackets) + "]"
    close_pattern = "[" + re.escape(close_brackets) + "]"

    s = re.sub(open_pattern, "（", s)
    s = re.sub(close_pattern, "）", s)

    return s


def unify_horizontal_bars(s: str) -> str:
    # 事前に to_full_width(s) 済みである前提

    horizontal_bars = (
        "－"  # fullwidth hyphen-minus
        "−"  # minus sign
        "‐"  # hyphen
        "-"  # non-breaking hyphen
        "‒"  # figure dash
        "–"  # en dash
        "—"  # em dash
        "―"  # horizontal bar
        "⁃"  # hyphen bullet
        "﹘"  # small em dash
        "﹣"  # small hyphen-minus
        "ー"  # Japanese prolonged sound mark
        "ｰ"  # halfwidth katakana-hiragana prolonged sound mark
        "─"  # box drawings light horizontal
        "━"  # box drawings heavy horizontal
        "╌"  # box drawings light double dash horizontal
        "╍"  # box drawings heavy double dash horizontal
        "┄"  # box drawings light triple dash horizontal
        "┅"  # box drawings heavy triple dash horizontal
        "┈"  # box drawings light quadruple dash horizontal
        "┉"  # box drawings heavy quadruple dash horizontal
        "╴"  # box drawings light left
        "╶"  # box drawings light right
        "╸"  # box drawings heavy left
        "╺"  # box drawings heavy right
    )

    pattern = "[" + re.escape(horizontal_bars) + "]"
    return re.sub(pattern, "－", s)


def unify_alphabet_case(s: str) -> str:
    # 事前に to_full_width(s) 済みである前提
    # 全角英小文字を全角英大文字に統一する

    lower_alphabets = "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ"
    upper_alphabets = "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"

    translation_table = str.maketrans(lower_alphabets, upper_alphabets)

    return s.translate(translation_table)


def unify_commas(s: str) -> str:
    # 読点・カンマっぽいものを「、」に統一する

    comma_chars = ("、" "，" "," "､" "﹐" "﹑")

    pattern = "[" + re.escape(comma_chars) + "]"
    return re.sub(pattern, "、", s)


def unify_periods(s: str) -> str:
    # 句点・ピリオドっぽいものを「．」に統一する

    period_chars = ("。" "．" "." "｡" "﹒")

    pattern = "[" + re.escape(period_chars) + "]"
    return re.sub(pattern, "．", s)


def unify_slashes(s: str) -> str:
    slash_chars = ("／" "/" "＼" "\\" "⁄" "∕")

    pattern = "[" + re.escape(slash_chars) + "]"
    return re.sub(pattern, "／", s)


def unify_waves(s: str) -> str:
    wave_chars = ("〜" "～" "~" "∼" "∽" "∿")

    pattern = "[" + re.escape(wave_chars) + "]"
    return re.sub(pattern, "～", s)


def unify_quotes(s: str) -> str:
    open_quotes = ("“" "‘" "〝" "＂" '"' "'")

    close_quotes = ("”" "’" "〟" "＂" '"' "'")

    s = re.sub("[" + re.escape(open_quotes) + "]", "「", s)
    s = re.sub("[" + re.escape(close_quotes) + "]", "」", s)

    return s


def unify_digits_012(s: str) -> str:
    # 事前に to_full_width(s) 済みである前提
    # [0]     -> ０
    # [1, 2]  -> １
    # [!012]  -> ３

    digit_table = str.maketrans(
        {
            "０": "０",
            "１": "１",
            "２": "１",
            "３": "３",
            "４": "３",
            "５": "３",
            "６": "３",
            "７": "３",
            "８": "３",
            "９": "３",
        }
    )

    return s.translate(digit_table)


def _normalize_legacy(sentence: str) -> str:
    sentence = pre_normalize(sentence)
    sentence = unify_brackets(sentence)
    sentence = unify_horizontal_bars(sentence)
    sentence = unify_alphabet_case(sentence)
    sentence = unify_commas(sentence)
    sentence = unify_periods(sentence)
    sentence = unify_slashes(sentence)
    sentence = unify_waves(sentence)
    sentence = unify_quotes(sentence)
    sentence = unify_digits_012(sentence)
    return sentence


def normalize_text(ctx=None) -> None:
    if ctx is None:
        ctx = _create_context()

    origin_db_file = _get_origin_db_path()
    normalized_db_file = Path(get_chabsa_normalized_db_path())
    normalized_db_file.parent.mkdir(parents=True, exist_ok=True)

    if normalized_db_file.exists():
        ctx.info(f"Removing existing normalized DB file: {normalized_db_file}")
        normalized_db_file.unlink()

    with sqlite3.connect(normalized_db_file) as write_conn:
        write_c = write_conn.cursor()
        write_c.execute(
            f"CREATE TABLE IF NOT EXISTS {TABLE_NAME} ("
            "row_id INTEGER PRIMARY KEY, "
            "sentence TEXT, "
            "target TEXT, "
            "polarity TEXT"
            ")"
        )

        with sqlite3.connect(origin_db_file) as read_conn:
            read_c = read_conn.cursor()
            read_c.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
            total_rows = read_c.fetchone()[0]
            ctx.info(f"[{USECASE_ID}/{UCB_ID}] total rows to normalize: {total_rows}")

            read_c.execute(f"SELECT row_id, sentence, target, polarity FROM {TABLE_NAME}")
            rows = read_c.fetchall()

            for irow, row in enumerate(rows, start=1):
                row_id, sentence, target, polarity = row

                # 旧実装に合わせて pre_normalize を2段で通す。
                normalized_sentence = pre_normalize(sentence)
                normalized_sentence = _normalize_legacy(normalized_sentence)

                write_c.execute(
                    f"INSERT INTO {TABLE_NAME} (row_id, sentence, target, polarity) VALUES (?, ?, ?, ?)",
                    (row_id, normalized_sentence, target, polarity),
                )

                if irow % 500 == 0:
                    ctx.info(f"[{USECASE_ID}/{UCB_ID}] normalized row {irow}/{total_rows}")


def _show_sample(sample_size: int = 3, ctx=None) -> None:
    if ctx is None:
        ctx = _create_context()

    origin_db_path = Path(_get_origin_db_path())
    normalized_db_path = Path(get_chabsa_normalized_db_path())

    if not origin_db_path.exists():
        raise FileNotFoundError(
            f"Origin DB not found: {origin_db_path}. Run 'python src/uca001/ucb002/main.py' first."
        )

    if not normalized_db_path.exists():
        raise FileNotFoundError(
            f"Normalized DB not found: {normalized_db_path}. Run 'python src/uca001/ucb003/main.py' first."
        )

    with sqlite3.connect(origin_db_path) as origin_conn:
        origin_cursor = origin_conn.cursor()
        origin_cursor.execute(
            f"SELECT row_id, sentence, target, polarity FROM {TABLE_NAME} ORDER BY RANDOM() LIMIT ?",
            (sample_size,),
        )
        origin_rows = origin_cursor.fetchall()

    if not origin_rows:
        ctx.info(f"[{USECASE_ID}/{UCB_ID}] no samples found in origin DB")
        return

    row_ids = [row[0] for row in origin_rows]
    placeholders = ",".join("?" for _ in row_ids)

    with sqlite3.connect(normalized_db_path) as normalized_conn:
        normalized_cursor = normalized_conn.cursor()
        normalized_cursor.execute(
            f"SELECT row_id, sentence FROM {TABLE_NAME} WHERE row_id IN ({placeholders})",
            row_ids,
        )
        normalized_rows = normalized_cursor.fetchall()

    normalized_by_row_id = {row_id: sentence for row_id, sentence in normalized_rows}

    for row_id, sentence_before, target, polarity in origin_rows:
        sentence_after = normalized_by_row_id.get(row_id)
        if sentence_after is None:
            ctx.warn(f"[{USECASE_ID}/{UCB_ID}] normalized row not found for row_id={row_id}")
            continue

        print("ーーーーー")
        print("ヘッダ情報")
        print(f"Row ID: {row_id}")
        print(f"Polarity: {polarity}")
        print(f"Target: {target}")
        print("ーーーーー")
        print("正規化前")
        print(sentence_before)
        print("正規化後")
        print(sentence_after)


def main() -> None:
    ctx = _create_context()
    option = sys.argv[1] if len(sys.argv) > 1 else ""
    ctx.info(f"[{USECASE_ID}/{UCB_ID}] start option='{option or 'default'}'")

    try:
        if option == "sample":
            _show_sample(3, ctx=ctx)
            return

        if option in ("",):
            normalize_text(ctx=ctx)
            return

        raise ValueError(f"Unknown option: {option}")
    finally:
        ctx.info(f"[{USECASE_ID}/{UCB_ID}] end")


if __name__ == "__main__":
    main()

"""
python src/uca001/ucb003/main.py
python src/uca001/ucb003/main.py sample
"""
