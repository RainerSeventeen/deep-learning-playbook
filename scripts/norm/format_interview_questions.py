#!/usr/bin/env python3
"""
格式化 interview/questions.md。

规则:
1. 每个 Markdown 标题后保留且只保留一个空行
2. 除标题外，所有非空内容都转成当前标题块下的有序列表

默认原地覆盖文件。
使用 --stdout 可输出到 stdout，使用 --check 可只检查是否需要格式化。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = ROOT / "interview/questions.md"

HEADING_RE = re.compile(r"^#{1,6}\s+\S")
ORDERED_ITEM_RE = re.compile(r"^\d+\.(?:\s+|$)")
UNORDERED_ITEM_RE = re.compile(r"^[-*+](?:\s+|$)")


def normalize_item(line: str) -> str:
    text = line.strip()
    text = ORDERED_ITEM_RE.sub("", text)
    text = UNORDERED_ITEM_RE.sub("", text)
    return text.strip()


def format_questions(text: str) -> str:
    output: list[str] = []
    item_index = 0

    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue

        if HEADING_RE.match(stripped):
            if output and output[-1] != "":
                output.append("")
            output.append(stripped)
            output.append("")
            item_index = 0
            continue

        item_text = normalize_item(stripped)
        if not item_text:
            continue

        item_index += 1
        output.append(f"{item_index}. {item_text}")

    while output and output[-1] == "":
        output.pop()

    if not output:
        return ""

    return "\n".join(output) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="格式化 interview/questions.md：标题后空一行，非标题内容全部转为有序列表。"
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_PATH),
        help="要格式化的 Markdown 文件路径，默认是 interview/questions.md",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="只检查文件是否已符合格式，未通过时返回 1",
    )
    mode.add_argument(
        "--stdout",
        action="store_true",
        help="输出到 stdout，而不是原地覆盖文件",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = Path(args.path).resolve()

    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    original = path.read_text(encoding="utf-8")
    formatted = format_questions(original)

    if args.check:
        if original == formatted:
            print(f"Already formatted: {path}")
            return 0
        print(f"Needs formatting: {path}")
        return 1

    if args.stdout:
        sys.stdout.write(formatted)
        return 0

    path.write_text(formatted, encoding="utf-8")
    print(f"Formatted: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
