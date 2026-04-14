#!/usr/bin/env python3
"""
校验 papers/ 下所有论文笔记 Markdown 文件的格式合规性。

检查项:
  - Front matter 必须存在且为合法 YAML
  - Front matter 必填字段: paper（论文全名，非空字符串）
  - Markdown 内部链接（非 http/https 开头）目标文件须在仓库中真实存在

用法:
  python3 scripts/ci/check_papers.py

依赖:
  PyYAML（pip install pyyaml）

返回码:
  0  全部通过
  1  存在格式错误（错误信息打印到 stdout）
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed. Run: pip install pyyaml")
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[2]
PAPERS_DIR = ROOT / "papers"

FRONT_MATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")


def extract_front_matter(text: str) -> tuple[dict | None, str]:
    m = FRONT_MATTER_RE.match(text)
    if not m:
        return None, text
    try:
        data = yaml.safe_load(m.group(1))
    except yaml.YAMLError:
        return None, text
    if not isinstance(data, dict):
        return None, text
    return data, text[m.end():]


def check_front_matter(fm: dict, rel: str) -> list[str]:
    paper = fm.get("paper")
    if paper is None:
        return [f"{rel}: front matter `paper` missing"]
    if isinstance(paper, str):
        if not paper.strip():
            return [f"{rel}: front matter `paper` is empty"]
    elif isinstance(paper, list):
        if not paper:
            return [f"{rel}: front matter `paper` list is empty"]
        for item in paper:
            if not isinstance(item, str) or not item.strip():
                return [f"{rel}: front matter `paper` list contains empty or non-string entry"]
    else:
        return [f"{rel}: front matter `paper` must be a string or list of strings"]
    return []


def check_internal_links(body: str, file_path: Path, rel: str) -> list[str]:
    errors: list[str] = []
    file_dir = file_path.parent

    for lineno, line in enumerate(body.splitlines(), start=1):
        for _text, url in LINK_RE.findall(line):
            url = url.strip()

            # 跳过外部链接和纯锚点
            if url.startswith("http://") or url.startswith("https://") or url.startswith("#"):
                continue

            path_part = url.split("#")[0]
            target = (file_dir / path_part).resolve()
            if not target.exists():
                errors.append(f"{rel}:{lineno}: internal link target not found: `{url}`")

    return errors


def check_file(path: Path) -> list[str]:
    errors: list[str] = []
    rel = path.relative_to(ROOT).as_posix()
    text = path.read_text(encoding="utf-8")

    fm, body = extract_front_matter(text)

    if fm is None:
        errors.append(f"{rel}: front matter (---) missing or not valid YAML")
    else:
        errors.extend(check_front_matter(fm, rel))

    errors.extend(check_internal_links(body, path, rel))
    return errors


def main() -> int:
    if not PAPERS_DIR.exists():
        print("papers/ directory does not exist")
        return 1

    paper_files = sorted(p for p in PAPERS_DIR.rglob("*.md") if p.name != "README.md")

    if not paper_files:
        print("No paper files found under papers/")
        return 0

    all_errors: list[str] = []
    for path in paper_files:
        all_errors.extend(check_file(path))

    if all_errors:
        print("\n".join(all_errors))
        return 1

    print(f"All checks passed ({len(paper_files)} paper files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
