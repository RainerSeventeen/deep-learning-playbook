#!/usr/bin/env python3
"""
校验 interview/questions.md、interview/mapping.yaml 和 interview/answer/ 下答案文件的映射关系。

文件结构（version 2 flat 格式）：
  - questions.md：
      - `##` 表示大章节（h2）
      - `###` 表示小节（h3），对应一个合并答案文件内的一个小节
      - 有序列表项是具体题目
  - mapping.yaml（version 2）：
      - 每个 h2 对应一个 answer_file（合并后的 flat md）
      - subsections 列出该文件内的 h3 小节名
  - answer 文件（flat md）：
      - `## 小节名` 作为小节分隔（对应 h3）
      - `### N. 题目` 作为答案块（level 3 heading）
      - 答案块正文不能为空

用法:
  python3 scripts/ci/check_interview_mapping.py

依赖:
  PyYAML（pip install pyyaml）

返回码:
  0  全部通过，或仅存在缺失映射 warning
  1  存在映射或内容错误
"""
from __future__ import annotations

import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed. Run: pip install pyyaml")
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[2]
INTERVIEW_DIR = ROOT / "interview"
QUESTIONS_FILE = INTERVIEW_DIR / "00_questions.md"
MAPPING_FILE = INTERVIEW_DIR / "mapping.yaml"
ANSWER_ROOT = INTERVIEW_DIR

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
ORDERED_LIST_RE = re.compile(r"^\s*(\d+)\.\s*(.*)$")
ORDERED_ANSWER_HEADING_RE = re.compile(r"^(\d+)\.\s+(.+?)\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
THEMATIC_BREAK_RE = re.compile(r"^([-*_])\1{2,}$")


@dataclass(frozen=True)
class SectionKey:
    h2: str
    h3: str

    def display(self) -> str:
        return f"{self.h2} / {self.h3}"


@dataclass(frozen=True)
class QuestionItem:
    key: SectionKey
    ordinal: int
    declared_ordinal: int
    text: str
    line_no: int


@dataclass(frozen=True)
class SectionMapping:
    key: SectionKey
    answer_file_rel: str
    answer_file_path: Path


@dataclass(frozen=True)
class AnswerBlock:
    heading: str
    ordinal: int | None
    question_text: str | None
    start_line: int
    body_lines: tuple[str, ...]


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_from_root(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve(strict=False)


def has_meaningful_content(lines: tuple[str, ...]) -> bool:
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if THEMATIC_BREAK_RE.fullmatch(stripped):
            continue
        return True
    return False


def normalize_question_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(
        ch
        for ch in normalized
        if not ch.isspace() and not unicodedata.category(ch).startswith(("P", "S"))
    )


# ---------------------------------------------------------------------------
# 解析 questions.md
# ---------------------------------------------------------------------------

def parse_questions_md() -> tuple[dict[SectionKey, list[QuestionItem]], dict[str, list[str]], list[str]]:
    errors: list[str] = []
    sections: dict[SectionKey, list[QuestionItem]] = {}
    h2_to_h3s: dict[str, list[str]] = {}

    if not QUESTIONS_FILE.exists():
        return {}, {}, [f"{rel(QUESTIONS_FILE)}: file not found"]

    text = QUESTIONS_FILE.read_text(encoding="utf-8")

    current_h2: str | None = None
    current_h3: str | None = None
    seen_h2: set[str] = set()
    seen_keys: set[SectionKey] = set()

    for lineno, line in enumerate(text.splitlines(), start=1):
        heading_match = HEADING_RE.match(line)
        if heading_match:
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()

            if level == 2:
                current_h2 = title
                current_h3 = None
                if current_h2 in seen_h2:
                    errors.append(
                        f"{rel(QUESTIONS_FILE)}:{lineno}: duplicate h2 heading `{current_h2}`"
                    )
                else:
                    seen_h2.add(current_h2)
                    h2_to_h3s[current_h2] = []
            elif level == 3:
                if current_h2 is None:
                    errors.append(
                        f"{rel(QUESTIONS_FILE)}:{lineno}: h3 heading `{title}` appears before any h2 heading"
                    )
                    continue
                current_h3 = title
                key = SectionKey(current_h2, current_h3)
                if key in seen_keys:
                    errors.append(
                        f"{rel(QUESTIONS_FILE)}:{lineno}: duplicate h3 heading under `{current_h2}`: `{current_h3}`"
                    )
                else:
                    seen_keys.add(key)
                    h2_to_h3s[current_h2].append(current_h3)
                    sections[key] = []
            continue

        list_match = ORDERED_LIST_RE.match(line)
        if not list_match:
            continue

        declared_ordinal = int(list_match.group(1))
        question_text = list_match.group(2).strip()
        if current_h2 is None:
            errors.append(
                f"{rel(QUESTIONS_FILE)}:{lineno}: question item appears before any h2 heading"
            )
            continue
        if current_h3 is None:
            errors.append(
                f"{rel(QUESTIONS_FILE)}:{lineno}: question item under h2 `{current_h2}` must be nested under an h3 heading"
            )
            continue
        if not question_text:
            errors.append(
                f"{rel(QUESTIONS_FILE)}:{lineno}: empty question under `{current_h2} / {current_h3}`"
            )
            continue

        key = SectionKey(current_h2, current_h3)
        ordinal = len(sections.setdefault(key, [])) + 1
        if declared_ordinal != ordinal:
            errors.append(
                f"{rel(QUESTIONS_FILE)}:{lineno}: question numbering under `{key.display()}` "
                f"must be sequential, expected `{ordinal}.` but got `{declared_ordinal}.`"
            )
        sections[key].append(
            QuestionItem(
                key=key,
                ordinal=ordinal,
                declared_ordinal=declared_ordinal,
                text=question_text,
                line_no=lineno,
            )
        )

    return sections, h2_to_h3s, errors


# ---------------------------------------------------------------------------
# 解析 mapping.yaml（version 2）
# ---------------------------------------------------------------------------

def parse_mapping_yaml() -> tuple[dict[SectionKey, SectionMapping], dict[str, Path | None], list[str]]:
    """
    返回：
      section_map: SectionKey -> SectionMapping（h3 小节 -> 答案文件路径）
      file_map:    h2 -> answer_file_path | None（大章节 -> 合并文件路径，None 表示尚无答案）
      errors
    """
    errors: list[str] = []
    section_map: dict[SectionKey, SectionMapping] = {}
    file_map: dict[str, Path | None] = {}

    if not MAPPING_FILE.exists():
        return {}, {}, [f"{rel(MAPPING_FILE)}: file not found"]

    try:
        raw = yaml.safe_load(MAPPING_FILE.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return {}, {}, [f"{rel(MAPPING_FILE)}: invalid YAML: {exc}"]

    if not isinstance(raw, dict):
        return {}, {}, [f"{rel(MAPPING_FILE)}: top-level YAML value must be a mapping"]

    version = raw.get("version")
    if version != 2:
        errors.append(f"{rel(MAPPING_FILE)}: `version` must be 2")

    sections = raw.get("sections")
    if not isinstance(sections, list):
        errors.append(f"{rel(MAPPING_FILE)}: `sections` must be a list")
        return {}, {}, errors

    seen_h2: set[str] = set()
    seen_target_files: dict[Path, str] = {}

    for index, section in enumerate(sections, start=1):
        prefix = f"{rel(MAPPING_FILE)}: sections[{index}]"
        if not isinstance(section, dict):
            errors.append(f"{prefix} must be a mapping")
            continue

        h2 = section.get("h2")
        answer_file_str = section.get("answer_file")
        subsections = section.get("subsections")

        if not isinstance(h2, str) or not h2.strip():
            errors.append(f"{prefix}.h2 must be a non-empty string")
            continue
        h2 = h2.strip()

        if h2 in seen_h2:
            errors.append(f"{prefix}: duplicate h2 mapping `{h2}`")
            continue
        seen_h2.add(h2)

        # answer_file 可以为 null（表示该章节暂无答案文件）
        if answer_file_str is None:
            file_map[h2] = None
            continue

        if not isinstance(answer_file_str, str) or not answer_file_str.strip():
            errors.append(f"{prefix}.answer_file must be a non-empty string or null")
            continue
        answer_file_str = answer_file_str.strip()
        answer_file_path = resolve_from_root(answer_file_str)

        try:
            answer_file_path.relative_to(ANSWER_ROOT.resolve(strict=False))
        except ValueError:
            errors.append(
                f"{prefix}.answer_file points outside `{rel(ANSWER_ROOT)}`: `{answer_file_str}`"
            )
            continue

        if answer_file_path in seen_target_files:
            errors.append(
                f"{prefix}: duplicate answer_file `{rel(answer_file_path)}` already used by `{seen_target_files[answer_file_path]}`"
            )
            continue
        seen_target_files[answer_file_path] = h2
        file_map[h2] = answer_file_path

        if not isinstance(subsections, list):
            errors.append(f"{prefix}.subsections must be a list")
            continue

        seen_h3: set[str] = set()
        for sub_index, sub in enumerate(subsections, start=1):
            sub_prefix = f"{prefix}.subsections[{sub_index}]"
            if not isinstance(sub, dict):
                errors.append(f"{sub_prefix} must be a mapping")
                continue

            h3 = sub.get("h3")
            if not isinstance(h3, str) or not h3.strip():
                errors.append(f"{sub_prefix}.h3 must be a non-empty string")
                continue
            h3 = h3.strip()

            if h3 in seen_h3:
                errors.append(f"{sub_prefix}: duplicate h3 mapping `{h2} / {h3}`")
                continue
            seen_h3.add(h3)

            key = SectionKey(h2, h3)
            section_map[key] = SectionMapping(
                key=key,
                answer_file_rel=rel(answer_file_path),
                answer_file_path=answer_file_path,
            )

    return section_map, file_map, errors


# ---------------------------------------------------------------------------
# 解析合并后的 flat answer 文件
# ---------------------------------------------------------------------------

def parse_answer_file_section(
    path: Path, h3_name: str
) -> tuple[list[AnswerBlock], list[str]]:
    """
    在合并后的 flat md 文件中，找到 `## h3_name` 小节，
    并解析其中的 `### N. 题目` 答案块。
    """
    errors: list[str] = []
    blocks: list[AnswerBlock] = []
    rel_path = rel(path)

    if not path.exists():
        return [], [f"{rel_path}: file not found"]
    if not path.is_file():
        return [], [f"{rel_path}: target is not a file"]

    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return [], [f"{rel_path}: file is empty"]

    lines = text.splitlines()

    # 1. 找到目标 ## 小节的行范围
    section_start: int | None = None
    section_end: int | None = None

    for i, line in enumerate(lines):
        m = HEADING_RE.match(line)
        if m and len(m.group(1)) == 2:
            title = m.group(2).strip()
            if title == h3_name:
                section_start = i + 1  # 跳过标题行本身
            elif section_start is not None:
                section_end = i
                break

    if section_start is None:
        return [], [
            f"{rel_path}: section `## {h3_name}` not found"
        ]

    section_lines = lines[section_start:section_end]  # None 表示到文件末尾

    # 2. 在小节内解析 ### N. 题目 块
    in_fence = False
    current_heading: str | None = None
    current_start_line = 0
    current_body: list[str] = []

    def flush_current() -> None:
        nonlocal current_heading, current_start_line, current_body
        if current_heading is None:
            return
        ordered_match = ORDERED_ANSWER_HEADING_RE.fullmatch(current_heading)
        block = AnswerBlock(
            heading=current_heading,
            ordinal=int(ordered_match.group(1)) if ordered_match else None,
            question_text=ordered_match.group(2).strip() if ordered_match else None,
            start_line=current_start_line,
            body_lines=tuple(current_body),
        )
        blocks.append(block)
        current_heading = None
        current_start_line = 0
        current_body = []

    for offset, line in enumerate(section_lines):
        abs_lineno = section_start + offset + 1  # 1-indexed

        if FENCE_RE.match(line):
            in_fence = not in_fence
            if current_heading is not None:
                current_body.append(line)
            continue

        if not in_fence:
            hm = HEADING_RE.match(line)
            if hm:
                level = len(hm.group(1))
                title = hm.group(2).strip()
                if level == 3:
                    flush_current()
                    if not title:
                        errors.append(f"{rel_path}:{abs_lineno}: empty h3 answer heading")
                        continue
                    if not ORDERED_ANSWER_HEADING_RE.fullmatch(title):
                        errors.append(
                            f"{rel_path}:{abs_lineno}: answer heading must use `### N. question` format, got `{title}`"
                        )
                    current_heading = title
                    current_start_line = abs_lineno
                    current_body = []
                    continue
                # level 1/2/4+ 在小节内直接忽略（blockquote、表格等不受影响）

        if current_heading is not None:
            current_body.append(line)

    flush_current()

    if not blocks:
        errors.append(
            f"{rel_path}: section `## {h3_name}` has no `### N. question` answer blocks"
        )
        return [], errors

    for block in blocks:
        if not has_meaningful_content(block.body_lines):
            errors.append(
                f"{rel_path}:{block.start_line}: answer block `{block.heading}` has empty body"
            )

    return blocks, errors


# ---------------------------------------------------------------------------
# 校验映射与结构
# ---------------------------------------------------------------------------

def validate_structure_mapping(
    questions: dict[SectionKey, list[QuestionItem]],
    question_tree: dict[str, list[str]],
    mappings: dict[SectionKey, SectionMapping],
    file_map: dict[str, Path | None],
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    question_h2s = set(question_tree)
    mapped_h2s = set(file_map)

    for h2 in sorted(question_h2s - mapped_h2s):
        warnings.append(f"WARNING: {rel(MAPPING_FILE)}: missing h2 mapping for `{h2}`")
    for h2 in sorted(mapped_h2s - question_h2s):
        errors.append(f"{rel(MAPPING_FILE)}: stale h2 mapping for `{h2}`")

    for h2, answer_file_path in sorted(file_map.items(), key=lambda x: x[0]):
        if answer_file_path is None:
            continue  # 显式声明无答案文件，跳过
        if not answer_file_path.exists():
            errors.append(
                f"{rel(MAPPING_FILE)}: answer_file for `{h2}` does not exist: `{rel(answer_file_path)}`"
            )
        elif not answer_file_path.is_file():
            errors.append(
                f"{rel(MAPPING_FILE)}: answer_file for `{h2}` is not a file: `{rel(answer_file_path)}`"
            )

    question_keys = set(questions)
    mapped_keys = set(mappings)

    for key in sorted(question_keys - mapped_keys, key=lambda k: (k.h2, k.h3)):
        warnings.append(f"WARNING: {rel(MAPPING_FILE)}: missing subsection mapping for `{key.display()}`")
    for key in sorted(mapped_keys - question_keys, key=lambda k: (k.h2, k.h3)):
        errors.append(f"{rel(MAPPING_FILE)}: stale subsection mapping for `{key.display()}`")

    return errors, warnings


def validate_question_answer_alignment(
    questions: dict[SectionKey, list[QuestionItem]],
    mappings: dict[SectionKey, SectionMapping],
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    ok_logs: list[str] = []

    # 缓存已解析的 (file, h3) -> blocks，避免重复读文件
    parsed_cache: dict[tuple[Path, str], tuple[list[AnswerBlock], list[str]]] = {}

    for key in sorted(questions, key=lambda k: (k.h2, k.h3)):
        mapping = mappings.get(key)
        if mapping is None:
            continue

        cache_key = (mapping.answer_file_path, key.h3)
        if cache_key not in parsed_cache:
            parsed_cache[cache_key] = parse_answer_file_section(
                mapping.answer_file_path, key.h3
            )
        answer_blocks, answer_errors = parsed_cache[cache_key]

        if answer_errors:
            errors.extend(answer_errors)
            continue

        question_items = questions[key]
        if len(question_items) != len(answer_blocks):
            errors.append(
                f"{mapping.answer_file_rel}: `{key.display()}` has {len(question_items)} questions "
                f"but {len(answer_blocks)} `### N. question` answer blocks"
            )
            continue

        for question, block in zip(question_items, answer_blocks, strict=True):
            if block.ordinal != question.ordinal:
                errors.append(
                    f"{mapping.answer_file_rel}:{block.start_line}: `{key.display()}` question #{question.ordinal} "
                    f"must map to `### {question.ordinal}. ...`, got `{block.heading}`"
                )
                continue

            normalized_question = normalize_question_text(question.text)
            normalized_answer = normalize_question_text(block.question_text or "")
            if normalized_question != normalized_answer:
                errors.append(
                    f"{mapping.answer_file_rel}:{block.start_line}: `{key.display()}` question #{question.ordinal} "
                    f"text mismatch, expected `{question.text}` but got `{block.question_text or block.heading}`"
                )
                continue

            ok_logs.append(
                f"OK: {key.display()} / 第{question.ordinal}题 `{question.text}` -> "
                f"{mapping.answer_file_rel} / 第{question.ordinal}个回答 `{block.heading}`"
            )

    return errors, ok_logs


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main() -> int:
    all_errors: list[str] = []
    all_warnings: list[str] = []

    questions, question_tree, question_errors = parse_questions_md()
    all_errors.extend(question_errors)

    mappings, file_map, mapping_errors = parse_mapping_yaml()
    all_errors.extend(mapping_errors)

    structure_errors, structure_warnings = validate_structure_mapping(
        questions, question_tree, mappings, file_map
    )
    all_errors.extend(structure_errors)
    all_warnings.extend(structure_warnings)

    alignment_errors, ok_logs = validate_question_answer_alignment(questions, mappings)
    all_errors.extend(alignment_errors)

    if all_warnings:
        print("\n".join(all_warnings))
    if all_errors:
        if all_warnings:
            print()
        print("\n".join(all_errors))
        return 1

    if ok_logs:
        if all_warnings:
            print()
        print("\n".join(ok_logs))
    print(f"All checks passed ({len(questions)} mapped sections)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
