#!/usr/bin/env python3
"""
校验 interview/questions.md、interview/mapping.yaml 和 interview/answer/ 下答案文件的映射关系。

规则：
  - questions.md 中：
      - `##` 表示 answer 目录层级
      - `###` 表示 answer 文件层级
      - 所有题目必须位于 `###` 标题下
  - mapping.yaml 中：
      - 显式声明 `## -> answer_dir`
      - 显式声明 `### -> answer_file`
      - 不重复存储题目文本
  - answer 文件中：
      - 文件首个 `#` 可作为文件标题
      - 每个具体回答必须使用 `##`
      - 第 N 个题目对应第 N 个 `##` 回答块
      - 回答块正文不能为空

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
from dataclasses import dataclass
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed. Run: pip install pyyaml")
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[2]
INTERVIEW_DIR = ROOT / "interview"
QUESTIONS_FILE = INTERVIEW_DIR / "questions.md"
MAPPING_FILE = INTERVIEW_DIR / "mapping.yaml"
ANSWER_ROOT = INTERVIEW_DIR / "answer"

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
ORDERED_LIST_RE = re.compile(r"^\s*(\d+)\.\s*(.*)$")
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
    start_line: int
    body_lines: tuple[str, ...]


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


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
        sections[key].append(QuestionItem(key=key, ordinal=ordinal, text=question_text, line_no=lineno))

    return sections, h2_to_h3s, errors


def parse_mapping_yaml() -> tuple[dict[SectionKey, SectionMapping], dict[str, str], list[str]]:
    errors: list[str] = []
    section_map: dict[SectionKey, SectionMapping] = {}
    dir_map: dict[str, str] = {}

    if not MAPPING_FILE.exists():
        return {}, {}, [f"{rel(MAPPING_FILE)}: file not found"]

    try:
        raw = yaml.safe_load(MAPPING_FILE.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return {}, {}, [f"{rel(MAPPING_FILE)}: invalid YAML: {exc}"]

    if not isinstance(raw, dict):
        return {}, {}, [f"{rel(MAPPING_FILE)}: top-level YAML value must be a mapping"]

    version = raw.get("version")
    if version != 1:
        errors.append(f"{rel(MAPPING_FILE)}: `version` must be 1")

    sections = raw.get("sections")
    if not isinstance(sections, list):
        errors.append(f"{rel(MAPPING_FILE)}: `sections` must be a list")
        return {}, {}, errors

    seen_h2: set[str] = set()
    seen_target_files: dict[Path, SectionKey] = {}

    for index, section in enumerate(sections, start=1):
        prefix = f"{rel(MAPPING_FILE)}: sections[{index}]"
        if not isinstance(section, dict):
            errors.append(f"{prefix} must be a mapping")
            continue

        h2 = section.get("h2")
        answer_dir = section.get("answer_dir")
        files = section.get("files")

        if not isinstance(h2, str) or not h2.strip():
            errors.append(f"{prefix}.h2 must be a non-empty string")
            continue
        h2 = h2.strip()

        if h2 in seen_h2:
            errors.append(f"{prefix}: duplicate h2 mapping `{h2}`")
            continue
        seen_h2.add(h2)

        if not isinstance(answer_dir, str) or not answer_dir.strip():
            errors.append(f"{prefix}.answer_dir must be a non-empty string")
            continue
        answer_dir = answer_dir.strip()
        answer_dir_path = resolve_from_root(answer_dir)
        if not is_within(answer_dir_path, ANSWER_ROOT.resolve(strict=False)):
            errors.append(
                f"{prefix}.answer_dir points outside `{rel(ANSWER_ROOT)}`: `{answer_dir}`"
            )
            continue

        dir_map[h2] = answer_dir

        if not isinstance(files, list):
            errors.append(f"{prefix}.files must be a list")
            continue

        seen_h3: set[str] = set()
        for file_index, file_entry in enumerate(files, start=1):
            file_prefix = f"{prefix}.files[{file_index}]"
            if not isinstance(file_entry, dict):
                errors.append(f"{file_prefix} must be a mapping")
                continue

            h3 = file_entry.get("h3")
            answer_file = file_entry.get("answer_file")

            if not isinstance(h3, str) or not h3.strip():
                errors.append(f"{file_prefix}.h3 must be a non-empty string")
                continue
            h3 = h3.strip()

            if h3 in seen_h3:
                errors.append(f"{file_prefix}: duplicate h3 mapping `{h2} / {h3}`")
                continue
            seen_h3.add(h3)

            if not isinstance(answer_file, str) or not answer_file.strip():
                errors.append(f"{file_prefix}.answer_file must be a non-empty string")
                continue
            answer_file = answer_file.strip()

            answer_file_path = (answer_dir_path / answer_file).resolve(strict=False)
            if not is_within(answer_file_path, answer_dir_path):
                errors.append(
                    f"{file_prefix}.answer_file escapes its answer_dir: `{answer_file}`"
                )
                continue
            if not is_within(answer_file_path, ANSWER_ROOT.resolve(strict=False)):
                errors.append(
                    f"{file_prefix}.answer_file points outside `{rel(ANSWER_ROOT)}`: `{answer_file}`"
                )
                continue

            key = SectionKey(h2, h3)
            previous_key = seen_target_files.get(answer_file_path)
            if previous_key is not None:
                errors.append(
                    f"{file_prefix}: duplicate answer target `{rel(answer_file_path)}` already used by `{previous_key.display()}`"
                )
                continue
            seen_target_files[answer_file_path] = key

            section_map[key] = SectionMapping(
                key=key,
                answer_file_rel=rel(answer_file_path),
                answer_file_path=answer_file_path,
            )

    return section_map, dir_map, errors


def parse_answer_file(path: Path) -> tuple[list[AnswerBlock], list[str]]:
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

    in_fence = False
    current_heading: str | None = None
    current_start_line = 0
    current_body: list[str] = []

    def flush_current() -> None:
        nonlocal current_heading, current_start_line, current_body
        if current_heading is None:
            return
        block = AnswerBlock(
            heading=current_heading,
            start_line=current_start_line,
            body_lines=tuple(current_body),
        )
        blocks.append(block)
        current_heading = None
        current_start_line = 0
        current_body = []

    for lineno, line in enumerate(text.splitlines(), start=1):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            if current_heading is not None:
                current_body.append(line)
            continue

        if not in_fence:
            heading_match = HEADING_RE.match(line)
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                if level == 2:
                    flush_current()
                    if not title:
                        errors.append(f"{rel_path}:{lineno}: empty h2 answer heading")
                        continue
                    current_heading = title
                    current_start_line = lineno
                    current_body = []
                    continue

        if current_heading is not None:
            current_body.append(line)

    flush_current()

    if not blocks:
        errors.append(f"{rel_path}: no `##` answer blocks found")
        return [], errors

    for block in blocks:
        if not has_meaningful_content(block.body_lines):
            errors.append(
                f"{rel_path}:{block.start_line}: answer block `{block.heading}` has empty body"
            )

    return blocks, errors


def validate_structure_mapping(
    questions: dict[SectionKey, list[QuestionItem]],
    question_tree: dict[str, list[str]],
    mappings: dict[SectionKey, SectionMapping],
    dir_map: dict[str, str],
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    question_h2s = set(question_tree)
    mapped_h2s = set(dir_map)

    for h2 in sorted(question_h2s - mapped_h2s):
        warnings.append(f"WARNING: {rel(MAPPING_FILE)}: missing h2 mapping for `{h2}`")
    for h2 in sorted(mapped_h2s - question_h2s):
        errors.append(f"{rel(MAPPING_FILE)}: stale h2 mapping for `{h2}`")

    for h2, answer_dir in sorted(dir_map.items()):
        answer_dir_path = resolve_from_root(answer_dir)
        if not answer_dir_path.exists():
            errors.append(
                f"{rel(MAPPING_FILE)}: answer_dir for `{h2}` does not exist: `{answer_dir}`"
            )
        elif not answer_dir_path.is_dir():
            errors.append(
                f"{rel(MAPPING_FILE)}: answer_dir for `{h2}` is not a directory: `{answer_dir}`"
            )

    question_keys = set(questions)
    mapped_keys = set(mappings)

    for key in sorted(question_keys - mapped_keys, key=lambda item: (item.h2, item.h3)):
        warnings.append(f"WARNING: {rel(MAPPING_FILE)}: missing file mapping for `{key.display()}`")
    for key in sorted(mapped_keys - question_keys, key=lambda item: (item.h2, item.h3)):
        errors.append(f"{rel(MAPPING_FILE)}: stale file mapping for `{key.display()}`")

    return errors, warnings


def validate_question_answer_alignment(
    questions: dict[SectionKey, list[QuestionItem]],
    mappings: dict[SectionKey, SectionMapping],
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    ok_logs: list[str] = []
    parsed_answers: dict[Path, tuple[list[AnswerBlock], list[str]]] = {}

    for key in sorted(questions, key=lambda item: (item.h2, item.h3)):
        mapping = mappings.get(key)
        if mapping is None:
            continue

        if mapping.answer_file_path not in parsed_answers:
            parsed_answers[mapping.answer_file_path] = parse_answer_file(mapping.answer_file_path)
        answer_blocks, answer_errors = parsed_answers[mapping.answer_file_path]
        if answer_errors:
            errors.extend(answer_errors)
            continue

        question_items = questions[key]
        if len(question_items) != len(answer_blocks):
            errors.append(
                f"{mapping.answer_file_rel}: `{key.display()}` has {len(question_items)} questions "
                f"but {len(answer_blocks)} `##` answer blocks"
            )
            continue

        for question, block in zip(question_items, answer_blocks, strict=True):
            ok_logs.append(
                f"OK: {key.display()} / 第{question.ordinal}题 `{question.text}` -> "
                f"{mapping.answer_file_rel} / 第{question.ordinal}个回答 `{block.heading}`"
            )

    return errors, ok_logs


def main() -> int:
    all_errors: list[str] = []
    all_warnings: list[str] = []

    questions, question_tree, question_errors = parse_questions_md()
    all_errors.extend(question_errors)

    mappings, dir_map, mapping_errors = parse_mapping_yaml()
    all_errors.extend(mapping_errors)

    structure_errors, structure_warnings = validate_structure_mapping(
        questions, question_tree, mappings, dir_map
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
