---
name: interview-answer-writer
description: Write or update section-based interview answer notes for the deep-learning-playbook repository. Use this skill when the user gives one chapter or subsection of questions from `interview/questions.md` and wants matching answer content added under `interview/answer/`, following the repository's concise Markdown style and the CI-required `## 1. 题目原文` answer-heading format.
---

# Interview Answer Writer

## Overview

Turn one chapter or subsection of interview questions into a finished answer note inside the repository.
Mirror the repository's existing study-note style: concise, structured, easy to scan, and aligned with the question order in `interview/questions.md`.
When formatting guidance from existing files conflicts with CI, treat `scripts/ci/check_interview_mapping.py` as the source of truth.

## Workflow

### 1. Read the source section and the format anchor

Always open:

- `interview/questions.md`
- `interview/mapping.yaml`
- `interview/answer/python/basic.md`

Also open the target answer file if it already exists.
Always open `scripts/ci/check_interview_mapping.py` before editing so the current CI rules are explicit.

Use `interview/answer/python/basic.md` as the formatting anchor unless a closer same-topic answer file already exists and is clearly the better match.
Use it for tone and density, not for legacy heading syntax if that file has not yet been migrated.

### 2. Determine the output location

Prefer updating or creating the most specific file under `interview/answer/`.

Follow these rules:

- Reuse an existing folder when the subject already has one.
- Reuse an existing file when the subsection already maps to it.
- Treat `interview/mapping.yaml` as the source of truth for `## -> answer_dir` and `### -> answer_file`.
- Before creating a new answer file, check whether the `h2/h3` subsection already has a mapping entry.
- If the mapping entry is missing, add the corresponding `h2` directory mapping and `h3` file mapping before creating the answer file.
- If a new file is needed, choose a short lowercase filename that reflects the subsection meaning.
- Do not reorganize the note hierarchy unless the user explicitly asks.
- Do not modify `interview/questions.md` unless asked.

If the user gives an explicit target file, use it.
If the mapping from subsection to file is ambiguous and cannot be inferred from nearby files, ask one concise question before writing.

### 3. Preserve the answer-note structure

Match the visible structure used in `interview/answer/python/basic.md`:

- Use one `#` title for the subsection.
- Use one `## N. 题目原文` heading per question.
- Keep the question order identical to the source section.
- Keep the heading number `N` strictly sequential starting from `1`.
- Keep each heading text aligned to the source question text in the same order.
- Separate question blocks with `---`.
- Write in concise Chinese unless the repository context clearly requires another language.

Interpret the answer heading rule strictly:

- If the source question is `1. python里的函数是不是对象` then the answer heading should be `## 1. python里的函数是不是对象`.
- Do not rewrite the heading into a summary label.
- Do not omit the numeric prefix.
- Do not reorder or merge questions unless the user explicitly asks for a structural rewrite.

Do not turn the document into a long essay.
Optimize for interview recall and fast review.

### 4. Write each answer in the house style

For each question:

- Start with a direct answer or definition.
- Add only the detail needed to make the answer interview-usable.
- Use flat bullet lists when they improve scanning.
- Use tables when comparing concepts is clearer than prose.
- Use small code snippets only when code is the most compact explanation.
- Keep examples minimal and educational.

Target answer qualities:

- accurate
- compact
- technically grounded
- easy to recite in an interview
- easy to expand verbally if the interviewer follows up

### 5. Keep the content level pragmatic

Prefer content that helps in an interview conversation:

- what it is
- why it matters
- common trade-offs
- typical usage or failure modes
- one short example when useful

Avoid:

- filler introductions
- textbook-length derivations unless the question explicitly asks for formulas
- over-formatting beyond the established style
- unsupported claims or vague buzzwords

When formulas are central, include the formula and a short explanation of each term.
When implementation is central, include the smallest useful code sample.

### 6. Run CI immediately after writing

After writing or updating any interview answer file, run:

- `python3 scripts/ci/check_interview_mapping.py`

Treat this as a required execution step, not an optional final suggestion.

Use this rule:

- If you changed an answer file, run the CI check.
- If you changed `interview/mapping.yaml`, run the CI check.
- If CI fails, distinguish between failures caused by your edit and pre-existing unrelated failures.
- Do not claim the task is complete until you have executed the CI check and reviewed the result.

## Formatting Checklist

Before finishing, verify all of the following:

- Every source question appears exactly once.
- No source question is skipped.
- No extra question headings were invented unless the user asked for expansion.
- Every answer heading uses `## N. 题目原文`.
- Answer heading numbers are strictly sequential and match the source order.
- Answer heading text matches the corresponding source question text.
- Heading levels are consistent with the reference file.
- Separators and spacing match the existing answer-note style.
- The output file lives in the intended `interview/answer/` path.
- The `h2/h3` subsection exists in `interview/mapping.yaml` and points to the file you updated or created.
- `python3 scripts/ci/check_interview_mapping.py` passes for the touched subsection, or any remaining failure is clearly identified as pre-existing unrelated debt.
- The final Markdown is readable without needing further cleanup.

## Default Operating Mode

When this skill triggers, default to doing the repository edit directly instead of only drafting prose in chat.

Use this decision order:

1. Locate the source subsection in `interview/questions.md`.
2. Check `interview/mapping.yaml` for the subsection's `h2/h3` mapping.
3. Infer or confirm the target file in `interview/answer/`; if missing, add the mapping entry first.
4. Read the reference answer style.
5. Create or update the Markdown file.
6. Immediately run `python3 scripts/ci/check_interview_mapping.py`.
7. Review whether failures come from your change or from untouched sections.

## Output Pattern

Use this pattern as a guide, not as literal text:

```md
# 小节标题

## 1. 问题 1 原文

直接回答。

- 补充点 1
- 补充点 2

---

## 2. 问题 2 原文

直接回答。

```

Adjust the density per question.
Some questions only need 3 to 5 lines.
Some questions justify a table, formula, or short code block.
