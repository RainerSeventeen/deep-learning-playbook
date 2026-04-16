---
name: interview-answer-writer
description: Write or update section-based interview answer notes for the deep-learning-playbook repository. Use this skill when the user gives one chapter or subsection of questions from `interview/questions.md` and wants matching answer content added under `interview/answer/`, following the concise Markdown structure and tone of `interview/answer/python/basic.md`.
---

# Interview Answer Writer

## Overview

Turn one chapter or subsection of interview questions into a finished answer note inside the repository.
Mirror the repository's existing study-note style: concise, structured, easy to scan, and aligned with the question order in `interview/questions.md`.

## Workflow

### 1. Read the source section and the format anchor

Always open:

- `interview/questions.md`
- `interview/answer/python/basic.md`

Also open the target answer file if it already exists.

Use `interview/answer/python/basic.md` as the formatting anchor unless a closer same-topic answer file already exists and is clearly the better match.

### 2. Determine the output location

Prefer updating or creating the most specific file under `interview/answer/`.

Follow these rules:

- Reuse an existing folder when the subject already has one.
- Reuse an existing file when the subsection already maps to it.
- If a new file is needed, choose a short lowercase filename that reflects the subsection meaning.
- Do not reorganize the note hierarchy unless the user explicitly asks.
- Do not modify `interview/questions.md` unless asked.

If the user gives an explicit target file, use it.
If the mapping from subsection to file is ambiguous and cannot be inferred from nearby files, ask one concise question before writing.

### 3. Preserve the answer-note structure

Match the visible structure used in `interview/answer/python/basic.md`:

- Use one `#` title for the subsection.
- Use one `##` heading per question.
- Keep the question order identical to the source section.
- Separate question blocks with `---`.
- Write in concise Chinese unless the repository context clearly requires another language.

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

## Formatting Checklist

Before finishing, verify all of the following:

- Every source question appears exactly once.
- No source question is skipped.
- No extra question headings were invented unless the user asked for expansion.
- Heading levels are consistent with the reference file.
- Separators and spacing match the existing answer-note style.
- The output file lives in the intended `interview/answer/` path.
- The final Markdown is readable without needing further cleanup.

## Default Operating Mode

When this skill triggers, default to doing the repository edit directly instead of only drafting prose in chat.

Use this decision order:

1. Locate the source subsection in `interview/questions.md`.
2. Infer or confirm the target file in `interview/answer/`.
3. Read the reference answer style.
4. Write or update the Markdown file.
5. Run a quick self-check for question coverage and formatting consistency.

## Output Pattern

Use this pattern as a guide, not as literal text:

```md
# 小节标题

## 问题 1

直接回答。

- 补充点 1
- 补充点 2

---

## 问题 2

直接回答。

```

Adjust the density per question.
Some questions only need 3 to 5 lines.
Some questions justify a table, formula, or short code block.
