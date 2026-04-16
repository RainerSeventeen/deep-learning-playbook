---
name: interview-question-importer
description: Import interview questions into `interview/questions.md` for the deep-learning-playbook repository. Use this skill when the user provides a screenshot, OCR result, note, or plain text containing interview questions and wants them classified into the existing sections of `interview/questions.md`, merged with deduplication, optionally folded into nearby existing questions as suffix expansions, and reported as added, skipped, expanded, or unmatched.
---

# Interview Question Importer

## Overview

Turn raw interview questions from an image or text block into clean updates to `interview/questions.md`.

The repository is a study notebook, so preserve the existing outline and keep edits pragmatic:

- classify into the most suitable existing section
- skip exact duplicates
- merge close variants into an existing question when that reads better than adding a new line
- report anything that does not fit an existing section

## Workflow

### 1. Extract all questions from the input

Always start by reading the full user input carefully.

If the input is an image:

- read every visible question from the image directly
- fix only obvious OCR-style issues such as spacing, casing, or punctuation
- preserve technical terms such as `ReLU`, `sigmoid`, `Cross Entropy`, `DIN`, `pooling`

If the input is text:

- split it into individual questions
- keep the user's wording unless a very small cleanup makes the meaning clearer

Extraction rules:

- do not silently drop a line that looks like a question
- if one line contains one core question plus a follow-up angle, keep it as one question unless the prompts are clearly independent
- if an image is blurry or a phrase is ambiguous, make the smallest reasonable correction and mention the uncertainty in the final report

### 2. Read the repository taxonomy first

Open `interview/questions.md` before editing.

Find the most specific existing placement for each extracted question:

- prefer an existing subsection over a broad top-level section
- match by topic, not by superficial keywords alone
- reuse nearby clusters when several adjacent questions already cover the same concept family

Do not create a new section by default.
If no suitable existing section is found, do not force the question into a weak match. Report it to the user as unmatched.

### 3. Choose exactly one action per question

For each extracted question, choose one of these outcomes:

#### `skip`

Use `skip` when an existing question is already effectively the same.

Treat as duplicate when:

- the wording is identical
- the wording differs slightly but the interview intent is clearly the same

Do not add a second copy just because the wording is shorter or longer.

#### `expand`

Use `expand` when the new question is very close to an existing one but adds a useful new angle.

Typical cases:

- existing question asks for the core concept, new question adds a comparison angle
- existing question asks for the method, new question adds a scenario or constraint
- existing question is broad, new question contributes a natural follow-up scope

When expanding:

- edit the existing question line directly
- keep the sentence natural and compact
- add the new angle as a suffix or rewrite the sentence into a slightly broader question
- do not turn one line into a paragraph

Example pattern:

- existing: `如何解决过拟合问题？`
- new angle: `怎么在模型层面解决`
- updated: `如何解决过拟合问题？有哪些数据、训练和模型层面的常见方法？`

#### `add`

Use `add` when the question is genuinely new for that subsection.

Insertion rules:

- default to appending at the end of the target subsection
- if there is a clearly related local cluster, insert beside that cluster instead
- preserve the repository's numbered-list style
- renumber only within the affected subsection when needed

### 4. Keep question wording clean and interview-oriented

The goal is a question bank, not polished prose.

Prefer wording that is:

- concise
- technically precise
- easy to scan
- faithful to the original interview intent

Editing guidance:

- keep Chinese as the default language unless the surrounding section clearly uses English terms
- retain standard English technical identifiers where natural
- normalize obvious mistakes like `sigmod` to `sigmoid` only when the intent is unambiguous
- avoid adding answer content into `interview/questions.md`

### 5. Respect repository boundaries

- edit `interview/questions.md` only
- do not touch `interview/answer/` unless the user explicitly asks
- do not reorganize headings unless the user requests structural cleanup
- do not modify unrelated files

## Matching Heuristics

Use semantic fit over literal overlap.

Examples:

- loss functions, entropy, KL, softmax gradients -> `DL 基础 / Loss 相关`
- activations, ReLU vs sigmoid, dropout, batch norm, overfitting mechanisms -> `DL 基础 / DL 网络机制`
- LLM sampling, quantization, fine-tuning, deployment -> matching `LLM 类通用` subsections
- RAG retrieval, rerank, chunking -> `Agent / RAG 应用 / RAG`

If a question spans multiple sections, prefer the section where a reviewer would most naturally look for it later.

## Final Report

After editing, always report the outcome in four buckets:

1. `实际新增`
2. `追加到已有问题`
3. `跳过`
4. `未找到合适分区`

For each item, include:

- the normalized question text
- the target section or matched existing question when applicable

If any extraction was uncertain, add one short note describing the assumption.

## Default Operating Mode

When this skill triggers, default to executing the edit directly instead of only describing a plan.

Use this order:

1. extract questions from the image or text
2. read `interview/questions.md`
3. classify each question
4. edit `interview/questions.md`
5. report `新增 / 追加 / 跳过 / 未匹配`
