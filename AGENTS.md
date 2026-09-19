# Deep Learning Playbook Agent Init

## Repository scope
- This repository is a structured study playbook for deep learning and AI interview preparation.
- The main content areas are concept notes, paper reading notes, interview question banks, and minimal code exercises.
- Unless the user explicitly asks for tooling or repository maintenance work, prioritize content editing and organization over infrastructure changes.

## Primary working areas
- `foundations/` stores concept-focused notes on core deep learning topics.
- `papers/` stores model and paper reading notes, grouped by topic.
- `interview/` stores question prompts and answer materials for interview prep.
- `coding/` stores minimal runnable implementations and notebooks used for explanation and practice.
- `references/` stores local reference materials such as books, cloned external repos, or course assets; only `references/README.md` is intended to be tracked.

## Editing guidance
- Preserve the repository's role as a study notebook rather than turning it into a large software project.
- For note-writing tasks, prefer concise and structured Markdown that is easy to scan and cross-reference.
- When adding new knowledge content, place it in the most specific existing section instead of creating broad catch-all files.
- When updating interview materials, keep question prompts and answer content clearly separated if the repository structure already does so.

## Code and verification
- Keep code examples minimal, readable, and educational.
- Prefer small local checks or targeted script runs over broad test scaffolding unless the user explicitly asks for more.
- Do not assume content under `references/` is part of the repository source of truth unless the task explicitly targets local reference materials.

## Remote training workflow
- Use the remote workflow only when the user explicitly asks to run work on the remote server or local execution is unsuitable for the requested experiment.
- `.codex/remote.env` is the local source for `REMOTE_SSH_ALIAS`, `REMOTE_PROJECT_PATH`, and any activation command. Do not put host credentials, passwords, or keys in tracked files.
- Before the first sync or after changing configuration, run `bash scripts/remote/check.sh` to inspect the SSH target and CUDA environment.
- Sync with `bash scripts/remote/rsync.sh`. It is one-way, does not use `--delete`, and excludes local configuration, environments, reference materials, datasets, logs, runs, and checkpoints.
- Run commands with `source .codex/remote.env && ssh "$REMOTE_SSH_ALIAS" "cd \"$REMOTE_PROJECT_PATH\" && <command>"`. Do not work outside the configured remote directory unless explicitly requested.
- Prefer small remote checks. Do not launch long training, large data generation, or destructive remote operations unless explicitly instructed.

## Boundary defaults
- Treat generated artifacts, temporary files, and local editor metadata as non-source material unless the user explicitly asks to keep them.
- Avoid large reorganizations of the note hierarchy unless the user asks for a structural refactor.
