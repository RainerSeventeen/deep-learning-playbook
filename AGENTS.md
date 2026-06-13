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

## Remote execution workflow
- Use the remote workflow only when the user explicitly asks to run work on the remote server or when local execution is clearly unsuitable for a requested code experiment.
- SSH configuration is read from `.env`; set `SSH_SERVER_NAME` there before remote work. `REMOTE_PROJECT_PATH` defaults to `/gpfs/yangsh/Code/deep-learning-playbook`.
- Standard flow:
  1. Edit files locally.
  2. Sync with `bash scripts/remote/rsync.sh`.
  3. Run remote commands directly:
     ```bash
     source .env
     ssh "$SSH_SERVER_NAME" 'cd /gpfs/yangsh/Code/deep-learning-playbook && <command>'
     ```
- If `REMOTE_PROJECT_PATH` is set in `.env`, use that path instead of the default in direct SSH commands. Avoid operating outside the remote project directory unless the user explicitly asks.
- Do not launch long training jobs, broad dataset generation, or destructive remote commands unless explicitly instructed. Prefer smoke tests or narrow checks by default.
- Keep local secrets, virtual environments, references, datasets, logs, runs, and model checkpoints out of sync; the rsync script excludes them intentionally.

## Boundary defaults
- Treat generated artifacts, temporary files, and local editor metadata as non-source material unless the user explicitly asks to keep them.
- Avoid large reorganizations of the note hierarchy unless the user asks for a structural refactor.
