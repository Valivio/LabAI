# LabAI Codex Instructions

## Project purpose

LabAI is a local-first AI platform for Apple Silicon.

Its goals are to:

- run and compare local AI models,
- keep private data and knowledge sources under user control,
- produce source-backed outputs,
- support hybrid workflows with external large language models,
- prepare controlled and minimized data packages for cloud execution,
- keep model backends replaceable,
- benchmark models on the user's own hardware and tasks.

## Repository boundaries

This repository contains only:

- source code,
- tests,
- concise project documentation,
- synthetic examples,
- non-private configuration templates.

Private resources are stored outside the repository:

- private data: `~/LabAI-data`
- models: `~/AIModels`

Do not copy, move, index, inspect, or modify those directories unless the user explicitly requests it.

Never add to the repository:

- books or real document content,
- private notes,
- model files,
- embeddings,
- vector indexes,
- databases containing private data,
- logs containing prompts or source text,
- secrets or real `.env` files.

## Development environment

- macOS on Apple Silicon
- Python 3.13
- dependency and environment management with `uv`
- local environment in `.venv`
- MLX and `mlx-lm` as the initial local inference stack

Use `uv` for dependency management and command execution.

Do not install Python packages globally.

## Engineering rules

- Inspect the current repository state before editing.
- Keep changes minimal and directly related to the task.
- Do not create speculative abstractions.
- Do not add modules only for possible future use.
- Preserve existing behavior unless the task explicitly changes it.
- Keep model-specific logic behind internal interfaces.
- Do not couple the core architecture directly to one model or provider.
- Prefer simple, typed, testable Python.
- Add or update tests for changed behavior.
- Use synthetic fixtures only.
- Do not require model downloads during normal unit tests.
- Do not commit, push, create branches, or open pull requests unless explicitly requested.
- Do not modify files outside this repository unless explicitly requested.

## Reporting

After implementation, report:

- files changed,
- behavior added or changed,
- tests executed,
- test results,
- unresolved issues or assumptions.

Do not produce lengthy progress narratives.
