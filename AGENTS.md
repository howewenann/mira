# MIRA Agent Guide

## Working Style

- Prefer small, readable changes over clever abstractions.
- Follow existing local patterns before introducing new helpers or structure.
- Keep modules direct and easy to trace; MIRA is intentionally educational.
- Prefer DeepAgents and LangGraph native behavior for tool calls, HITL resume,
  backend routing, and stream handling.
- Avoid MIRA-managed workarounds unless a real library/runtime edge case is
  confirmed. Keep any fallback narrow, documented by tests, and out of the
  normal path.

## Environment

- Use the shared Conda environment for Python checks:
  `conda run -n ai_agents python ...`
- Do not edit `.env` unless explicitly asked.
- Do not rely on installed entrypoints being fresh when testing code changes;
  prefer `conda run -n ai_agents python -m cli.main ...` for current checkout
  behavior.

## Testing

- Run focused unit tests for the areas you touch.
- Use `git diff --check` before finishing.
- For user-visible agent/runtime behavior, smoke test with real MIRA, for
  example:
  `conda run -n ai_agents python -m cli.main -p "..."`
- Use a disposable `--workspace` for smoke tests that may write `.mira/`,
  session files, or project artifacts.
- For TUI changes, add or update `tests.test_textual_app`.
- For one-shot terminal output, check `ui/renderer.py` behavior.
- For streamed turn/HITL behavior, check `runtime/runner.py` and add focused
  runner tests.

## Documentation

- Keep `README.md` focused on user-facing usage, setup, commands, and brief
  features.
- Keep `ARCHITECTURE_DECISIONS.md` focused on design rationale, high-level
  behavior, overwrite/precedence rules, and code pointers.
- When changing user-visible behavior, CLI flags, settings, project resources,
  sessions, planning mode, HITL, context handling, or UI behavior, update the
  relevant sections of both files in the same change.
- When asked why MIRA behaves a certain way, consult
  `ARCHITECTURE_DECISIONS.md` first, then verify against the code.

## Repo Safety

- Respect dirty worktrees. Do not revert or overwrite unrelated user changes.
- Leave untracked notebooks or local scratch files alone unless asked.
- Do not edit `.mira/_sessions` except when investigating a reported session
  issue.
- Do not remove generated or local metadata directories unless the task is
  specifically about cleanup and the target path has been verified.

## Architecture Map

- `cli/` starts MIRA and selects TUI or one-shot mode.
- `config/` loads `.env`, LLM settings, metadata, and `.mira/settings.yml`.
- `agent/factory.py` builds action and planning agents.
- `agent/resources/` loads backends, memories, skills, subagents, and tools.
- `runtime/runner.py` streams one turn and handles HITL approval loops.
- `runtime/*_events.py` handles stream event projections.
- `ui/app.py` and `ui/widgets/` implement the Textual TUI.
- `ui/renderer.py` implements plain `mira -p` terminal output.
- `session/` stores durable session JSON, replay context, and checkpoints.

## Settings And Execute

- Workspace settings live in `.mira/settings.yml`; use `/settings` in the TUI
  for user-facing changes.
- `execute` is special: enabling it switches the project backend to
  `LocalShellBackend`; disabling it uses `FilesystemBackend`.
- Keep `execute.always_allow` conservative by default. Approval-mode behavior
  and always-allow behavior should stay transcript-compatible.
- When fixing execute or HITL issues, compare real behavior in both modes and
  prefer restoring DeepAgents normal flow over reimplementing tool execution.
