# MIRA Agent Guide

## Working Style

- Prefer small, readable changes over clever abstractions.
- Follow existing local patterns before introducing new helpers or structure.
- Keep modules direct and easy to trace; MIRA is intentionally educational.
- Keep changes neat and tidy: avoid stacking small helper functions across
  unrelated modules when a focused module or simple local class would make the
  behavior easier to understand.
- Do not grow catch-all classes or functions that do many jobs. Prefer narrow
  modules with clear ownership, and keep UI layers from accumulating formatting,
  logging, persistence, and runtime-control responsibilities.
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
- For one-shot terminal output, check `ui/terminal/renderer.py` behavior.
- For streamed turn/HITL behavior, check `core/execution/runner.py` and add focused
  runner tests.
- For behavior that needs user-driven manual verification, add or update
  prompts and expected results in `tests/manual/prompts.md` so the user can
  run the manual checks.

## TUI Interaction Requirements

- When a request says a disclosure or collapsible should behave like a tool
  bubble, treat `ui/textual/widgets/tool_bubble.py` and the
  `.tool-args-collapsible > CollapsibleTitle` rules in `ui/textual/styles/mira.tcss` as
  the complete interaction reference, including its native glyph behavior,
  automatic title height, padding, focus treatment, and hover stability. Keep
  the target surface's requested colors unless the user also asks to copy the
  tool-bubble palette. Do not remove hover feedback to hide a glyph-rendering
  problem; align the title sizing and state rules, then test the collapsed and
  expanded glyph geometry before and during hover.
- Non-input menus, panels, reports, and diagnostic modals open without initial
  keyboard focus. In Textual, disable automatic focus and clear any framework-
  assigned focus after refresh when necessary; tests must assert that the
  screen has no focused widget and that a close button is not focused. Give a
  widget initial focus only when the surface exists to collect immediate user
  input, such as a prompt/HITL editor, or when the behavior is explicitly
  requested.

## Commits

- Use a concise, imperative, sentence-case subject that describes the complete
  outcome. Conventional Commit prefixes such as `feat:`, `fix:`, and scoped
  variants are allowed when they make the subject clearer.
- Add a body for every non-trivial change. Use complete prose paragraphs to
  explain the user-visible behavior, important implementation or architecture
  decisions, documentation changes, and compatibility considerations. Derive
  this detail from the actual staged diff; a good subject does not replace the
  body.
- End the body with the verification performed, including focused or full test
  results and any known failures or limitations. Keep the detail proportional
  to the change, but do not reduce a substantive commit to a one-line subject.
- Review the staged diff and relevant recent commits before writing the message
  so the commit accurately describes everything being committed.

## Releases

- Bump the single source of truth in `config/version.py` and commit all release
  preparation changes before drafting release notes.
- Draft release notes from the exact committed comparison range: use the base
  commit or tag supplied by the user and compare it with the resulting release
  HEAD.
- Review recent releases at `https://github.com/howewenann/mira/releases` and
  follow their established template, tone, and purpose.
- Write for MIRA users. Lead with outcomes and select only changes that affect
  installation, configuration, common workflows, reliability, safety, or the
  visible experience. Do not dump internal architecture, refactors, tests, or a
  commit-by-commit changelog unless they materially affect users.
- Use the established release shape: `# MIRA vX.Y.Z - <title>`, a short summary,
  `This release includes:`, concise user-facing bullets, migration or
  compatibility guidance only when needed, local and Conda install commands, a
  short `Happy coding.` close, and the GitHub compare link.
- Keep claims grounded in the committed diff. Do not invent features,
  migration steps, or compatibility requirements.

## Documentation

- Treat `README.md` as a concise user quick-start, not a changelog, behavioral
  specification, architecture record, or test plan. Keep it focused on
  installation, essential configuration, common commands, safety, and links to
  deeper documentation.
- Do not add implementation details, minor UI behavior, internal lifecycle or
  precedence rules, exhaustive edge cases, or manual verification matrices to
  `README.md`. Put design rationale in `ARCHITECTURE_DECISIONS.md` and manual
  procedures in `tests/manual/`.
- Prefer replacing or tightening existing README text over appending another
  paragraph. Keep the file under roughly 200 lines unless essential user setup
  genuinely requires more.
- Keep `ARCHITECTURE_DECISIONS.md` focused on design rationale, high-level
  behavior, overwrite/precedence rules, and code pointers.
- Update `README.md` only when a change affects installation, required
  configuration, public CLI options, or a common user workflow. Minor visual
  fixes and internal behavior changes normally require only architecture notes,
  tests, or manual prompts.
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
- `core/execution/runner.py` streams one turn and handles HITL approval loops.
- `core/execution/streams/` handles stream event projections.
- `core/interface/` implements the internal Core-to-consumer boundary.
- `mira/` exposes the supported public Python API used by consumers.
- `ui/textual/app.py` and `ui/textual/widgets/` implement the Textual TUI.
- `ui/terminal/renderer.py` implements plain `mira -p` terminal output.
- `session/` stores durable session JSON, replay context, and checkpoints.

## Settings And Execute

- Workspace settings live in `.mira/settings.yml`; use `/settings` in the TUI
  for user-facing changes.
- `execute` is special: enabling it switches the project backend to
  `LocalShellBackend`; disabling it uses `FilesystemBackend`.
- Keep `execute.always_allow` conservative by default. Approval-mode behavior
  and always-allow behavior should stay transcript-compatible.
- Treat `write_file` as create-or-replace and `edit_file` as targeted editing.
  Keep recursive `delete` action-only and governed by the configured HITL
  policy.
- Planning todos are opt-in. When enabled, keep their middleware and tool
  metadata aligned across action, planning, and compiled dynamic subagents.
- Plan mode is one persistent read-only conversation. Formal Plans always use
  `prepare_plan` -> `SuccessCriteriaService` -> forced `finalize_plan`, regardless
  of rubric settings. Keep exactly one durable session `current_plan`; do not
  route Plan construction through `prepare_goal` or activate a Goal.
- `show_plan` and `/plan-show` must reuse the exact Plan bubble renderer.
  Implement and `/plan-resume` execute the retained Plan in Act; Revise replaces
  it; Close only hides controls; `/plan-clear` alone removes it.
- Formal Goals always use `prepare_goal` -> `SuccessCriteriaService` -> forced
  `finalize_goal`, regardless of rubric settings. A Goal contains only its exact
  Objective and Success Criteria; never generate or persist a hidden Plan.
- Keep exactly one current formal artifact: `current_plan` or `current_goal`,
  never both. Confirm replacement of incomplete formal work and do not replace
  it until the new artifact is successfully presented.
- Persist only the current Plan and Goal schemas. Do not migrate retired
  `active_goal`, proposal-event, generic stage, pre-staging payload, or
  nested-message checkpoint shapes.
- `show_goal` and `/goal-show` must reuse the exact Goal bubble renderer.
  Implement and `/goal-resume` execute the retained Goal only for that explicit
  attempt; Revise replaces it; Close only hides controls; `/goal-clear` removes it.
- When fixing execute or HITL issues, compare real behavior in both modes and
  prefer restoring DeepAgents normal flow over reimplementing tool execution.
