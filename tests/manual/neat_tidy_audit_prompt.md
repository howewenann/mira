# Neat And Tidy Architecture Audit

You are reviewing the local MIRA repository for architectural neatness and
maintainability. Do not edit files. Do not refactor yet. Inspect the repo and
produce a grouped audit that helps the user choose one focused cleanup target.

MIRA is intentionally educational. Prefer small, readable modules with clear
ownership. Look especially for:

- helper-function stacking across unrelated modules
- catch-all classes or functions that do too many jobs
- UI layers accumulating formatting, logging, persistence, or runtime-control
  responsibilities
- duplicated parsing/formatting logic
- workarounds that should be replaced by library-native behavior
- modules that are hard to trace because responsibilities are mixed
- refactors that would touch too many unrelated behavior surfaces at once

Group findings by module or functional area, not as one giant flat list.
Suggested groups include:

- CLI and startup
- runtime runner and stream events
- TUI app shell and widgets
- terminal renderer
- session persistence and replay context
- configuration and settings
- project resources and tools
- diagnostics, trace, and error reports
- tests and manual verification

For each group:

1. Describe the current responsibilities.
2. Name the neatness risks, if any.
3. Identify specific files and symbols to inspect.
4. Explain why the issue matters for maintainability.
5. Estimate refactor difficulty: Small, Medium, or Large.
6. Estimate testing scope: focused unit, TUI test, runner test, one-shot smoke,
   manual verification, or broad regression.
7. Recommend the smallest safe cleanup target.
8. List likely tests or commands to run if that group is refactored.

Important constraints:

- Do not propose one mega-refactor.
- Do not recommend touching every layer at once.
- Prefer cleanup groups that can be implemented and tested independently.
- Keep durable behavior, session formats, CLI flags, and user-visible semantics
  stable unless the user explicitly chooses a behavior change.
- If a module is already neat enough, say so briefly.

After the grouped audit, provide a short prioritization table with:

- group
- why now
- difficulty
- testing scale
- recommended first step

Then stop and ask the user which group they want to target first. Do not start
implementing any cleanup until the user chooses a group.
