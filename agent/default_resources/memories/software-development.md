# Software Development

Use this workflow for software changes across Plan, Goal, and Act work. Adapt its
depth to the risk and size of the task, but do not skip understanding,
verification, or review merely because a change appears small.

## Establish the task

Before making substantial changes, establish four things:

- **Goal:** State the behavior or outcome that must change, from the user's or
  system's perspective.
- **Context:** Gather the relevant code, tests, documentation, errors, examples,
  and current behavior.
- **Constraints:** Identify architecture, compatibility, safety, permission,
  performance, and explicit user decisions that limit the solution.
- **Done when:** Define observable conditions that demonstrate completion,
  including required behavior and verification.

Resolve minor uncertainty through inspection and reasonable, reversible
assumptions. If a missing product decision would materially change behavior,
data, public interfaces, permissions, or scope, do not silently invent it. Make
the ambiguity and its consequences clear and obtain a decision when necessary.

## Inspect before editing

Understand the existing system before selecting a solution. Inspect the relevant
implementation paths and existing tests. Find nearby patterns, naming
conventions, public interfaces, callers, and data flows. Read architecture or
decision documents when the change touches established design. Consult
repository-specific memory and project configuration for the actual build,
test, lint, formatting, and type-check commands; do not guess when they can be
discovered.

Trace behavior far enough to identify the correct ownership boundary. Prefer
existing abstractions and conventions over a bespoke mechanism. Check whether
apparently local changes affect persistence, serialization, UI projections,
error reporting, permissions, or compatibility. For complex, ambiguous, or
cross-cutting work, form a coherent plan before implementation and keep it
aligned with what inspection reveals.

## Derive tests from requirements

Before or alongside implementation, translate each requirement into observable
behavior. Consider expected use, boundary conditions, invalid input, failure
handling, state transitions, persistence and reload behavior, backward
compatibility, adjacent regressions, and security or permission boundaries when
they are relevant.

Test at the boundary where the contract is visible. A useful test should fail if
the requested behavior is absent or broken. Avoid tests that merely prove a
function was called, a private implementation detail exists, a mock returned its
configured value, or code compiles without exercising the behavior. Use mocks
or fakes at genuine external boundaries, but do not mock away the behavior under
test.

Where practical, add or update focused tests first and confirm that they expose
the missing behavior. Do not force test-first work when it is meaningless,
excessively costly, or unsuitable for the change. Never weaken, delete, skip, or
rewrite a valid test merely to make an implementation pass.

## Implement coherently

Make the smallest coherent change, which is not always the fewest changed lines.
Preserve architectural intent, public behavior, compatibility, error handling,
and permission boundaries. Keep related behavior under clear ownership and
follow established local style.

Avoid unrelated cleanup unless it is required for the requested result. Do not
add abstractions, dependencies, configuration switches, fallback paths, or
compatibility shims without a demonstrated need. Keep the code readable to a
future maintainer who cannot see the conversation. Comments should explain
non-obvious intent or constraints, not restate the code.

## Verify progressively

Use the repository's documented commands. During development, run the narrowest
relevant tests, then tests for adjacent affected behavior. Run broader relevant
suites when practical, followed by applicable lint, formatting, type, build, or
static checks.

For UI, terminal, integration, concurrency, persistence, or environment-specific
changes, include realistic verification at the boundary where the problem
occurs. A unit test alone may not prove wiring, rendering, reload behavior, or a
real input path. Do not claim that a check passed unless it was actually run to
completion. If a check cannot run, report why and distinguish that limitation
from a failure.

## Review before completion

Review the final diff against the original goal, constraints, and completion
conditions. Look for missing requirements, regressions, unnecessary complexity,
incorrect assumptions, weak or misleading tests, neglected error paths,
security or permission changes, stale documentation, and temporary debugging
artifacts. Confirm that only intended files changed and that tests demonstrate
the requested contract.

Report concisely what changed, what was tested, what could not be verified, and
any remaining risk or unresolved issue.

## Examples

**Feature addition — Clear History:** First trace the visible history, persisted
storage, confirmation flow, refresh path, and error display. Derive tests showing
that cancellation preserves history; confirmation deletes persisted history;
the visible state refreshes after success; and deletion failure is reported
without falsely showing success. Then implement through the existing action and
persistence boundaries. A test that only proves the button exists is
insufficient.

**Bug fix — cross-environment shortcut:** Reproduce the failing keyboard input
through the environment-specific input path. Add a regression test at that input
boundary, correct the shortcut handling, and verify both the corrected shortcut
and nearby keyboard behavior. Testing only an internal key-normalization helper
would miss failures in event translation or dispatch.

**Refactor — unchanged public behavior:** Establish behavioral coverage for the
current public outputs, errors, and compatibility expectations. Refactor the
internals without rewriting tests to mirror the new structure, then rerun the
behavioral coverage and adjacent checks. Compilation is useful, but it does not
prove that callers observe unchanged behavior.
