# Phase 2C implementation and validation

Comparison: Phase 2B `f614fe9c` to this change. Phase 3 is not implemented.

1. **Coverage.** The completed [capability inventory](../../docs/frontend-architecture.md#migration-inventory)
   maps the existing frontend's lifecycle, history, recovery, execution,
   artifacts, Rubric, Inspector, workflows, runtime, model, MCP, skills,
   workspace, diagnostics and telemetry functions to supported public operations.
   No remaining frontend capability requires a private Core entry point.
2. **Public contracts.** Application operations now cover history metadata,
   configuration, runtime reconstruction, catalogs, prompt preparation and
   workspace maintenance. Session operations cover clearing, compaction,
   context reports, uploads, workflow execution and artifact conveniences.
   MCP has a supported client returning detached metadata. Lightweight snapshots
   expose pending recovery and execution state without native descriptors.
   `mira.settings` supplies preference editors and explicit offline configuration
   operations. MCP approval now exposes server identity and transport rather
   than the native server state; consumers must use that supported contract.
3. **Bypasses removed.** Textual no longer executes native graphs, saves session
   records, restores checkpoints, launches workflow factories, rebuilds agents,
   cleans inspection stores, or reads private recovery descriptors. Backend
   file discovery, imports and clearing go through public operations. Core no
   longer discovers renderer naming, animation or execution capabilities.
4. **Ownership.** Core owns execution, persistence, resources, recovery,
   inspection capture and runtime changes. Frontends own widgets, dialogs,
   formatting, focus, scrolling, selection and local presentation state.
   Local DirectoryTree/file previews remain ordinary frontend presentation.
5. **Simplification.** One reconstruction path replaces separate startup,
   model, MCP, settings and reload implementations. Textual's operation worker
   consolidates repeated UI lifecycle handling while Core owns native cleanup.
   Workflow history observes typed events once, replacing callback forwarding.
   Message/tool provenance is applied in the existing presentation projection;
   inspection/configuration events are consumed directly. Lightweight snapshots,
   cached workflow event positions and direct tool-call lookup avoid repeated
   transcript copies/scans. Redundant frontend restoration and persistence are
   deleted, rather than moved into another frontend wrapper.
6. **Production accounting.** For tracked production Python, excluding tests
   and examples, `git diff --numstat --no-renames f614fe9c` records **2,203 added,
   2,620 deleted, net 417 fewer lines**. Six complete module relocations account
   for 450 original lines: branding, trace presentation, upload handling, two
   workflow diagnostics and tool-group presentation. Branding/group relocation
   includes small edits. Partial relocations include prompt history, backend
   file scanning and Rubric/correction/context formatting; these are ownership
   changes, not claimed deletions. The physical totals include all relocations;
   the net reduction is for the whole production change, not just Textual.
7. **Retained structure.** The shared UI renderer adapter still translates typed
   events into established terminal/Textual rendering callbacks. Detached UI
   dictionaries remain presentation caches, refreshed from public snapshots.
   Small elapsed-time formatting remains independently readable in Core timing
   and UI formatting. Native workflow frozen values and existing persistence
   schemas remain in place. These preserve compatibility without introducing
   another framework or storage layer. Python public contracts are not an
   access-control sandbox.
8. **Documentation.** The API guide, architecture diagrams/inventory, design
   decisions, full example, example README, tracing notes and manual prompts
   describe the final boundary. All **six Mermaid diagrams parsed successfully
   with the actual Mermaid 11.12.0 parser**, using a DOM supplied by linkedom
   0.18.12. Five message-text semicolons were replaced. The parser and DOM were
   temporary validation tools, not dependencies. GitHub's pinned renderer and
   browser SVG appearance were not independently tested.
9. **Verification.** See the results below. Static checks are supplemented by
   a public-handle-only Textual proxy and a subprocess blocking every `ui` and
   `textual` import while exercising native public application/session operations.
   Existing deterministic tests cover native approvals/recovery, formal work,
   inspection, workflow persistence, runtime changes and MCP behavior.
10. **Isolation assessment.** Complete Textual API dogfooding is implemented:
    supported operations and detached data cover the inventoried capabilities;
    typed events and requests are the only Core-to-frontend protocol. Another
    frontend can implement these capabilities from the public reference without
    reproducing Textual's native integration. This assessment rests on the
    capability/path audit plus behavioral isolation checks, not import checks
    or line counts alone.
11. **Phase 3 readiness.** The public identities, durable display records and
    inspection relationships remain available for the approved transcript layout.
    The integration boundary is ready for that work. Known baseline regression
    failures remain separate; this refactor does not claim to resolve them.

## Verification results

- Baseline full suite at `f614fe9c`: 1,695 tests, six failures, ten errors and
  one skip. Baseline failures concern ACP HTTP permission expectations,
  autocomplete filtering, Issues timing, unset-model messaging, Inspector CSS
  and Git-dialog focus. Baseline errors concern the MCP reload panel,
  middleware/dynamic-subagent construction, six obsolete model-button selectors
  and a policy-refresh fixture with no live session.
- Final full suite (`conda run --no-capture-output -n mira python -m unittest
  discover -s tests -t .`): **1,699 tests in 691 seconds, seven failures,
  ten errors and one skip**. All six baseline failures and all ten baseline
  errors recur. The additional failure was
  `test_workflow_tree_retains_non_json_final_state_without_session_event`:
  its 300 ms copy-feedback assertion observed `Copy` instead of `Copied`.
  Four focused reruns passed without code changes. The copy-feedback widget
  implementation is unchanged; the full-suite timing failure remains recorded
  rather than treated as a clean full-suite pass.
- Focused final checks: 41 public operation, SQLite session, workspace and chat
  clearing tests passed. An additional six targeted Settings/public-handle/
  revision checks passed. Earlier migration slices exercised MCP/resources,
  workflow history and inspection persistence; regressions they exposed were
  corrected and rerun.
- Disposable live configured-model checks: CLI returned `PHASE2C_OK`, the
  minimal frontend returned `PONG`, and the full frontend returned
  `PHASE2C_FULL_OK`. These verify real startup, streaming and shutdown; they do
  not claim live coverage of every approval/provider/artifact combination.
- Compilation, 224 relative document targets, 16 Python snippets and
  working/staged whitespace checks passed. Manual visual interaction was not performed;
  additional user-driven checks are in [prompts.md](prompts.md#complete-public-api-ownership).
