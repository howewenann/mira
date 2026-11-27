# Phase 3 summary focus and transient waiting follow-up

Validated on 2026-11-28 against `f98e83b`, using the shared Conda `mira`
environment and Textual 8.2.8. This is a targeted follow-up to the
[Phase 3 correction](phase3-validation.md).

## Corrections

- Removed only the summary focus outline. Textual paints that outline over
  both content rows even though geometry still reports a two-row widget.
  Background-based hover/focus feedback, height and interaction are retained.
- Restored the original `d3484c5` working bubble through `show_waiting()`,
  `hide_waiting()`, `tick_waiting()` and `_render_waiting()`. Spinner frames,
  yellow styling, label, elapsed rendering, grace period and animation timer
  remain unchanged. A narrow `transient` option on `_add_block()` mounts waiting
  directly in ChatLog, bypassing the grouped activity registry. Persistent
  status activities such as compaction still use their existing grouping path.
- Consolidated all five standalone experiments under [probe/](../../probe/README.md).
  The native keyline reference moved from `tests/probes/`; the integration probe
  now imports that sibling rather than depending on the tests package. Check
  modes accept `--out` so repeated exports need not modify approved artifacts.

## Verification

**36 selected regression tests passed in 90.244 seconds.** The run included:

- Three new `tests.test_textual_app` regressions: rendered summary visibility
  with hover, clicks, native Tab focus, Enter and Space at 60/110 columns;
  grouped waiting animation across initial/tool/reasoning/model phases; and
  cleanup during cancellation and chat switching.
- The strengthened turn-failure test waits for a visible working bubble before
  releasing the failing operation and checks that the indicator is removed.
- Existing waiting/working regressions for stream silence, slow text chunks,
  tool drafts, eval, Rubric, compaction, explicit labels and elapsed clocks.
- All 14 `tests.test_grouped_transcript` cases, including native keylines,
  natural scrolling, full details, exact final spacing, controls and recovery.
- Native seven-read graph/replay coverage, the public-handle UI guard, Workflow
  node/agent Inspector navigation, and both static public-boundary checks.

The waiting tests check the original yellow renderable and changing spinner,
direct ChatLog ownership, no activity registration/count/history entry, no
saved-record change from waiting, and no reset of the outer clock. The summary
test reads Textual's styled render strips, so an overlay that hides text fails
even if the widget's height remains correct.

All five standalone probe check modes passed when executed from `probe/` with
separate temporary output directories. The focus A/B probe also asserts the
actual rendered text: the historical outline hides both rows, while removing
it retains them. No new screenshots, caches or generated observations were added
to Git; existing approved frames remain in place.

A real configured-model smoke run using `python -m cli.main --workspace ... -p`
completed successfully in a disposable workspace with final text `PHASE3_OK`.
This checks native one-shot execution, not a live-provider TUI interaction.
Mouse/keyboard and waiting lifecycle checks used the mounted Textual pilot;
desktop manual interaction and the full repository suite were not run for this
targeted follow-up. The [manual prompts](prompts.md#grouped-assistant-transcript)
include the corresponding interactive scenarios.

Changed Python sources compile, documentation targets resolve, and
`git diff --check` passes. Public API contracts, Inspector implementation,
session schemas and version are unchanged.
