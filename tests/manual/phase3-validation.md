# Grouped transcript correction and validation

Correction base: `f55cd08`; Phase 2C comparison: `d3484c5`. The user approved
the native keyline integration probe before production integration. Core,
the public API, persistence schemas, dependencies and Inspector code are unchanged.

## Delivered presentation

- One Assistant container per outer public turn identity, with user messages
  outside. Model phases do not split interactions.
- A clickable, shaded (`#17313a`), two-line summary replaces the separate history
  disclosure. Its clock is flush right; colored counts include completed logical
  activities and exclude active work and the final response. Working, waiting,
  cancelling and terminal states remain distinct.
- Native ListView/ListItem/Horizontal/Label rows use parent `keyline` borders:
  N activities occupy **2N+1 terminal rows**. Content occupies one row, with
  fixed glyph and duration columns and a flexible ellipsized argument preview.
  Existing bubble colors are retained; thinking and message labels have no
  generated descriptions. Tool previews use actual arguments without LLM calls.
- History has natural height and uses the main conversation scrollbar. Rows
  mount on first expansion, with no pagination, truncation or separate viewport.
  Native keyboard selection, mouse highlighting and one selected detail below
  the entire list replace per-activity disclosure hierarchies.
- Selecting another row replaces the detail; selecting it again, Close, or
  collapsing history hides it. Existing detailed widgets retain full captured
  content, copy/display controls and Inspector navigation. Tool Arguments and
  Output initially collapse; their details grow naturally with the conversation.
- Live activities retain their existing bubbles and stream identity. Parallel
  activity, unresolved Plan/Goal controls, HITL and failures remain visible.
  Completed workflows enter history/counts; failed/interrupted Rubric activity
  remains visible. Rubric verification keeps the response under review visible
  without counting it as an intermediate message. The final response stays
  outside history, with exactly one blank row before it in Text and Markdown
  modes, for all history/detail states.
- Detached historical records use the existing grouping path. Unknown legacy
  spans remain flat and ordered. Historical turns without authoritative timings
  show Completed; cold recovery explicitly labels its new timed segment.

## Missing read information: confirmed cause and limits

Native execution on unchanged `f55cd08` captured a **2,163-character read result**
including `READ_END`, while its ToolBubble exposed a truncated **119-character**
presentation without that tail (using the fixture's 80-character preview limit).
The intermediate assistant message and tool-call identity were present.
This reproduces missing read-result content, rather than
loss of a public call event.

The trace followed native execution into public frontend events, TextualFrontend,
ChatLog's call-ID mapping, AssistantTurn completion and historical replay. The
confirmed loss was in `ChatLog._update_tool_block`: it always applied the terminal
preview limit before updating the detailed ToolBubble. Opening history could not
recover the omitted text. Grouped tool details now receive the complete existing
captured result. Inspector's flat presentation retains its existing behavior.

The native graph regression performs seven concurrent `read_file` calls, each
with a real filesystem result, and verifies every full result plus the preceding
assistant message both live and after restoration. Typed-event checks additionally
verify prompt live mounting, tool drafts, parallel completion, errors and recovery.
Complete disappearance of a read-call event was **not reproduced**; this finding
does not establish the cause of every possible instance of that reported symptom.

## Simplification and accounting

Removed the obsolete ActivityViewport and its tail/anchor management, native
disclosure wrappers around every activity, and individually rendered row borders.
Active widgets mount directly. Completed full widgets stay detached until selected;
the native list mounts once on demand. Lifecycle updates touch the affected row,
not a rebuilt history. No second execution history or authoritative state was added.

The approved [standalone probe](../../probe/README.md) established native keyline
geometry and focus behavior on Textual **8.2.8**. One Highlighted handler uses native
ancestor scrolling for keyboard navigation; lifecycle insertion preserves native
cursor identity without moving the conversation. The production integration also
stops list clicks reaching ChatLog's generic focus handler. Native batching and
coalesced lifecycle moves preserve streamed widget identity and existing controls.

[Deep Agents Code's LazyToolGroupSummary](https://github.com/langchain-ai/deepagents/blob/main/libs/code/deepagents_code/tui/widgets/messages.py)
was reviewed for deferred detail mounting and native batching techniques only.
No rendering framework, virtualization or dependency was imported.

Workflow footer controls now compose explicitly inside their retained native
container, so moving a completed tree into selected details reattaches its
Final state button rather than leaving that control detached.

Compared with `f55cd08`, production Python changes are **283 added / 232 deleted,
net +51 lines**. Stylesheet changes are **64 added / 28 deleted, net +36**.
Across all Phase 3 work relative to Phase 2C: **809 Python lines added / 75 deleted,
net +734**, plus **100 stylesheet lines**. Tests, probes, documentation, benchmarks
and exports are excluded; no files were relocated. This reduces widget/lifecycle
overhead, but does not claim a net reduction in total production Python lines.

## Performance

[Benchmark](benchmark_grouped_transcript.py), same machine and environment,
110 × 40 cells: 250 saved records over 50 turns, then 200 public content fragments
with an event-loop yield per fragment. The three revisions ran sequentially;
values are medians of three samples. [Raw samples](artifacts/phase3/benchmark.json).

| Measurement | Phase 2C `d3484c5` | Phase 3 `f55cd08` | Correction |
| --- | ---: | ---: | ---: |
| Mounted transcript widgets after restore | 900 | 1,900 | 700 |
| Restore mounts, counted widgets | 900 | 1,900 | 700 |
| Restore removals | 0 | 0 | 0 |
| Restore time | 2.1884 s | 5.6670 s | 2.2371 s |
| 200-fragment streaming time | 0.7517 s | 0.8143 s | 0.3752 s |
| Mounts for that new interaction | 8 | 19 | 13 |
| Streaming removals | 0 | 0 | 0 |

Collapsed restoration mounts 63% fewer widgets than original Phase 3. Restoration
time improves against Phase 3 and is close to Phase 2C. Timing samples vary; these
are rendering microbenchmarks, not model latency guarantees. Opening history adds
its native rows; selecting details mounts the retained full widget. This workload
does not measure those interactions, process memory or multi-thousand-turn sessions.

## Regression and live validation

- **242 public API, identity, inspection, SQLite and execution/runner tests passed**
  after integration, including the subprocess that blocks UI/Textual imports.
- The broad relevant UI run executed **447 tests in 828 seconds, with seven
  failures**. Four expected completed workflow trees to remain mounted; these
  tests now select the native history detail before checking the same tree,
  final-state and Inspector controls, including the public-handle guard. This
  uncovered the detached footer-control defect described above; the explicit
  native composition fix passed all four focused workflow/control checks.
- The two scroll failures were
  `test_chat_log_pauses_active_updates_until_user_returns_to_bottom` and
  `test_initial_artifact_implementation_preserves_manual_scroll`. Both passed in
  isolation on unchanged `f55cd08`. The general scroll test captured a fractional
  position while its native animation was still running; it now waits for that
  animation to settle before testing incoming updates. The artifact path passed
  the focused current rerun. Git-dialog focus
  (`test_git_configure_yes_initializes_persists_and_clears_issue`)
  also failed on unchanged `f55cd08`, matching the known baseline. No unrelated
  production focus behavior was changed.
- The final **20 focused acceptance/regression checks passed in 82.7 seconds**,
  including all 14 grouped cases, both scroll paths, the public-handle guard,
  and the three workflow/Inspector integration paths. These include the final
  footer composition and Rubric-response visibility fixes. The history test
  helper waits for actual attachment and completed lifecycle moves, rather than
  Textual's historical `is_mounted` flag. The broad run was not repeated after
  these final adjustments and is not represented as a clean pass.
- A configured Gemma/LM Studio run before keyline integration completed read_file
  and ls, retained the intermediate assistant message, and exposed its full
  **6,017-character** read result after restoration. The post-integration attempt
  timed out after 180 seconds before any assistant/tool event was saved; only the
  user and interruption records were present. Live provider success after keyline
  integration is therefore unverified. Native graph regression coverage uses the
  actual MIRA graph and filesystem tools with a scripted model.
- Inspector interactions, Plan/Goal controls, Rubric, workflows, HITL, cancellation,
  recovery and session restoration/switching are covered by the relevant UI suite
  and focused grouped checks. No full repository rerun is claimed.
- Compilation, **37 relative documentation targets**, and whitespace validation
  passed. No Mermaid diagram was changed.

## Actual rendered evidence

The frames below are actual Textual exports with color enabled, rasterized with
headless Chrome and visually reviewed against the approved design. They are not
browser mockups. [Capture script](capture_grouped_transcript.py); SVGs sit beside PNGs.

1. [Completed, collapsed](artifacts/phase3/completed.png)
2. [Expanded joined rows](artifacts/phase3/history.png)
3. [Highlighted and selected row](artifacts/phase3/highlighted.png)
4. [Selected 400-line thinking block](artifacts/phase3/thinking.png)
5. [Long tool arguments and output](artifacts/phase3/details.png)
6. [Running interaction](artifacts/phase3/running.png)
7. [Narrow layout](artifacts/phase3/narrow.png)
8. [Final response spacing](artifacts/phase3/final-spacing.png)

Mounted layout checks cover 5/20/50 rows at narrow/wide widths, stable native
glyphs, flush-right durations, keyboard/mouse selection, parallel insertion,
400-line thinking and 350-line output, natural scrolling, detail replacement,
Close, and exact final spacing in both display modes. Desktop terminal interaction
was not performed. The remaining manual scenarios are in
[prompts.md](prompts.md#grouped-assistant-transcript). Conservative legacy grouping,
unavailable historical durations, narrow-header count ellipsis, and the live-model
timeout above remain explicit limits.
