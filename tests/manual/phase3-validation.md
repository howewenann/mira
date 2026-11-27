# Grouped transcript implementation and validation

Base: `d3484c5`. This change affects presentation; Core APIs and persistence remain unchanged.

## Implemented behavior

- One Assistant container per public outer `turn_id`, with user messages outside.
  Model-message boundaries and internal phases do not create additional turns.
- A working/waiting/cancelling header and completed duration; category-colored
  counts derive from completed logical widget identities. Active parallel work
  stays visible together and is excluded from those counts.
- Completed activity is collapsed by default. Its five-row native scroll
  viewport retains every row, supports independent expansion and follows the
  tail only while already following. Arrivals preserve expansion, focus and
  the reader's position, including earlier parallel activities settling late.
- The final response streams outside history. Intermediate messages enter
  history; a final answer remains prominent after Rubric postprocessing.
  A message-stream end alone does not finalize an outer interaction.
- Existing tools, reasoning, subagents, Rubric phases, formal artifacts and
  workflow trees are reused. Nested tool argument/output disclosures start
  collapsed. Failures, approvals, AskUser, MCP elicitation, artifact reviews,
  live artifact controls and recovery remain accessible.
- Inspector transcripts keep their existing flat layout. Workflow trees retain
  their node, agent and final-state navigation.

## Identity and restoration

[TextualFrontend](../../ui/textual/adapter.py) supplies public identity during the
existing callback projection. [ChatLog](../../ui/textual/widgets/chat_log.py)
routes existing widgets through [AssistantTurn](../../ui/textual/widgets/assistant_turn.py).
Widget references, category totals and expansion/scroll state are local view
state, not another execution history. No Core operations, public contracts,
persistence schemas, provider behavior or dependencies changed.

Tool drafts and durable results continue to reuse the established call/origin
mapping. Native message identities preserve streaming widgets across interleaved
updates. Recovery retains the outer container and reactivates existing activity.
Eval children restored from public inspection summaries use their durable parent
event and inspection identities, including separate eval scopes.

Saved display records use the same widget/grouping path. Unidentified legacy
records remain flat and ordered; identified spans interleaved with unassigned
legacy entries also stay flat rather than inventing boundaries. Sessions are
never rewritten for layout. Historical turns have no authoritative saved turn
duration, so they show Completed. A cold recovery labels its new timed segment
"since recovery"; warm recovery retains the known accumulated duration.

## Simplification and accounting

The change adds the requested layout rather than claiming a net code reduction.
Shared mount/removal routing replaces repeated direct placement and cleanup of
the existing widgets. Artifact widgets now own their mount-time focus, removing
the ChatLog's delayed focus forwarding. No second renderer, event hierarchy,
storage layer, lazy-loading framework or virtualization system was introduced.

Native remove/mount moves occur at activity lifecycle transitions, not per token.
Textual resets ContentSwitcher initialization and stops widget timers during
those moves; a small local switcher retains the display preference and the
assistant copy control retains its existing feedback deadline. Mounted tests
cover both behaviors. Native `batch_update`, deferred callbacks and scrolling
are used. Upstream [Deep Agents Code's message widgets](https://github.com/langchain-ai/deepagents/blob/main/libs/code/deepagents_code/tui/widgets/messages.py)
were consulted for Textual batching techniques only.

Production Python accounting: **745 lines added, 62 deleted, net 683 added**.
The stylesheet adds **64 lines**. There are no relocations.
Test helpers, the benchmark, documentation and image exports are excluded from
production Python totals; stylesheet additions are reported separately.

## Performance comparison

[The reproducible benchmark](benchmark_grouped_transcript.py) uses the same
deterministic fixtures on an archived `d3484c5` checkout and the working checkout.
Window: 110 × 40 cells. Workload: 250 saved records across 50 interactions, then
200 public content fragments for one new response, yielding to the event loop
between fragments. Values below are medians of three local runs, including
mounting and a layout refresh; they are not portable latency guarantees.

| Measurement | Phase 2C | Grouped transcript |
| --- | ---: | ---: |
| Mounted transcript widgets after restoration | 900 | 1,900 |
| Restoration mount calls, counted widgets | 900 | 1,900 |
| Restoration removals | 0 | 0 |
| Restoration time | 1.7545 s | 4.1674 s |
| 200-fragment streaming workload | 0.2915 s | 0.5484 s |
| Mounts for that entire new interaction | 8 | 19 |
| Removals during that simple response | 0 | 0 |

The five-row viewport reduces visible clutter, not mounted widget count. The
additional native containers/disclosures increase restoration and streaming
overhead. Streaming still updates one response widget rather than mounting a
widget per fragment or replaying history. No performance improvement is claimed.
Process memory and large multi-thousand-turn sessions were not measured.

## Verification

- Before modification, six targeted Phase 2C smoke tests passed, covering native
  typed streaming, tool recovery, phase boundaries, workflow restoration and the
  restricted public-handle integration.
- The full repository suite ran **1,704 tests in 778 seconds: eight failures,
  ten errors and one skip**. Six failures and all ten errors match the recorded
  Phase 2C baseline in [its report](phase2c-validation.md). The additional failures
  were narrow AskUser button geometry and fixed-subagent-panel tail timing.
  Both passed isolated on the unchanged baseline and on this checkout; the
  full-suite timing failures remain recorded, not treated as a clean run.
- All **64** public API, identity, inspection, SQLite and isolation checks passed,
  including the subprocess with every UI/Textual import blocked. No API bypass
  was introduced.
- A later Textual/workspace/layout run executed **439 tests in 496 seconds**
  with three failures and one error. Git-dialog focus is the known baseline
  failure. The other paths were manual-scroll timing, natural Goal recall
  mounting, and escaped context-overflow information. The latter two now wait
  for actual mounted controls in their tests; interrupted turns also keep their
  information notice visible. The final **11 focused acceptance/regression
  checks passed**, including all eight new grouped-layout cases and those three
  paths. Nine display/copy/layout checks and 13 startup/recovery/error checks
  also passed during the final review. The full suite was not repeated after
  those final fixes; its observed failures remain explicitly recorded above.
- Real configured-model Textual probes in a disposable workspace read a file
  through `read_file`, returned **PHASE3_GROUPED_OK**, and produced one completed
  container with one thinking activity and one tool. Native v3 streaming warnings
  and an unawaited delayed-waiting coroutine warning were observed; the probe
  completed successfully. These probes do not establish live coverage of every
  provider, approval, artifact and recovery combination.
- Architecture/manual documentation was updated. No Mermaid diagram was added
  or modified, so the existing Mermaid validation remains unchanged. Compilation,
  24 relative document targets and whitespace checks passed before delivery.

## Visual evidence and limits

Actual headless Textual exports were rasterized with installed headless Chrome
and visually inspected. These are rendered UI frames, not design mockups:

- [Running](artifacts/phase3/running.png)
- [Completed response](artifacts/phase3/completed.png)
- [Expanded history](artifacts/phase3/history.png)
- [Expanded activity details](artifacts/phase3/details.png)
- [Configured-model result](artifacts/phase3/live.png)

The first four use deterministic events; the last uses the actual configured
model. Corresponding SVG exports are retained beside them. Native desktop/manual
interaction was not performed: browser-control access was unavailable and app
preview approval timed out. Headless layout checks and static image inspection
do not replace the user-driven checks in [prompts.md](prompts.md#grouped-assistant-transcript).

The intentional limitations are conservative legacy grouping, unavailable old
turn durations and the measured widget/restoration overhead. The approved
layout is implemented without pagination, history truncation or load-more
controls. Known repository baseline failures and the recorded intermittent
layout/timing failures remain outside a claim of a fully passing repository.
