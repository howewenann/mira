# Build a MIRA frontend

A frontend communicates with MIRA in three ways: **operations** ask it to do
work, **events** report what happens, and **requests** pause execution for a
user decision. Your frontend renders events and answers requests. MIRA executes
the agents and saves their conversation.

`MiraApplication` owns workspace resources and shutdown. A `MiraSession` owns
one durable conversation and its active operation. Your `Frontend` implements
`emit(event)` and `async request(request)`. A `FrontendEvent` is an observation;
a `FrontendRequest` needs an answer before execution continues. A
`SessionSnapshot` gives detached state for opening or refreshing a view.

Read the [architecture walkthrough](frontend-architecture.md) for diagrams,
identity examples, ownership, and the actual Textual integration. Continue below
for a working frontend and precise contracts. The public entry points are
`mira`, `mira.api`, and the pure preference editors in `mira.settings`; importing `core`, `agent`, `session`, `ui`, or `protocols`
is not required or supported for custom frontend implementations.

## Build a minimal frontend

Install and configure MIRA's Main model using the [README](../README.md).
Save this as a Python file and run it from your configured workspace:

```python
import asyncio
from mira import MiraApplication
from mira.api import FrontendEvent, FrontendRequest, MessageEvent

class MinimalFrontend:
    def emit(self, event: FrontendEvent) -> None:
        if isinstance(event, MessageEvent) and event.phase == "content":
            print(event.text, end="", flush=True)

    async def request(self, request: FrontendRequest):
        raise RuntimeError(f"Unsupported interaction: {type(request).__name__}")

async def main():
    app = await MiraApplication.start(workspace=".", frontend=MinimalFrontend())
    try:
        session = await app.open_session()
        await session.prompt("Reply only with PONG")
    finally:
        await app.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

This is the complete lifecycle: implement the two callbacks, start the
application, open a session, await a prompt, and always shut down. Unsupported
requests abort safely; they never approve an action. `emit` is synchronous, so
keep it fast and do not perform blocking input there. `request` is async because
Core waits for the answer. A GUI can update widgets or enqueue observations on
its own event loop.

The runnable [minimal example](../examples/mira_api/minimal_frontend.py) keeps
this lifecycle visible. Next read the
[full frontend](../examples/mira_api/full_frontend.py) for streaming tools,
blocking decisions, explicit inspection selection, and session restoration.
These two examples share the public API, not a custom framework.

## Operation reference

All operations below belong to `MiraApplication` or its returned `MiraSession`.
Async operations must be awaited. Use one prompt at a time per session.

| Operation | When to use it / result | Lifecycle implications |
| --- | --- | --- |
| `await MiraApplication.start(workspace=..., frontend=...)` | Load configuration, MCP, and native agents; returns application | Can request MCP approval and raise startup/configuration errors. |
| `await app.open_session(session_id=None, resume=False)` | Create a conversation, or reopen saved state with `resume=True`; returns session | Emits session `opened`; discovers saved work without executing it. |
| `app.session_exists(session_id)` | Check an externally supplied ID before opening | Returns bool; unknown IDs can otherwise create a session. |
| `await session.prompt(text)` | Execute a complete outer interaction; returns turn result with `final_text` | Streams events and requests; raises provider/request failures or `CancelledError`; emits `ready` after cleanup. Use events and snapshot as display contracts. |
| `await session.set_mode("act" or "plan")` | Select execution mode | Synchronizes native conversation context; refuses active operations or pending recovery. |
| `await session.cancel()` | Request cancellation from another task | Does not wait for prompt cleanup; await the prompt task as well. |
| `session.snapshot(include_transcript=True)` | Open/refresh authoritative display state | Detached state; pass `False` for telemetry and command metadata without transcript copying. |
| `session.inspection(id)` | Select a live or saved inspection by inspection ID or durable run ID | Detached `LiveInspection` or `None`; historical selection refreshes saved contents. |
| `session.inspections(origin_event_id=None)` | Discover saved run summaries, optionally for a tool entry | Detached list of mappings excluding transcript `events`; no execution. |
| `await session.resume_pending()` | Redisplay a saved request and resume its checkpoint | Same outer identity; returns result or `None` if no pending request. |
| `await session.continue_incomplete()` | Explicitly continue uncertain saved work | Side effects may repeat; returns result or `None`. |
| `await session.begin_goal(objective)` | Stage explicit Goal preparation | Follow with `prompt(objective)`; requires Act and Plan agents. |
| `await session.start_artifact(kind)` | Begin an explicit retained Plan/Goal attempt | Returns artifact mapping or `None`; follow with prompt to execute. |
| `await session.begin_artifact_revision(kind, artifact, feedback)` | Stage replacement of an exact retained artifact | Follow with a prompt; successful presentation commits replacement. |
| `await session.clear_artifact(kind)` | Remove retained artifact | Returns removed mapping or `None`. |
| `session.pause_active_artifacts()` | Mark interrupted formal work resumable | Synchronous; Core already uses it during interrupted turns. |
| `await session.close()` | Release one conversation's live state and connection | Cancels/joins active prompt, saves duration, emits `closed`; inspection queries become empty. |
| `await app.delete_session(session_id)` | Remove a saved conversation | Closes active projection first; returns bool. |
| `await app.clear_sessions()` | Remove all saved conversations | Closes sessions first; returns count. |
| `await app.shutdown()` | Release all sessions and shared resources | Always use in `finally`; safe after normal work ends. |

`kind` for artifact methods is `"plan"` or `"goal"`. State transitions reject
concurrent operations with `RuntimeError`. Core pauses interrupted artifacts;
frontends do not repeat cleanup or save session records.

### History, runtime and workspace operations

| Operation | Contract |
| --- | --- |
| `app.list_sessions()` | Detached saved summaries, including pin/custom title and latest user prompt; no transcript, database, or live session objects. Sort and wrap these for your own history UI. |
| `app.update_session(id, title=..., pinned=...)` | Change either metadata field without changing conversation recency; blank title restores the automatic title. |
| `await session.clear()` | Clear this transcript, inspections and native conversation threads while retaining formal work. |
| `await app.clear_history()` | Close sessions and remove saved conversations plus compaction archives; returns both counts. Open a session afterward. |
| `app.prompt_history()`, `app.record_prompt(text)`, `app.clear_prompt_history()` | Read, append and clear workspace composer history. |
| `await session.compact()` | Run native forced compaction for the selected mode; returns `compacted`, `reason` and archive/error metadata without adding a user turn. |
| `await session.context_report()` | Detached context composition report; the frontend formats its rows. |
| `app.configuration()`, `app.settings()`, `app.runtime()` | Detached editable preferences/profile names/issues, current disk preferences, and sanitized effective connection metadata respectively. No model credentials or native clients. |
| `await app.apply_settings(preferences, reset_mcp=False)` | Save preferences and prepare replacement native agent pairs before installation. Returns `(ok, message)` and restores saved preferences if reconstruction fails. Tracing configuration changes require Reload Runtime. |
| `await app.refresh_model()` | Refresh provider context metadata and rebuild only if the context window changed; normal prompts do this in Core. |
| `await app.reload(restart_mcp=False)` | Re-read effective configuration/resources and rebuild open sessions. `True` also restarts MCP; default retains live connections. Returns effective runtime metadata. |
| `app.tracing_preview(preferences)` | Credential-redacted tracing preview for an editor. |
| `app.configure_git(bool)` | Apply the existing Git-protection choice; returns `(ok, message, preferences)`. The frontend owns confirmation. |
| `await app.list_files()` | File names through the configured native project backend. |
| `await session.upload_files(paths)` | Import selected local files; returns visible references and per-file failures. |
| `await session.clear_uploads()`, `await app.clear_uploads()` | Remove imported files for this session or workspace respectively. |
| `app.report_error(error, source=..., session_id=..., context=...)`, `app.clear_diagnostics()` | Existing error reporting and retention operations. `mira.api.report_error(..., workspace=...)` also works before application startup. |

Runtime mutations require idle sessions. Startup failure can leave no application;
`mira.settings.read_configuration(workspace)` reads detached editor data and
`configure_workspace(workspace, preferences)` saves it for the next startup.
Offline `configure_git` and `preview_tracing` retain those editor capabilities too.
These offline operations never install changes into a running application.

### Commands, workflows and MCP

`app.commands(kind=None)` returns immutable `CommandInfo` descriptors for
`prompt`, `skill`, and `workflow` commands. They contain usage, argument data,
source and description, with no resolver, factory, graph, or credential.
`await app.prepare_input(text)` resolves a prompt/skill invocation. Pass its
`messages`, `display_text`, and optional `pinned_skills` to `session.prompt` as
`prepared_messages`, `display_text`, and `pinned_skills`. Ordinary text returns
`None`. MCP resource references come from `app.mcp.attachments_from_text(text)`
and go to the prompt's `attachments` parameter.

`app.workflow_invocation(text)` validates arguments and returns a command
descriptor plus detached inputs, or `None` for ordinary text. Execute a registered
command with `await session.run_workflow(text)`. Core constructs the native graph,
records the command, captures workflow history/inspections, and handles cleanup.
Use `session.cancel()` for the active workflow exactly as for an agent prompt.
The hidden `/workflow-demo` and `/workflow-demo-phase2` diagnostics use this same
operation; they retain their existing process-local history behavior.

`app.mcp` is a supported client, not the internal MCP manager:

- `servers` returns detached server snapshots; `resources` returns resource
  descriptors. Tools are metadata, never executable tool instances.
- `await discover_prompts(name=None)` and `await discover_resources(name=None)`
  lazily discover capabilities, optionally for one server.
- `await set_server_enabled(name, bool)`, `await set_server_always_allow(name, bool)`,
  and `await restart_server(name)` retain native connection/approval semantics.
- `settings()`, `commands()`, `resource_errors()`, `configured_count`,
  `usable_count`, and `show_status` support status panels and autocomplete.

MCP operations can issue the same blocking approval and elicitation requests as
normal execution. A frontend must keep serving requests while an operation is
awaited. It must not reproduce MCP persistence or agent reconstruction.

## Start MIRA and open a session

`MiraApplication.start` accepts optional `LaunchOptions(llm_direct=True)` for
the existing process-level Direct connection option. It resolves the workspace, loads configuration, starts
MCP resources, and constructs MIRA's native agents without constructing a UI.

```python
frontend = MyFrontend()
app = await MiraApplication.start(workspace=".", frontend=frontend)
session = await app.open_session()
```

The workspace must have a configured Main model. An explicit `config` mapping
may be passed to `start` when an embedding application already loaded MIRA's
configuration.

Open the latest saved session with:

```python
session = await app.open_session(resume=True)
```

Or request a specific durable session ID:

```python
session = await app.open_session(session_id="existing-session-id", resume=True)
```

Consumers that accept external session IDs can first call
`app.session_exists(session_id)` so an unknown ID is not created implicitly.

Keep the returned `MiraSession` if the frontend needs to call `snapshot()`
while handling an `ArtifactDisplayRequest`.

## Prompts, modes, and cancellation

Send one complete workflow and await its completion:

```python
await session.prompt("Summarize this project")
```

Events are the supported streaming output. After the await completes, use
`session.snapshot()` for authoritative current state.

ACT and PLAN are modes of one durable MIRA session:

```python
await session.set_mode("plan")
await session.prompt("Plan the requested refactor")

await session.set_mode("act")
await session.prompt("Inspect the current implementation")
```

`"action"` and `"planning"` are accepted aliases. PLAN uses its own LangGraph
thread identity; changing mode does not change the MIRA session ID.

To cancel from another task, wait until the frontend receives the session's
`RuntimeEvent(kind="session", state="turn_started")`. Calling `cancel()`
immediately after `create_task()` can run before the operation starts. For
example, let your frontend set an `asyncio.Event` named `turn_started` when it
receives that notification:

```python
frontend.turn_started.clear()
turn = asyncio.create_task(session.prompt("Long-running task"))
started = asyncio.create_task(frontend.turn_started.wait())
try:
    # Also surface startup errors rather than waiting for a missing notification.
    await asyncio.wait({turn, started}, return_when=asyncio.FIRST_COMPLETED)
    if not turn.done():
        await session.cancel()
    try:
        await turn
    except asyncio.CancelledError:
        pass
finally:
    started.cancel()
    await asyncio.gather(started, return_exceptions=True)
```

## Goals and retained Plans

Formal Plans are created by entering PLAN mode and prompting. The frontend
receives an `ArtifactReviewRequest` when the Plan is ready for review.

An explicit Goal uses the Goal preparation pipeline:

```python
await session.create_goal("Deliver the requested outcome")
```

Owned frontends also use these public session operations for retained formal
work:

- `session.start_artifact("goal" | "plan")` starts or resumes the retained
  artifact and returns its mapping, or `None` if none exists.
- `session.begin_artifact_revision(kind, artifact, feedback)` stages an exact
  replacement before the next prompt.
- `session.clear_artifact("goal" | "plan")` clears only that retained artifact
  and returns the removed mapping, or `None`.
- `session.pause_active_artifacts()` makes interrupted formal work resumable.

The first three operations above are async and must be awaited. Only one
current Goal or Plan is authoritative at a time. `await session.revise_artifact(kind,
feedback)` stages and executes a revision in one operation. `await
session.dismiss_artifact(kind)` hides historical controls while retaining current
formal work; clearing is a different operation.

## Interrupted execution

`await session.resume_pending()` redisplays a checkpointed interaction and
resumes the native graph after the frontend answers. It returns `None` if
there is no pending interaction. It does not submit another user message.

`await session.continue_incomplete()` explicitly continues saved incomplete
work. External side effects may have happened before interruption and may
repeat; obtain the user's decision before continuing. It returns `None` when
there is no incomplete operation. Ordinary prompts are refused while recovery
is pending. `session.snapshot(include_transcript=False).pending` is `none`,
`interaction`, or `incomplete`; no internal recovery descriptor crosses the API.

## Events

Every event includes `session_id`, `turn_id`, `message_id`, `event_id`,
`origin_event_id`, `namespace`, `metadata`, and `created_at`. Fields without a
corresponding identity remain empty or zero. Consumers should ignore fields
they do not display and accept new metadata keys.

`turn_id` is an opaque outer-interaction identity, independent of the completed
turn counter, user messages, model invocations, and graph thread IDs. New
prompts receive distinct IDs even after failure or cancellation. Explicit
checkpoint recovery retains the saved interaction ID; historical checkpoints
without one receive a new ID without regrouping old transcript entries.

`event_id` references a persisted entry's integer `id` within the session.
Several deltas may update the same entry. `origin_event_id` associates tool
updates and results with their original call entry. Transient drafts have no
durable event ID until promoted. Persisted entries retain turn and available
native message/provenance fields. Historical entries missing those fields
remain unassigned; do not infer exact turn boundaries from user messages.

### `MessageEvent`

`phase` is one of:

- `user`: a submitted user message.
- `content`: an assistant text delta.
- `reasoning`: a visible reasoning delta.
- `discard_reasoning`: remove reasoning later classified as internal.
- `end`: the current model-message stream ended.

`text` contains the delta. `content_blocks` preserves normalized LangChain
content blocks instead of converting them into a MIRA-specific block schema.
`mode` identifies action/planning transcript context and `attachments` contains
displayable attachment metadata.

### `ToolEvent`

`phase` is one of `delegation`, `arguments_delta`, `start`, `update`,
`approval_resolved`, `result`, `error`, `completed_result`, `completed_error`,
`recovered_start`, `recovered_result`, `recovered_error`, or `stop`.

Important fields are `name`, native `arguments`, `result`, `tool_call_id`,
`calls`, `status`, and `duration_ms`. Preserve tool-call IDs when updating an
existing visual row.

An individual `stop` includes `name`, `tool_call_id` when available, the call's
durable event reference, terminal `status`, and authoritative `duration_ms`.
A `stop` without a tool identity describes bulk cleanup. Later authoritative
results can replace a stopped outcome.

### `SubagentEvent`

`phase` is one of `live_start`, `live_tick`, `live_stop`,
`delegation_update`, `start`, `request_update`, `finish`, `cancel`,
`cancel_all`, `eval_start`, `eval_finish`, or `eval_cancel`.

The event includes task/result text plus graph provenance such as `origin`,
`eval_id`, `row_id`, `model`, and `label`.

### `RuntimeEvent`

`kind` is one of `startup`, `waiting`, `message_group`, `session`, `inspection`, or `configuration`.

- `waiting`: state is `start` or `finish`; start detail may contain `label`,
  `immediate`, and `elapsed` presentation hints.
- `message_group`: current state is `finish`.
- `session`: known states are `opened`, `turn_started`, `ready`, `cancelling`,
  and `closed`.
- `startup`: state is intentionally extensible human-readable progress text.
- `configuration`: state `changed` tells consumers to refresh detached runtime/settings metadata.
- `inspection`: state is `reset`, `append`, `delta`, or `status`; see
  [inspection retrieval](#inspection-retrieval-and-updates).

### `UsageEvent`

`usage` is the current usage/context update mapping. Use the snapshot dashboard
for the durable aggregate display.

### `CompactionEvent`

`phase` is `start` or `finish` for native DeepAgents context compaction.

### `ArtifactEvent`

`artifact_type` is `goal` or `plan`. `phase` is one of `proposed`,
`implement`, `active`, `revise`, `close`, `clear`, or `cancelled`. The event
includes `artifact`, `artifact_id`, and optional decision details. Durable
completion/pause status is authoritative in `session.snapshot()`.

### `RubricEvent`

`phase` is one of `verifying`, `grading`, `lifecycle`, `finish`, `cancel`, or
`status`. It carries the run/pass identity, iteration limit, grader model,
completed `evaluation`, nested native `lifecycle` event, or reconciled
`status`, as applicable.

### `MCPEvent`

`phase` is `initializing`, `initialized`, or `error`. `server` identifies a
server when relevant and `detail` contains the current projection or error.

### `WorkflowEvent`

`phase` is `run_start`, `task_start`, `task_waiting`, `task_resume`,
`task_finish`, `agent_start`, `agent_waiting`, `agent_resume`, `agent_finish`,
`run_finish`, or `run_cancel`. `workflow_id`, `task_id`, `agent_id`, and
`inspection_id` identify the corresponding execution objects. Fields include
task `name`, `step`, `status`, `error`, `input_state`, `result`, and
`final_state`. Respect `result_available` and `final_state_available`: a
legitimate `None` result differs from an unavailable result. Registered
Workflow launching still uses internal execution and persistence operations;
the public workflow authoring API is not a session execution API.

### `InformationEvent`

Contains displayable `text` or a structured `correction`. Known `kind` values
are `system`, `startup`, `info`, `status`, `warning`, `error`, `muted`, and
`correction`. The vocabulary is intentionally extensible; render an unknown
kind as ordinary system information.

## Blocking requests and exact responses

### `ApprovalRequest`

Sent for native LangGraph action approvals. `interrupts` contains the original
interrupt objects. Return one decision dictionary for every action request,
in encounter order:

```python
{"type": "allow_once"}
{"type": "allow_always"}
{"type": "reject_once"}
{"type": "reject_once", "message": "Reason"}
{
    "type": "edit",
    "edited_action": {"name": "write_file", "args": {"file_path": "safe.txt"}},
}
```

`allow_always` persists MIRA's tool policy and approves this invocation; a
persistence failure aborts before resume. The supported `approve` and `reject`
aliases remain accepted. Core translates the flat decisions to native
LangGraph values, using `Command(resume={"decisions": decisions})` for one
interrupt and partitioning by native interrupt ID for multiple interrupts.
The frontend does not construct that native resume mapping. Never approve by
default when no trusted approval UX is available.

### `AskUserRequest`

Sent for MIRA's native `ask_user` interrupt. Inspect `request.interrupt` for
`question` and `options`; return the selected option or free-form answer as a
string. Core passes it directly to `Command(resume=answer)`.

### `ArtifactReviewRequest`

Sent with `artifact_type`, the native finalizer `interrupt`, and the complete
provisional `artifact`. Return:

```python
{"action": "implement"}
{"action": "close"}
{"action": "revise", "feedback": "What should change"}
{"action": "clear"}
```

Implement accepts and starts the artifact. Close accepts and retains it without
starting. Revise and clear reject the proposal and preserve previously retained
formal work.

### `ArtifactDisplayRequest`

Sent for `show_goal` or `show_plan`. Read the requested retained artifact from
`session.snapshot()`, display it, and return a short result string such as
`"Current Plan displayed."`. Core uses the stringified value as the control
tool result and ends the display-only turn.

### `MCPApprovalRequest`

`server` contains only the server name and transport and `preview` is a safe launch/use
summary. Return exactly:

- `"allow"`: approve for the current process/fingerprint.
- `"always_allow"`: persist approval for the current fingerprint.
- `"deny"`: do not connect.

Any other value is interpreted as denial.

### `MCPElicitationRequest`

Unwrap `request.interrupt.value` when present. Its `requests` list describes
form or URL requests with stable `key` values. Return the native keyed shape:

```python
{"responses": {"request-key": {"action": "accept", "content": {"field": "value"}}}}
```

Form acceptance supplies a `content` object matching `requested_schema`.
URL acceptance needs only `{"action": "accept"}`. To decline or cancel a
request, return `{"action": "decline"}` or `{"action": "cancel"}` under
its key, without `content`. The full frontend example handles both modes.

## Session snapshots

```python
snapshot = session.snapshot()
```

`SessionSnapshot` fields:

- `session_id`: durable MIRA session ID.
- `workspace`: resolved workspace path.
- `mode`: `action` or `planning`.
- `runtime_state`: `ready`, `running`, `cancelling`, or `closed`.
- `title`: current session title.
- `turns`: completed/recorded turn count.
- `current_goal`, `current_plan`: mutually exclusive retained artifact mapping
  or `None`.
- `transcript`: tuple of detached normalized persisted event mappings, empty when `include_transcript=False`.
- `pending`: `none`, `interaction`, or `incomplete`.
- `execution`: mode/authoring/execution flags, without graph thread IDs or recovery objects.
- `history`: creation/update time, custom title and pin metadata.
- `dashboard`: usage/duration/context aggregate mapping.
- `model`: `name`, `alias`, `available`, `context_limit_tokens`, and `context_limit_source`.
- `tools`: tuple of mappings such as `{"name": ..., "description": ...}` for
  the current mode/stage.
- `resources`: mapping from resource category to tuples of metadata mappings.
- `rubric`: `{"enabled": bool, "max_iterations": int}`.
- `mcp`: server summaries, issues, and capability counts, for example
  `{"servers": (...), "issues": (...), "capabilities": {"tools": 2, ...}}`.

Nested mappings remain ordinary detached copies. Mutating them, including
nested tool arguments, attachments, artifact data, or MCP projections, cannot
change authoritative session/runtime state. The snapshot dataclass itself is
frozen. Copying the transcript costs time and memory proportional to its size;
use incremental events for streaming and take snapshots at session/operation
boundaries rather than on every token. Use `include_transcript=False` for lightweight status refreshes.

### Restore a conversation without execution

Render `snapshot.transcript` in its saved order. These are supported detached
**display records**, not graph messages to submit back to Core. Preserve IDs and
keep unknown historical fields; do not infer missing turn boundaries.

| Record `type` | Display data |
| --- | --- |
| `user`, `assistant`, `reasoning`, `info`, `system_error`, `interrupted` | `text`; user records can include `mode` and `attachments`. |
| `tool_call` | `name`, structured `args`, native `call_id`; its `id` is the origin for results/inspection navigation. |
| `tool_result` | `name`, `output`, `call_id`, `is_error` and optional `duration_ms`; update the originating call. |
| `correction` | Structured check/workflow/retry feedback, matching the correction information event. |
| `plan`, `goal` | Nested artifact under the matching key plus historical `status`; snapshot current artifact determines whether controls remain available. |
| `rubric` | `evaluation` and `max_iterations`, including criterion verdicts and verifier/grader lifecycle data. |
| `subagent` | `name`, `task_input`, `output`, `origin`, `status` and timing. Use public inspection queries for its transcript. |
| `workflow` | `command`, `workflow_id`, `workflow_name`, `status`, ordered `tasks`, timing and frozen final-state data. The command is embedded here; do not add a second user row. |
| `compaction` | The saved compaction summary/archive metadata. |

Records also carry `id`, `created_at` and available outer/native identity fields.
Historical runs are read through `inspections`, rather than by scanning storage.
Resource metadata retains tool `group` and subagent `kind`, `source_model` and
`graph_id` so another frontend can reproduce configuration controls without
inspecting native agents. The full example shows a simple history display.

## Inspection retrieval and updates

Inspection capture is session-owned and enabled for a frontend implementing
only the two public protocol methods. Opening an Inspector does not enable or
disable capture. `session.snapshot()` deliberately excludes inspection contents.

Discover IDs in `SubagentEvent.metadata["inspection_id"]`, rubric lifecycle payloads, and
`WorkflowEvent.inspection_id`. Live mutations also arrive as ordinary
`RuntimeEvent(kind="inspection")` observations. The notification has:

| Field | Contract |
| --- | --- |
| `session_id`, `turn_id` | Owning session and current outer interaction when available |
| `state` | `reset`, `append`, `delta`, or `status` |
| `detail["inspection_id"]` | Transcript key |
| `detail["update"]` | Detached `InspectionUpdate(operation, event)`; `event` is an `InspectionEvent` or `None` |

For a restored rubric evaluation, import `rubric_inspection_id` from `mira.api`
and call `rubric_inspection_id(grading_run_id, iteration, "verifier" or "grader")`.
`iteration` is zero-based. This existing identity builder connects a saved
evaluation to its phase transcript without reproducing an internal ID format.
The generated key can still be unavailable when an older session has no run.

Retrieve the initial state when selected:

```python
inspection = session.inspection(selected_id)
if inspection is not None:
    for item in inspection.events:
        print(item.kind, item.text, item.name, item.args)
```

`LiveInspection` is the existing transcript payload, also used for historical
results: `id`, `title`, `inspection_type`, `status`, and ordered `events`.
A live result's `id` is its inspection key; a historical result's `id` is its
durable run ID. Keep the selected query key for subsequent live notifications.
Statuses are `RUNNING`, `DONE`, `ERROR`, and `CANCELLED`. Event kinds are `user`,
`reasoning`, `assistant`, `tool_call`, `tool_result`, `tool_error`, and `error`.
Each `InspectionEvent` has `kind`, `text`, `name`, `args`, and `call_id`; fields
not used by its kind are empty. These payload classes are imported from
`mira.api`; they do not grant access to the store.

For a selected live transcript, apply `append` as a new item and `delta` as
text growth of its final item. On `reset`, retrieve and replay the selected
transcript because a prior tool draft or input may have changed. On `status`,
refresh status if displayed. Notifications carry only a detached changed item,
never all inspection contents. Ignore notifications for unselected IDs or other
sessions. If an update was missed while the view was closed, query again when
opening it. Do not query the full transcript for each text delta.

For historical navigation:

```python
runs = session.inspections(origin_event_id=parent_tool_event_id)
for run in runs:
    print(run["id"], run["display_name"], run["status"])
    transcript = session.inspection(run["id"])
```

Omit the filter to list all durable runs, including origin-less rubric and
workflow inspections. Summaries retain existing run metadata: `id`,
`inspection_id`, `origin_event_id`, `origin_tool`, `origin_call_id`, `row_id`,
`eval_id`, `inspection_type`, `display_name`, `task`, `status`, `started_at`,
`updated_at`, `duration_ms`, `output`, and optional `finished_at`. They exclude
`events`; fetch that transcript only on selection. Origin-less runs have
`origin_event_id=None`. Do not invent parent relationships for them.

Queries return detached nested values; modifying arguments, events, or summary
mappings cannot change authoritative state. Live queries and run summaries use
session-owned state. Historical transcript selection reads the existing saved
session once so externally updated history remains visible; it does not reload
storage during streaming. Storage read failures can raise `OSError` or a format
error. An unknown or
unavailable transcript returns `None`, an unknown parent yields `[]`, and closed
sessions yield `None` / `[]`. Older sessions may simply have no captured runs.

Live capture is process-local. Existing `runs[]` persist captured inputs,
messages, tools, terminal status, and origins. Text updates use the existing
batched save schedule; text is coalesced before copying the accumulated transcript.
Structural and terminal changes flush immediately.
Coordinator namespace/row maps and notifications are transient. Explicit
checkpoint recovery retains stored identities; cancellation finalizes observed
children without clearing their history. Session close clears live contents,
while reopening retrieves saved runs without rerunning an agent.

## Errors and cleanup

`MiraApplication.start()` may fail for configuration, model, MCP, or resource
startup errors. `session.prompt()` may raise provider/runtime errors and
`asyncio.CancelledError`. A frontend request handler may raise when it cannot
safely answer; Core propagates that failure and does not fabricate a decision.

Close an individual session when it is no longer needed:

```python
await session.close()
```

Always shut down the application in `finally`; shutdown closes remaining
sessions and MCP resources and is safe to call once normal work ends:

```python
app = await MiraApplication.start(workspace=".", frontend=frontend)
try:
    session = await app.open_session()
    await session.prompt("Your task")
finally:
    await app.shutdown()
```

Session IDs identify durable MIRA conversations. LangGraph thread IDs identify
execution threads inside those conversations; PLAN deliberately uses a
different graph thread. Do not use one as a substitute for the other.
