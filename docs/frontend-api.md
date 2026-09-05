# MIRA Python API

Use the MIRA API to run headless MIRA or build a custom frontend, GUI,
protocol adapter, service, or integration.

The same API is used by MIRA's Textual and terminal consumers. Application
classes are imported from `mira`; frontend events, requests, and snapshots are
imported from `mira.api`. Modules under `core`, `agent`, `session`, `ui`, and
`protocols` are implementation packages rather than supported developer entry
points.

## Installation and imports

Install MIRA and configure the workspace model as described in the main
README:

```bash
pip install mira
```

```python
from mira import MiraApplication, MiraSession
from mira.api import Frontend, FrontendEvent, FrontendRequest, MessageEvent
```

Start with the copyable
[`minimal_frontend.py`](../examples/mira_api/minimal_frontend.py), then see
[`full_frontend.py`](../examples/mira_api/full_frontend.py) for complete
interactive request handling.

## Lifecycle

```text
implement Frontend
      |
      v
MiraApplication.start()
      |
      v
open_session()
      |
      v
session.prompt()
      |
      +------> frontend.emit(event)
      |
      <------ frontend.request(request)
      |
      v
session.snapshot()
      |
      v
app.shutdown()
```

`emit` is synchronous and receives ordered notifications. `request` is async
because Core is paused until the consumer supplies a decision.

```python
class MyFrontend:
    def emit(self, event: FrontendEvent) -> None:
        print(type(event).__name__, event)

    async def request(self, request: FrontendRequest):
        raise RuntimeError(f"Interaction not implemented: {type(request).__name__}")
```

A production frontend must handle every request it can encounter. Raising an
exception aborts the active startup or turn rather than silently approving an
operation.

## Start MIRA and open a session

`MiraApplication.start` resolves the workspace, loads configuration, starts
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

To cancel from another task:

```python
turn = asyncio.create_task(session.prompt("Long-running task"))
await session.cancel()
try:
    await turn
except asyncio.CancelledError:
    pass
```

## Goals and retained Plans

Formal Plans are created by entering PLAN mode and prompting. The frontend
receives an `ArtifactReviewRequest` when the Plan is ready for review.

An explicit Goal uses the Goal preparation pipeline:

```python
session.begin_goal("Deliver the requested outcome")
await session.prompt("Deliver the requested outcome")
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

Only one current Goal or Plan is authoritative at a time.

## Events

Every event includes `session_id`, `turn_id`, `message_id`, `namespace`,
`metadata`, and `created_at` identity/provenance fields. Consumers should
ignore fields they do not display and accept new metadata keys.

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

### `SubagentEvent`

`phase` is one of `live_start`, `live_tick`, `live_stop`,
`delegation_update`, `start`, `request_update`, `finish`, `cancel`,
`cancel_all`, `eval_start`, `eval_finish`, or `eval_cancel`.

The event includes task/result text plus graph provenance such as `origin`,
`eval_id`, `row_id`, `model`, and `label`.

### `RuntimeEvent`

`kind` is one of `startup`, `waiting`, `message_group`, or `session`.

- `waiting`: state is `start` or `finish`; start detail may contain `label`,
  `immediate`, and `elapsed` presentation hints.
- `message_group`: current state is `finish`.
- `session`: known states are `opened`, `turn_started`, `ready`, `cancelling`,
  and `closed`.
- `startup`: state is intentionally extensible human-readable progress text.

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
{"type": "approve"}
{"type": "reject"}
{"type": "reject", "message": "Reason"}
{
    "type": "edit",
    "edited_action": {"name": "write_file", "args": {"file_path": "safe.txt"}},
}
```

Core resumes with `Command(resume={"decisions": decisions})`. Never approve by
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

`server` is the configured server state and `preview` is a safe launch/use
summary. Return exactly:

- `"allow"`: approve for the current process/fingerprint.
- `"always_allow"`: persist approval for the current fingerprint.
- `"deny"`: do not connect.

Any other value is interpreted as denial.

### `ConfirmationRequest`

Known `kind` values are `create_git_repo` and `continue_without_git`. Return
`True` to proceed or `False` to decline. These application confirmations are
not LangGraph HITL decisions.

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
- `transcript`: immutable tuple of persisted event mappings.
- `dashboard`: usage/duration/context aggregate mapping.
- `model`: `name`, `context_limit_tokens`, and `context_limit_source`.
- `tools`: tuple of mappings such as `{"name": ..., "description": ...}` for
  the current mode/stage.
- `resources`: mapping from resource category to tuples of metadata mappings.
- `rubric`: `{"enabled": bool, "max_iterations": int}`.
- `mcp`: server summaries, issues, and capability counts, for example
  `{"servers": (...), "issues": (...), "capabilities": {"tools": 2, ...}}`.

Nested mappings intentionally remain lightweight projections rather than a
second DTO hierarchy.

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
