# MIRA middleware reference

See the [overview](README.md) for registration order, settings defaults, scope
definitions and the [upstream reference](upstream.md) for native behavior.
Examples below illustrate the implemented rules; they are not promises about
what an LLM will choose to do. “If removed” describes the component's own
responsibility; other layers may still impose independent restrictions.

## Contents

- [RepeatedToolCallMiddleware](#repeatedtoolcallmiddleware)
- [FileReferenceMiddleware](#filereferencemiddleware)
- [ModelCompatibilityMiddleware](#modelcompatibilitymiddleware)
- [ProviderContextOverflowMiddleware](#providercontextoverflowmiddleware)
- [ExecutionContextMiddleware](#executioncontextmiddleware)
- [ExecuteToolDescriptionRewriteMiddleware](#executetooldescriptionrewritemiddleware)
- [ModelToolVisibilityMiddleware](#modeltoolvisibilitymiddleware)
- [PlanningStageEnforcementMiddleware](#planningstageenforcementmiddleware)
- [CorrectionMiddleware](#correctionmiddleware)
- [ModeTransitionMiddleware](#modetransitionmiddleware)
- [ProjectToolErrorMiddleware](#projecttoolerrormiddleware)
- [ContextReportMiddleware](#contextreportmiddleware)
- [SummarizationMiddleware, customized instance](#summarizationmiddleware-customized-instance)
- [MiraCompactionMiddleware](#miracompactionmiddleware)
- [InspectableCodeInterpreterMiddleware](#inspectablecodeinterpretermiddleware)
- [MiraRubricMiddleware](#mirarubricmiddleware)
- [_VerifierToolObserver](#_verifiertoolobserver)
- [VerifierReadCoverageGuard](#verifierreadcoverageguard)

## RepeatedToolCallMiddleware

**Purpose:** Stop a turn from repeatedly executing the same consecutive tool
request without changing its arguments.

**Why it exists:** A model can reread one file indefinitely, including after
errors. This guard bounds that narrow loop. It is locally maintained code
adapted from a DeepAgents contribution, not an automatically installed upstream
middleware.

**When it runs:** Sync/async tool wrappers decide whether to call the handler.
Sync/async before-model hooks stop the next model invocation after a terminal
block has been recorded.

**How it works:**

1. Collect AI tool calls from retained state since the last HumanMessage.
   Flatten parallel declarations in their AIMessage order, not completion order.
2. Locate the current call by ID, searching backward, and consider only calls
   through that declaration. If it is absent, append it for this calculation.
3. Fingerprint tool name plus schema-normalized arguments, serialized as sorted
   JSON. Pydantic validation fills defaults and performs configured coercion;
   fallback tries the public call schema, then raw arguments. IDs and outputs
   are not part of equivalence. Invalid arguments still reach the handler while
   within budget. Unknown non-JSON values fall back to `repr`.
4. Count backward until a different fingerprint. Permit `max_repeats` calls,
   then return `max_blocked_attempts` recovery errors without executing them.
   The next identical request returns a terminal error carrying
   `repeated_tool_call_guard_terminated=True`.
5. After the complete tool batch, the before-model hook sees that marker,
   appends a visible AI diagnostic, and redirects to end. Every parallel tool
   ID still gets its result; the guard does not cancel the batch halfway.

**Example:** With defaults and repeated `read_file('/a.py')`, requests 1–3
execute, 4–5 return recoverable errors, and 6 returns the terminal error. There
is no seventh model call on that ordinary route. Changing to `/b.py` before
termination starts a new consecutive sequence. `A, B, A, B` does not accumulate
repeats for A. Argument key order does not matter; a schema default explicitly
supplied or omitted can normalize to the same call.

**Scope:** Both main modes and owned raw subagent specs when enabled, including
native forks and MIRA-precompiled/workflow children. External compiled/remote
graphs are opaque. The normal configured general-purpose spec is owned raw.

**Configuration:** `system.repeated_tool_call_guard`: `enabled=false`,
`max_repeats=3`, `max_blocked_attempts=2`. Settings accept integer thresholds
1–100; the standalone class accepts positive non-boolean integers.

**State and persistence:** No mutable instance counters. Counts are derived
from messages; blocked ToolMessages and the final diagnostic are graph
messages. The terminal marker survives checkpoint resume. A new HumanMessage
resets the sequence; compaction/retained history limits what can be counted.

**Interaction with other middleware:** Errors count as calls too, regardless
of success. Request-only mode notices do not reset this state-based scan.
After-agent hooks still run after the end redirect: Rubric can ask for another
attempt, and a new revision HumanMessage resets the scan. This is not semantic
progress checking, interleaved-cycle detection, or a universal run limit.

**If removed:** Consecutive identical calls can continue until another runtime
limit, interruption or model decision ends them.

**Source:** [implementation](../../agent/middleware/repeated_tool_call_guard.py),
[standalone tests](../../tests/test_repeated_tool_call_guard.py),
[child construction tests](../../tests/test_subagent_compilation.py).

## FileReferenceMiddleware

**Purpose:** Tell the model how to inspect files and MCP resources explicitly
mentioned by the user without loading their contents automatically.

**Why it exists:** An `@file` token is a reference, not the file's contents, and
MIRA file tools expect virtual workspace paths.

**When it runs:** Sync/async model wrappers on every model call.

**How it works:** Read the latest real HumanMessage from graph state; parse
plain or quoted `@` mentions, trim trailing punctuation from unquoted values,
replace backslashes, remove leading `./`, add `/`, and deduplicate exactly in
input order. Separately scan message metadata for all `mira_mcp_attachments`
of kind `mcp_resource`, deduplicating by server/URI. Append system guidance with
normalized paths or exact `read_mcp_resource(server, uri)` instructions.
An email-like `word@host` is excluded by the token boundary. This parser does
not check file existence or resolve host paths.

**Example:** `Compare @src\main.py and @"design notes.md".` adds guidance
listing `/src/main.py` and `/design notes.md`. The original prompt stays intact;
the model must still call a reader. An attached MCP resource adds its server
and URI rather than its body.

**Scope:** Main Plan/Act; native forks inherit it. Ordinary isolated children
do not automatically receive it.

**Configuration:** Always registered on main graphs; acts only when references
or resource metadata are present. No enablement setting.

**State and persistence:** Temporary system-message override only. Uses state
messages, so it does not mistake the transient mode notice for a new user turn.

**Interaction with other middleware:** Complements filesystem tools and MCP
resource readers. Guidance honors an explicitly requested reader and otherwise
asks for an appropriate available read-only reader; it neither widens tool
visibility nor performs the reads.

**If removed:** Models receive the user's reference tokens but lose MIRA's
normalized path and attachment-reading guidance.

**Source:** [implementation](../../agent/middleware/file_references.py),
[tests](../../tests/test_file_references.py).

## ModelCompatibilityMiddleware

**Purpose:** Repair specific message representations and file-tool arguments
at the model boundary.

**Why it exists:** MIRA uses ChatAnyLLM and virtual workspace file paths;
provider/library representations do not always line up with those interfaces.

**When it runs:** Sync/async model wrappers normalize the incoming request,
then normalize each AIMessage in the successful response.

**How it works:** These are the complete implemented transformations:

| Transformation | Exact condition | Before → after |
| --- | --- | --- |
| Standard image block to OpenAI-style image block | Message content is a list and block `type == 'image'`; model profile does not explicitly reject `image_inputs`; for ToolMessage it must also not reject `image_tool_message` | `{'type':'image','url':'https://example.test/a.png'}` → `{'type':'image_url','image_url':{'url':'https://example.test/a.png'}}` via LangChain's converter; base64 blocks become image data URLs |
| Supply response provider identity | AI response lacks `response_metadata['model_provider']` | `{}` → `{'model_provider':'anyllm'}`; an existing provider value is preserved by `setdefault` |
| Rename file argument alias | Tool name is exactly `read_file`, `write_file` or `edit_file`; `args` is a dict; `path` exists and `file_path` does not | `read_file({'path':'/notes.txt'})` → `read_file({'file_path':'/notes.txt'})`; if both keys exist, no alias is removed |
| Convert workspace-contained host absolute path | Same three tools; string `file_path` does not start `/`; expanded path is absolute and resolves beneath configured workspace | On Windows, workspace `D:\project`: `D:\project\src\a.py` → `/src/a.py` |
| Mirror the file-call repairs in content blocks | AIMessage has tool calls and its list content contains dicts with the same `name`/`args` shape | A structured tool block with `name='write_file', args={'path':...}` receives the same alias/path repair as `message.tool_calls` |

The metadata addition supports LangChain's provider-aware reported-token
validation; it is not a claim that the underlying endpoint is itself named
AnyLLM. File repairs address generated argument aliases and host paths rather
than performing general tool-call validation.

Paths already starting `/`, relative paths, absolute paths outside the
workspace, non-string values, and resolution failures pass through. On POSIX,
an absolute path already starts `/`, so that early return applies. No rewrite
is implemented for `ls`, `glob`, `grep`, `delete`, arbitrary tools, raw JSON
argument strings, or other provider-specific content formats. Unsupported
images are left for native UnsupportedContentMiddleware, not dropped here.

**Example:** A returned `write_file(path='D:\project\notes.txt', content='Hi')`
becomes `write_file(file_path='/notes.txt', content='Hi')` before tool dispatch.
Other argument values and the call ID are unchanged.

**Scope:** Main Plan/Act, owned raw subagents, and forks. Not automatically
installed in the plain LangChain Rubric verifier/grader graphs.

**Configuration:** Always installed on main graphs and appended to owned raw
specs if absent; workspace comes from construction. Profile capability flags
gate image conversion; missing flags are treated as supported here.

**State and persistence:** Incoming image repair copies messages in a request
override. Response metadata and tool-call/content repairs mutate returned
AIMessage objects, so the corrected response enters graph state and execution.

**Interaction with other middleware:** Native UnsupportedContent filters by
capability later. The summarizer receives the repaired response metadata.
Filesystem tools execute the corrected `file_path`; permissions still apply.

**If removed:** Standard image blocks may reach a provider adapter in the wrong
representation, generated file aliases/host paths may fail, and provider-aware
usage checks lose the supplied identity when the adapter omits it.

**Source:** [implementation](../../agent/middleware/model_compatibility.py),
[middleware tests](../../tests/test_middleware.py),
[filesystem argument tests](../../tests/test_filesystem_tool_args.py).

## ProviderContextOverflowMiddleware

**Purpose:** Convert recognized context-window failures into the exception that
native DeepAgents compaction recovery understands.

**Why it exists:** Providers report an oversized prompt using different error
types and text. A recognized error should enter the native recovery route.

**When it runs:** Sync/async model wrappers catch exceptions from their inner
handler. They do not proactively count tokens.

**How it works:**

1. An existing LangChain `ContextOverflowError` is re-raised unchanged after
   attaching a MIRA UI notice if it has none.
2. For other exceptions, concatenate `str(exc)` and truthy `message`, `body`,
   `response` attributes, then lowercase. No traversal of exception causes is
   performed. Empty text or any text containing `rate limit` is rejected.
3. Recognize any of these direct substrings: `context_length_exceeded`,
   `input tokens exceed the configured limit`, `exceeds the context window`,
   `prompt is too long`.
4. Also recognize `context` or `prompt` plus any of `exceed`, `exceeded`,
   `exceeds`, `too long`, `maximum`, `configured limit`; or `token` plus one of
   those words plus `limit`. These are broad text heuristics, not a provider
   exception-class allowlist.
5. Raise `ContextOverflowError('provider context limit reached')` chained from
   the original exception, with `_mira_context_notice` attached. Unrecognized
   exceptions propagate unchanged.

**Example:** A provider exception whose body says
`{'code':'context_length_exceeded'}` becomes a chained ContextOverflowError.
The outer summarizer can reduce context and retry. An error containing
`rate limit` passes through even if other words would match. The attached UI
notice says “Provider context limit reached. Compacting older context and
retrying.” That notice describes intended recovery, not a guarantee it can fit.

**Scope:** Main Plan/Act and inherited forks. Ordinary isolated subagents use
the upstream summarizer's own classifier without this MIRA translator.

**Configuration:** Always registered on main graphs; no provider setting.

**State and persistence:** No graph update or conversation insertion. Notice
helpers use exception attributes, a ContextVar and a process fallback value;
rendering consumes/clears the notice and can mark it already rendered.

**Interaction with other middleware:** It must remain inside automatic
summarization for the translated error to be caught there. Summarization does
the reduction/retry; ContextReport only observes estimated usage. Irreducible
input or recovery exhaustion can still escape to the runner as overflow.

**If removed:** Provider errors already understood upstream still recover;
additional signatures recognized here can escape without the MIRA notice or
native overflow route.

**Source:** [implementation](../../agent/middleware/context_overflow.py),
[tests](../../tests/test_context_overflow.py).

## ExecutionContextMiddleware

**Purpose:** Make the currently executing native ToolRuntime available to
trusted deterministic nested Python calls.

**Why it exists:** Bound tool/agent capabilities need the outer call's state,
configuration, stream writer and interrupt context when invoked inside tools.

**When it runs:** Sync/async tool wrappers for calls reaching this middleware.

**How it works:** Set `_ACTIVE_TOOL_RUNTIME` to `request.runtime`, invoke the
remaining handler inside the narrowly scoped unparameterized-runtime warning
filter, and reset the ContextVar token in `finally`, including errors and
interrupts. Typed MiraContext separately carries bound tools, bound agents,
workspace, and optional mode-transition metadata.

**Example:** A project tool calls a bound child agent; that nested call can
retrieve the outer ToolRuntime. After the project tool returns or raises, the
previous runtime binding is restored.

**Scope:** Main Plan/Act and owned raw subagent specs, including workflow-local
compilation and forks. PlanningToolContext extends the context for formal tools.

**Configuration:** No toggle; appended to raw specs only when absent.

**State and persistence:** Process/task-local runtime binding only. Does not
persist capabilities or runtime objects into conversation/checkpoints.

**Interaction with other middleware:** It preserves native tool/HITL execution
instead of substituting a tool executor. Outer guards can reject before this
wrapper is entered; cleanup still runs when an inner layer raises.

**If removed:** Trusted nested Python calls lose the current outer runtime
binding used by MIRA's capability API.

**Source:** [implementation](../../agent/execution/context.py),
[tests](../../tests/test_execution_context.py).

## ExecuteToolDescriptionRewriteMiddleware

**Purpose:** Give the model MIRA-specific shell and workspace-path instructions
in the visible `execute` tool description.

**Why it exists:** File tools use `/file.py` as a virtual workspace path, while
the host shell needs a workspace-relative or real host path.

**When it runs:** Sync/async model wrappers.

**How it works:** For visible tools named exactly `execute`, copy the dict,
Pydantic tool, or copyable object with `MIRA_EXECUTE_TOOL_DESCRIPTION`. Leave
other tools and already-rewritten descriptions alone. If generic copying
fails, return the original tool. Guidance covers directory checks, quoting,
path conversion, search/read tools, command separators and timeout behavior.

**Example:** The model sees `python scripts/check_path.py` as the shell form of
file-tool path `/scripts/check_path.py`, instead of treating that virtual path
as a host absolute path.

**Scope:** Registered in both main modes and inherited forks. Plan normally
hides execute, so registration usually has no visible Plan effect.

**Configuration:** No own setting. Execute visibility/backend support is
configured elsewhere; execute defaults disabled.

**State and persistence:** Copied request tool descriptions only; no change to
the executor, arguments, working directory, approval policy or graph state.

**Interaction with other middleware:** Runs after Filesystem's description
shaping. Later visibility can hide the tool altogether. Prompt instructions
are guidance, not shell enforcement.

**If removed:** The native execute description remains, without these MIRA
path examples and host-shell guidance.

**Source:** [implementation](../../agent/middleware/execute_tool_description_rewrite.py),
[tests](../../tests/test_middleware.py).

## ModelToolVisibilityMiddleware

**Purpose:** Make advertised tool names and permitted direct tool calls agree
with MIRA's configured availability or a subagent's explicit allowlist.

**Why it exists:** A tool can remain registered in ToolNode even when hidden
from the model; a fabricated call to its name must also be rejected.

**When it runs:** Sync/async model wrappers and tool wrappers.

**How it works:** Remove tools whose names are excluded or absent from a
non-`None` allowlist. At execution, return a status-error ToolMessage
`Error: <name> is not available.` without calling the handler for those names.
Exclusion wins over allowlisting; an empty allowlist allows no named tools.
With no allowlist, an unnamed entry is retained unless another rule applies.

**Example:** `execute` can remain in the native registry while disabled;
it is omitted from request.tools, and a direct generated execute call gets an
error instead of running. A worker allowlist `('read_file',)` also hides the
other automatically registered filesystem tools.

**Scope:** Main modes always; raw children with explicit tool declarations;
forks inherit/replace by middleware name. It is not automatically added as a
global exclusion filter to every isolated child lacking an explicit tools list.

**Configuration:** Main excluded names combine Act formal-control exclusions,
Plan-disabled tools, disabled built-ins, execute/backend availability, and Plan
read-tool access. Subagent discovery supplies `allowed_tools`. Local/MCP tools
are also selected by factory/discovery before model requests.

**State and persistence:** Request tool-list override; rejected ToolMessages
enter normal tool history. No additional state channels.

**Interaction with other middleware:** Stage enforcement restricts formal
workflow stages and `tool_choice`; this middleware enforces configured names.
Filesystem permissions remain a separate backend boundary. Native profile
tool exclusions run later and implement another provider-owned filter.

**If removed:** A merely hidden registered tool can remain callable, and
explicit worker allowlists no longer filter middleware-injected tools.

**Source:** [implementation](../../agent/middleware/model_tool_visibility.py),
[tests](../../tests/test_model_tool_visibility.py),
[subagent tests](../../tests/test_subagent_tools.py).

## PlanningStageEnforcementMiddleware

**Purpose:** Expose the correct formal Plan/Goal tools for research and force
the matching finalizer once preparation has completed.

**Why it exists:** Formal artifacts must follow preparation, separate Success
Criteria generation, then finalization; prose or a wrong-stage call cannot
substitute for that lifecycle.

**When it runs:** Sync/async model and tool wrappers on the Plan graph.

**How it works:** Read `planning_stage`, defaulting to `plan_research`.

| Stage | Model-visible surface | Tool execution guard |
| --- | --- | --- |
| `plan_research` | Retain available tools except other prepare/finalize controls; expose `prepare_plan`; put `show_plan` first; clear forced tool choice | Reject `prepare_goal`, either finalizer; permit `prepare_plan` and ordinary tools |
| `goal_research` | Corresponding `prepare_goal`, prioritize `show_goal`; clear forced choice | Reject `prepare_plan`, either finalizer |
| `plan_finalize` | Only `finalize_plan`, with `tool_choice='required'` | Reject every other name, including research/read/show tools |
| `goal_finalize` | Only `finalize_goal`, with `tool_choice='required'` | Reject every other name |

Research filtering removes only the other formal prepare/finalize controls;
it does not itself remove every non-research tool or the other show tool.
The factory and visibility/permission layers own broader Plan restrictions.
If the required finalizer is missing from the request, raise RuntimeError.
Wrong-stage calls return actionable status-error ToolMessages without execution.
The four control names are checked against their exact required stage.

**Example:** After `prepare_plan` and Success Criteria generation move state
to `plan_finalize`, a previously available `read_file` disappears. A fabricated
read call returns “Only finalize_plan can be called now.” The model must call
the single finalizer rather than stop with prose.

**Scope:** Plan graph, including Goal construction on that graph. Never
directly installed on the Act graph. Native Plan forks inherit the instance.

**Configuration:** Always enabled for Plan regardless of Rubric settings.
Stages are supplied by orchestration/formal tools, not a middleware toggle.

**State and persistence:** Declares checkpointed stage and planning payload
channels (authoritative request, prior artifact/criteria/feedback, objective,
constraints, research evidence, criteria). It reads them for enforcement;
stage advancement is owned by tools/orchestration, not these wrappers.

**Interaction with other middleware:** ModelToolVisibility can further restrict
names but does not choose a stage or force a tool. Correction applies only in
research; formal finalization uses this forced tool contract instead.

**If removed:** The model can see wrong-stage formal controls and use other
tools during finalization; missing finalizer wiring is no longer detected here.

**Source:** [implementation](../../agent/middleware/planning_stage_enforcement.py),
[policy](../../agent/planning/policy.py),
[tests](../../tests/test_middleware.py).

## CorrectionMiddleware

**Purpose:** Turn an invalid no-tool stop into a bounded corrective retry using
workflow-owned rules.

**Why it exists:** A model may announce research, a question, or preparation
and stop without performing the corresponding action.

**When it runs:** Before-agent resets bookkeeping; model wrappers append
reminders; after-model clears a resolved correction when tools are called;
after-agent inspects natural stops and may redirect to the model. Both sync
and async variants are provided.

**How it works:** MIRA installs two PlanningResponseStatusRules, one for
`plan_research` and one for `goal_research`. Exactly one applies in those stages;
neither applies in finalization. The generic middleware knows no Plan/Goal
vocabulary itself. Multiple active rules at a stop raise RuntimeError.

Every research model call gets the rule's **ordinary reminder** appended to
the system prompt, even on the first attempt. A **corrective retry** only
occurs after a no-tool AI response fails inspection: store the retry prompt and
counter by protocol ID, emit a `correction` stream event, and return
`jump_to='model'`. The next request includes both ordinary reminder and pending
correction. It does not insert a synthetic conversation message.

| Exact terminal line on a no-tool response | Implemented decision |
| --- | --- |
| `RESPONSE_STATUS: COMPLETE` | Accept if this is the sole recognized status line and final non-empty line |
| `RESPONSE_STATUS: NEEDS_RESEARCH` | Retry with instruction to call a research tool |
| `RESPONSE_STATUS: NEEDS_USER_INPUT` | Retry with instruction to call `ask_user` |
| `RESPONSE_STATUS: READY_TO_PREPARE_PLAN` | In Plan research, retry to call `prepare_plan`; invalid in Goal research |
| `RESPONSE_STATUS: READY_TO_PREPARE_GOAL` | In Goal research, retry to call `prepare_goal`; invalid in Plan research |
| Missing, duplicated, nonterminal, whitespace-altered or wrong-stage line | Retry to perform the action or return a valid completed answer |

Matching uses exact ASCII lines from visible `message.text`. Blank trailing
lines are ignored. The reminder describes substantive completion, but the
implementation does **not** semantically verify that a response labeled
COMPLETE actually satisfies the user. Responses containing tool calls bypass
natural-stop classification and clear retry/pending maps; another invalid
stop after tools starts a fresh correction sequence.

**Example:** `I will inspect the file.\nRESPONSE_STATUS: NEEDS_RESEARCH` with
no tool call is rejected. The next model request says to perform research now.
If it calls `read_file`, the pending retry is cleared. If it keeps stopping
incorrectly, the default allows two retries after the first failed stop; a
further rejection appends a fixed AI failure message saying the requested work
may be incomplete and emits an exhausted correction event.

**Scope:** Plan/Goal research in the main Plan graph; inherited Plan forks
subject to their stage. Not main Act or the Rubric verifier.

**Configuration:** `system.planning_response_status.max_retries=2`, settings
range 1–20; class requires a positive integer and nonempty rules with unique
protocol IDs. Research enforcement has no Rubric-dependent enable switch.

**State and persistence:** Private `_correction_retries` and
`_correction_pending` are graph channels that survive checkpoint resume within
the attempt. Before-agent resets independent external turns. Retry instructions
remain request-only, but rejected AI responses remain in messages and the
exhausted failure message is appended to state.

**Interaction with other middleware:** PlanningStageEnforcement owns the
finalization boundary. ModeTransition instead identifies the current mode's
authority for one bound turn; it neither validates stops nor retries. Rubric
evaluates outcome criteria after Act work through a different lifecycle.

**If removed:** Research can finish with missing/invalid status or an announced
action never performed, without deterministic repair or exhaustion reporting.

**Source:** [middleware](../../agent/middleware/correction.py),
[actual rules](../../agent/planning/response_status.py),
[generic tests](../../tests/test_middleware.py),
[workflow tests](../../tests/test_planning_response_status.py).

## ModeTransitionMiddleware

**Purpose:** Remind the model of the current Plan/Act authority throughout the
first user turn after a mode change, without adding that reminder to history.

**Why it exists:** Inherited conversation can contain prior-mode restrictions
or instructions that conflict with the current mode's tools and prompt.

**When it runs:** Sync/async model wrappers, after automatic summarization and
earlier MIRA request transforms, on each model invocation of the bound turn.

**How it works:** The middleware does **not** detect mode switches itself.

1. Application `select_mode` compares the requested mode with `mode['planning']`.
   A real change stores `mode_transition={'mode':'plan' or 'act'}`; selecting
   the already-active mode does not create a notice. Other formal-workflow
   entry/implementation paths also queue the corresponding transition.
2. Turn orchestration binds the pending transition to the first real user
   HumanMessage's stable ID and current thread ID before graph execution,
   persisting this small metadata record. The session semantic state carries
   it across restart; the prose is not saved there.
3. Orchestration puts a copy into trusted `runtime.context.mode_transition`,
   including checkpoint recovery. The middleware requires a known mode and
   nonempty message ID.
4. Copy request.messages, remove an existing notice with the same anchor/name/
   ID, and insert a HumanMessage immediately before the bound message. Its
   name is `mira_mode_transition`; ID is `mira_mode_transition:<anchor>`.
5. If native compaction removed the anchor from the effective request, locate
   it in raw state and insert before the first retained continuation from
   that turn, or at the end if none remains. If the anchor is absent from raw
   state too, do nothing; never resurrect archived messages or split tool pairs.

**Example:** After Act → Plan, a model request contains
`[earlier context, PLAN notice, real user request, ...tool continuation]`.
The notice says the environment is read-only, previous Act instructions are
historical, and current Plan restrictions/tools govern. Plan → Act says prior
Plan restrictions no longer govern, while current permissions and approvals
still apply. A later `read_file` continuation receives the same notice again,
deduplicated, before the same real user anchor.

**Scope:** Main Plan/Act; inherited forks only act when their runtime context
and retained anchor support the notice. Not ordinary isolated workers.

**Configuration:** Always registered; activation is controlled by trusted
transition metadata, not inference from prose or a user-supplied status token.

**State and persistence:** `request.override(messages=...)` only. No graph
message update, no saved synthetic turn, no notice in session transcript,
summary archives or mode handoffs. Only mode/thread/message IDs persist in
session semantic metadata. Normal completion clears the binding; interrupted
execution keeps it for resume. Restore logic clears a stale binding when the
checkpoint already completed before the semantic save. This ownership is why
guidance survives tools, cancellation and restart without becoming history.

**Interaction with other middleware:** Native summarization runs first, so
the notice is neither summarized nor archived. FileReference and the repeated
guard inspect real state messages. ContextReport observes the notice. It adds
no tool permissions, stage changes, response inspection or retry budget.

**If removed:** Current tools and policy still apply, but the model loses
explicit turn-local guidance that prior-mode instructions are historical.

**Source:** [middleware](../../agent/middleware/mode_transition.py),
[mode detection](../../core/application/app.py),
[binding/lifetime](../../core/execution/turns.py),
[session recovery](../../core/application/session.py),
[tests](../../tests/test_mode_transition.py),
[recorded probe evidence](../../tests/probes/mode_transition_results.md).

## ProjectToolErrorMiddleware

**Purpose:** Convert ordinary exceptions from selected project tools into
model-visible tool errors so the agent can recover.

**Why it exists:** User workspace Python tools can raise exceptions that would
otherwise escape the turn instead of becoming useful feedback.

**When it runs:** Sync/async tool wrappers, only for the captured project-tool
name set.

**How it works:** Call the inner handler; re-raise `GraphBubbleUp` so native
interrupt/control flow survives. Catch other Exceptions and return a
ToolMessage with status error, matching name/call ID, and
`<name> failed in the MIRA runtime.\n<ExceptionType>: <detail>`. Empty exception
text falls back to its type. Non-project tools bypass this catch.

**Example:** Project `lookup` raises `ValueError('unknown key')`; the model
receives that typed error and can choose a different key instead of losing
the whole turn.

**Scope:** Main modes when active project tools exist; added to owned raw
children and inherited by forks. The plain Rubric verifier is not automatically
given this middleware even when it receives a selected project tool object.

**Configuration:** Factory captures names of active tools whose metadata
source is `project`; no own toggle or blanket MCP/built-in error policy.

**State and persistence:** Returned error ToolMessage becomes ordinary graph
history; no extra state schema.

**Interaction with other middleware:** Does not swallow HITL interrupts. The
repeated guard still counts failed requests. It is scoped exception recovery,
not tool visibility or validation.

**If removed:** Ordinary project-tool exceptions may abort the turn rather
than enter the tool-result/model continuation route.

**Source:** [implementation](../../agent/middleware/project_tool_errors.py),
[tests](../../tests/test_project_tool_errors.py).

## ContextReportMiddleware

**Purpose:** Capture the request at its position in the chain for context
reports and native token-usage observations.

**Why it exists:** The UI needs inputs and estimates that reflect actual
request preparation rather than reconstructing context from visible chat.

**When it runs:** Sync/async model wrappers just before passing to the next
handler.

**How it works:** Record messages, system message and tools through context
observation helpers. If the shared summarizer exposes callable `_count_tokens`,
call it with those three inputs and record the resulting DeepAgents estimate.
Then pass the original request onward unchanged.

**Example:** After compaction, observe summary plus retained messages rather
than counting the entire raw conversation again; include the mode notice when
one was injected earlier.

**Scope:** Main modes and inherited forks; not automatically isolated workers.

**Configuration:** Always installed with the main graph's shared summarizer.
No independent thresholds or compaction decisions.

**State and persistence:** Records observation data through context-scoped
helpers, not graph channel updates. Does not rewrite requests or results.

**Interaction with other middleware:** Runs after MIRA transforms but before
upstream Skills/Memory/cache/unsupported-content wrappers. Its observation is
therefore not the final provider request. Token monitoring cannot itself
compact, reject or retry a model call.

**If removed:** Execution can continue, but the context-report inputs and
native estimate recorded at this boundary are lost.

**Source:** [middleware](../../agent/middleware/context_report.py),
[observation helpers](../../core/context/observation.py),
[tests](../../tests/test_context_report.py).

## SummarizationMiddleware (customized instance)

**Purpose:** Reduce the effective conversation when native thresholds or a
provider overflow require it, with MIRA-safe archives and summary replay.

**Why it exists:** Full history can outgrow a model's input window; its archive
must remain useful and must not replay provider-bound reasoning metadata.

**When it runs:** Native sync/async model wrappers. MIRA customizes an instance
returned by `create_summarization_middleware`; its public name replaces the
DeepAgents default slot in place. There is no local class of this name.

**How it works:** The native engine applies prior summary events, estimates
tokens including the system message/tool schemas at its boundary, truncates
older large tool arguments when configured, archives an older prefix, generates
a summary, and calls the model with summary plus a retained suffix. It also
does bounded overflow recovery and can offload oversized retained tool results.
See [native details](upstream.md#summarizationmiddleware-and-its-langchain-helper).

MIRA then applies these instance customizations:

1. An optional Summary model replaces only LangChain helper `_summary_model`,
   wrapped with the helper's normal retry exception policy. Main still supplies
   profile limits, token counting and reported-usage identity checks.
2. Before archive append, download the existing history path and distinguish
   genuinely missing files from read errors/absent content. Refuse unsafe reads.
3. Sanitize archive messages: retain visible text and plain tool-call facts;
   drop reasoning/provider metadata and prior summary messages. Preserve roles
   as new safe messages rather than raw provider objects.
4. Normalize newly built and checkpoint-replayed summary messages to plain
   HumanMessages tagged `mira_summary`, retaining visible summary text while
   dropping provider-hostile metadata.

**Example:** Instead of sending 100 old turns, the next request contains a
summary and a recent suffix with tool pairs preserved. Raw state still holds
the original messages and an event describing the cutoff. A summary containing
provider reasoning blocks is replayed as visible-text HumanMessage content.

**Scope:** Main Plan/Act. Isolated children normally get independently created
unmodified native engines. Native forks can replace their engine with the
inherited prepared instance because the public middleware name matches.

**Configuration:** Factory defaults with integer `max_input_tokens` profile:
trigger at 85%, retain roughly 10%; truncation uses the same fractions. Without
that profile field: 170,000-token summary trigger, retain 6 messages; argument
truncation uses a 20-message trigger/20-message keep policy. Retention respects
native tool-pair cutoffs, so these are policies rather than exact output sizes.
`models.summarization=null` uses Main; no separate MIRA auto-compaction toggle.

**State and persistence:** Native `_summarization_event` and
`_summarization_session_id` persist; raw messages are not deleted for normal
summary projection. Tail offload can replace message content by ID. History is
archived through the backend. System/request addenda are not graph messages.

**Interaction with other middleware:** MiraCompactionMiddleware shares this
exact object, as does ContextReport. DeepAgents' own summarizer is replaced,
not run alongside it. Overflow translation runs inside it; the mode notice
runs after its projection. Forced `/compact` uses its retention/summary/archive
methods but a separate completed-checkpoint commit path.

**If removed:** If omitted from the MIRA-supplied list, DeepAgents inserts its
default native summarizer, so automatic compaction remains but MIRA's optional
summary-model assignment, safe archive precheck/sanitization and summary replay
normalization are lost. Removing the effective native slot entirely would
also remove automatic reduction/recovery.

**Source:** [instance preparation](../../agent/middleware/compaction.py),
[builder](../../agent/middleware/builder.py),
[native integration tests](../../tests/test_compaction_native.py),
[metadata tests](../../tests/test_middleware.py).

## MiraCompactionMiddleware

**Purpose:** Expose native `compact_conversation` and plan explicit forced
compaction using the same automatic summarization engine.

**Why it exists:** The model can request early compaction, and the user can
force `/compact` without starting another conversational turn.

**When it runs:** Inherited sync/async model wrappers support optional guidance,
but MIRA leaves `system_prompt=None`, so they pass requests through unchanged.
The contributed tool compacts only when called. MIRA's
`aplan_forced_compaction_update` is an explicit application call, not a hook.

**How it works:** Native SummarizationToolMiddleware contributes the tool and
uses the supplied engine. The ordinary tool is eligible at approximately half
the automatic trigger (rounded up) and otherwise returns a nothing-to-compact
result. For forced compaction, MIRA converts state messages, normalizes any
existing summary event, applies it, chooses a native retained-tail cutoff,
and rejects empty/no-progress cutoffs. It generates summary text and returns a
ForcedCompactionPlan without immediately editing state or the archive.

The application commit path verifies the checkpoint is still the same completed
checkpoint, appends and verifies the archive, rechecks for concurrent changes,
then calls native `aupdate_state` with only summary event/session ID. It restores
this attempt's archive append on a failed commit where safe. It does not force
the model node, which would accidentally schedule after-model HITL.

**Example:** `/compact` on a completed conversation can summarize an older
prefix even below the compact tool's eligibility threshold. It creates no
new user/assistant turn and does not resume a pending approval.

**Scope:** Main Plan/Act and inherited forks; ordinary isolated children do
not automatically get this compact tool adapter.

**Configuration:** Constructed with the main shared summarizer and no system
prompt addendum; the tool description supplies its usage guidance. Forced mode
bypasses eligibility only; it still uses native retention and requires actual
cutoff progress. No separate summary model or counter is created.

**State and persistence:** The ordinary tool returns native Command updates
including its ToolMessage and summary event/session ID. Forced planning is
read/summary work until the application commits archive and event channels.

**Interaction with other middleware:** It is not a second automatic summary
pass. Both automatic and explicit compaction refer to the same engine/history
path. HITL/visibility still govern calls reaching the ordinary tool.

**If removed:** Automatic compaction remains, but the model's compact tool and
the attached forced-compaction planner disappear.

**Source:** [adapter](../../agent/middleware/compaction.py),
[forced commit](../../core/execution/compaction.py),
[tests](../../tests/test_compaction_native.py).

## InspectableCodeInterpreterMiddleware

**Purpose:** Keep QuickJS Eval behavior native while exposing child-agent
streams and making interrupted thread-mode replay restore JavaScript state.

**Why it exists:** QuickJS's dedicated task bridge omits callbacks needed for
nested model deltas; replayed Eval task IDs can otherwise create duplicate UI
rows, and an interrupted Eval can leave changed JavaScript globals behind.

**When it runs:** Inherited QuickJS before-agent/model-wrapper/after-agent
hooks manage the interpreter lifecycle. MIRA replaces the Eval tool's sync and
async implementation functions, not the generic tool dispatcher.

**How it works:** Native QuickJS adds `eval`, appends JavaScript/tool-reference
instructions, installs selected PTC tools, restores thread snapshots before
execution and saves them afterward. MIRA's wrapper snapshots the REPL before
each thread-mode Eval. It wraps only the dedicated `task` bridge to inherit
callbacks and map task identities to stable logical row IDs derived from Eval
ID, agent type, description, label and occurrence. It forwards those IDs in
tool runtime/config metadata and rewrites matching subagent custom events.
On GraphInterrupt, restore the pre-Eval snapshot with globals re-injected and
re-raise. A rollback failure becomes a RuntimeError. This restores JavaScript
state; it cannot undo external side effects of completed tool calls.

**Example:** Eval starts a child, then reaches an approval interrupt. The UI
sees the real child's streamed response. Replaying the Eval uses the same
logical child row and starts from pre-interrupt JS state instead of accumulating
partially executed global-variable changes.

**Scope:** Main Plan/Act and native forks. Isolated workers do not inherit
QuickJS automatically.

**Configuration:** MIRA passes 64 MiB memory, 5-second timeout, `mode='thread'`,
factory-selected PTC tool names, and the effective dynamic-subagent switch.
QuickJS's inherited default bridge budget is 256 PTC calls per Eval. Dynamic
subagents default off and require both Eval/task enabled. Middleware is still
registered when Eval is hidden; visibility determines the advertised tool.

**State and persistence:** Native QuickJS owns private slot/snapshot/HMAC
channels; the REPL registry is process-local. Prompt additions are transient.
MIRA's stable-ID map is per Eval wrapper setup, not a new durable session table.

**Interaction with other middleware:** ExecutionContext supports nested native
calls. Project error/visibility/approval layers retain their own jobs. QuickJS
prepares PTC from the request it sees before later MIRA filters; factory PTC
selection pre-excludes global unavailable names, but stage-specific behavior
is a separate ordering concern documented in the overview.

**If removed:** Removing the subclass in favor of stock QuickJS retains Eval
but loses callback forwarding, stable child rows and interrupt snapshot rollback.
Removing QuickJS entirely removes Eval, PTC bindings and interpreter persistence.

**Source:** [subclass](../../agent/middleware/code_interpreter.py),
[builder limits](../../agent/middleware/builder.py),
[compatibility/replay tests](../../tests/test_deepagents_compat.py),
[subagent inspection tests](../../tests/test_subagent_inspection.py).

## MiraRubricMiddleware

**Purpose:** Run isolated evidence collection before native structured Rubric
grading and bounded Act revisions.

**Why it exists:** Main-agent claims can be incomplete or stale; grading needs
both transcript evidence and useful independent observations, without giving
the final grader tools or mixing verifier chatter into the main conversation.

**When it runs:** Inherited sync/async before-agent initializes grading runs;
after-agent evaluates an eligible no-tool completion. Local overrides replace
the nested grader invocation, not the native iteration lifecycle.

**How it works:**

1. With no invocation-state `rubric`, native lifecycle hooks do nothing. A new
   rubric or a new attempt after a terminal result resets bookkeeping/run ID.
2. Build a bounded role-labeled transcript: original non-grader user request
   plus recent 30 messages. Selected messages over 4,000 characters are replaced
   in the evaluation copy by references to evaluation-local StateBackend files,
   retaining their complete coerced text, including tool-call facts.
3. Run an evidence-only verifier with permitted verification tool objects and
   optional native HITL. If files were offloaded, add the evidence filesystem
   and read-coverage guard. The verifier is instructed to finish with
   VERIFICATION_COMPLETE, not assign a verdict; the ordinary path has no
   deterministic terminal-token checker.
4. Keep only actual paired verifier AI tool calls and ToolMessages as fresh
   evidence; discard free-form verifier assistant prose as grading evidence.
   Run a separate tool-free final grader using ProviderStrategy(GraderResponse).
   Its prompt treats transcript/tool outputs as untrusted evidence and normally
   prefers fresh observations when they contradict historical current-state
   claims. Transcript evidence alone can still establish a criterion.
5. Native Rubric validates verdict/criterion consistency, freezes the first
   nonempty criterion names for subsequent coverage checks, and may repeat the
   nested evaluation once for an unusable criterion accounting. An unverified
   satisfied verdict is downgraded. `needs_revision` appends a grader-sourced
   HumanMessage and jumps back to the main model until the iteration cap.
6. Publish verification/grading/inspection events. A callback relays raw grader
   output for Inspector; that callback is not another middleware. GraphBubbleUp
   remains native control flow. Ordinary verification/grading failure reaches
   the inherited grader-error lifecycle.

**Example:** The main answer claims a file contains `enabled=true`; verifier
`read_file` observes `false`. Both evidence channels reach the final grader,
which can return needs_revision with that criterion's gap. The main model gets
a revision prompt. If every selected oversized evidence file is not read after
coverage nudges, grading fails rather than silently accepting unread content.

**Scope:** Optional main Act middleware; not main Plan. Native Act forks can
inherit the instance, though normal native child state excludes `rubric`.
Verifier and grader have their own small stacks, not the main stack.

**Configuration:** `system.rubric.enabled=false`, `max_iterations=3` (1–20
in settings). Optional `models.rubric` assignment, else Main. Invocation state
must supply rubric text. Verifier tools require enablement and `rubric` access;
selected dangerous tools retain normal approvals unless always allowed.
Default read-only built-ins have Rubric access; execute defaults disabled with
Rubric access false. Plan/Goal preparation is independent of this setting.

**State and persistence:** Native rubric, run ID, iteration/status/evaluation
and frozen-criteria channels are checkpointed; revision prompts are main graph
messages. Verifier conversations/files/manifests are nested evaluation input,
not added to the parent graph. MIRA consumers project grading results into
session/UI data. The final grader has no filesystem/tool middleware.

**Interaction with other middleware:** _VerifierToolObserver supplies tool
events; VerifierReadCoverageGuard checks retrieved evidence. Repeated-call
termination still runs after-agent Rubric and can therefore lead to a new
revision attempt. No automatic main summarizer/compatibility/ProjectToolError
stack is installed in the plain verifier/grader graphs. Long evidence and
unavailable structured-output provider support can still fail grading.

**If removed:** Act loses this verify-then-grade revision lifecycle. Replacing
it with stock Rubric would retain native grading but lose MIRA's separate
verifier/evidence offload/inspection path.

**Source:** [subclass](../../agent/rubric/middleware.py),
[offload](../../agent/rubric/offload.py),
[tests](../../tests/test_rubric_middleware.py),
[offload tests](../../tests/test_rubric_offload.py),
[inspection tests](../../tests/test_rubric_inspection.py).

## _VerifierToolObserver

**Purpose:** Report real verifier tool execution to the Rubric stream.

**Why it exists:** Tool-call text alone does not prove a verifier actually ran
a tool or show its real result, duration or error status.

**When it runs:** Sync/async verifier tool wrappers.

**How it works:** Emit rubric_tool_start with ID/name/arguments, time the
handler, emit rubric_tool_end with output/error/duration, and return the exact
result. Ordinary exceptions produce an error-end event and are re-raised;
GraphBubbleUp propagates without a fake completion. Event writing failures are
ignored. Run ID/iteration come from the Rubric ContextVar; the nested runtime
writer is preferred, with an outer writer fallback.

**Example:** A real verifier read produces start and end events linked by its
tool ID, with actual returned content and elapsed milliseconds. A planned
read that never executes produces no fabricated successful result.

**Scope:** Both normal and offloaded Rubric verifiers only.

**Configuration:** Always appended when each verifier is lazily constructed;
events require a usable writer/run identity. No user setting.

**State and persistence:** No graph updates or result rewriting. Events may be
recorded by consumers, but this middleware does not save conversation state.

**Interaction with other middleware:** Verifier HITL can interrupt before a
tool executes; events follow real execution. This observer is not a tool-error
recovery wrapper or evidence-completeness validator.

**If removed:** Verification can still run, but loses authoritative tool start/
end events for Rubric presentation and inspection.

**Source:** [implementation](../../agent/rubric/middleware.py),
[tests](../../tests/test_rubric_middleware.py),
[event tests](../../tests/test_rubric_events.py).

## VerifierReadCoverageGuard

**Purpose:** Reject verifier completion until all selected offloaded transcript
source lines have been retrieved through the evidence reader.

**Why it exists:** A verifier could otherwise finish after reading only the
reference, an incomplete page or an unsuccessful tool result.

**When it runs:** Sync/async before-agent resets counters; model wrappers add
pending guidance; after-model clears the pending text when tools are requested;
after-agent checks coverage and can redirect to model.

**How it works:** Derive expected path/line counts from
`offloaded_message_manifest`. Match read_rubric_evidence calls by ID to successful
ToolMessages. Require the native `@@ lines start-end of total @@` header,
matching path, expected total, offset+1, bounded limit and actual body line
count; when source files are supplied, require exact source-body equality.
Union covered lines across pages and compute missing line numbers. If gaps
remain, store a system nudge with those gaps and jump to model. After two
nudges by default, another incomplete completion raises RuntimeError.

**Example:** A 250-line offload read in pages 1–100 and 101–200 still lacks
201–250. Returning VERIFICATION_COMPLETE triggers a nudge to retrieve those
lines. A forged header without matching body or an error result counts for
nothing. The check proves retrieval, not that the LLM understood the contents.

**Scope:** Only the offloaded Rubric verifier, activated when selected transcript
messages exceed the stock per-message text bound.

**Configuration:** `max_nudges=2` in construction; accepts a nonnegative integer
when instantiated directly. No workspace setting exposes it.

**State and persistence:** Verifier-local manifest/files plus private
`_rubric_read_attempts`/`_rubric_read_pending`; not parent state. Pending nudge is
request-only system text. A tool response clears pending text but does not
reset the nudge count for this verifier run.

**Interaction with other middleware:** The renamed native filesystem tool
provides pagination. Coverage failure propagates into Rubric grader_error and
prevents grading that unread evidence. Unlike main Correction, there is no
general response-status protocol; unlike Rubric verdicts, this is a mechanical
source-line retrieval check.

**If removed:** Oversized selected evidence is still available in files, but
the verifier can finish without retrieving it completely.

**Source:** [implementation](../../agent/rubric/offload.py),
[tests](../../tests/test_rubric_offload.py).
