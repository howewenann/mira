# DeepAgents and LangChain middleware

This reference covers upstream behavior that MIRA relies on. Read the
[overview](README.md) for the effective stacks and the [MIRA reference](mira.md)
for local changes. Versions are those recorded in the overview, not an
assertion about future dependency releases.

## Contents

- [Inventory](#inventory)
- [FilesystemMiddleware](#filesystemmiddleware)
- [SubAgentMiddleware](#subagentmiddleware)
- [AsyncSubAgentMiddleware](#asyncsubagentmiddleware)
- [_ParentNotifierMiddleware](#_parentnotifiermiddleware)
- [_ForkTaskToolMiddleware](#_forktasktoolmiddleware)
- [PatchToolCallsMiddleware](#patchtoolcallsmiddleware)
- [SkillsMiddleware](#skillsmiddleware)
- [MemoryMiddleware](#memorymiddleware)
- [HumanInTheLoopMiddleware](#humanintheloopmiddleware)
- [UnsupportedContentMiddleware](#unsupportedcontentmiddleware)
- [TodoListMiddleware](#todolistmiddleware)
- [Provider prompt caching](#provider-prompt-caching)
- [_ToolExclusionMiddleware](#_toolexclusionmiddleware)
- [Summarization and its helper](#summarizationmiddleware-and-its-langchain-helper)
- [Bases used through MIRA subclasses](#bases-used-through-mira-subclasses)
- [Model-specific harness middleware](#model-specific-harness-middleware)
- [Source locations](#source-locations)

## Inventory

The first 13 rows are **unmodified upstream implementations reachable through
MIRA's regular construction paths**, including optional todos, remote workers,
forks and profile tool exclusions. The last 14 rows require additional packages
or an exact model harness match. “Yes” means registration can occur;
it does not mean every hook has an effect on every invocation. `M/T` are model/
tool wrappers; `BA/BM/AM/AA` are lifecycle hooks, including async equivalents.

| Middleware | Purpose | Plan | Act | Subagents | Hook(s) | Source file |
| --- | --- | --- | --- | --- | --- | --- |
| FilesystemMiddleware | Backend file tools, permissions, large-content offload | Yes | Yes | Native/precompiled; special evidence instance | M, T; contributes tools | [filesystem.py][fs] |
| SubAgentMiddleware | Synchronous task delegation | If children | If children | Parent dispatcher | M; contributes `task` | [subagents.py][sub] |
| AsyncSubAgentMiddleware | Remote/background task lifecycle | If remote specs | If remote specs | Remote dispatcher | BA, BM, AM, M; contributes tools | [async_subagents.py][async] |
| _ParentNotifierMiddleware | Notify a launching parent about completion/failure | Yes | Yes | Deep-agent server runs | AA, M; meaningful reporting async only | [_parent_notifier.py][parent] |
| _ForkTaskToolMiddleware | Install recursion-guarded task in forks | No direct registration | No direct registration | Native forks only | Tool/schema contribution, no lifecycle hook | [subagents.py][sub] |
| PatchToolCallsMiddleware | Pair dangling valid/invalid calls with error results | Yes | Yes | Native/precompiled | BA (sync implementation) | [patch_tool_calls.py][patch] |
| SkillsMiddleware | Skill index, pinning and skill-tool disclosure | If skills supplied | Same | Declared/inherited skills | BA, BM, M, T | [skills.py][skills] |
| MemoryMiddleware | Load configured instruction files into model prompt | If memory supplied | Same | Native forks; not ordinary isolated | BA, M | [memory.py][memory] |
| HumanInTheLoopMiddleware | Pause for tool approval and apply decisions | If policy present | Same | Native/precompiled, selected verifier tools | AM, T | [human_in_the_loop.py][hitl] |
| UnsupportedContentMiddleware | Replace unsupported model input blocks | Yes | Yes | Native/precompiled | M | [unsupported_content.py][unsupported] |
| TodoListMiddleware | Editable agent todo list | Optional | Optional | MIRA precompiled; forks; matching harness profile | M, AM; contributes `write_todos` | [todo.py][todo] |
| AnthropicPromptCachingMiddleware | Provider cache breakpoints | Lazy wrapper always | Same | Native; not MIRA precompiled | M | [_prompt_caching.py][caching], [provider implementation][anthropic] |
| _ToolExclusionMiddleware | Apply harness-profile tool-name exclusions | If exclusions | Same | Native profile-dependent | M, T | [_tool_exclusion.py][exclusion] |
| BedrockPromptCachingMiddleware | Bedrock request caching | If package installed | Same | Native | M | [insertion logic][caching]; delegate absent locally |
| FireworksPromptCachingMiddleware | Fireworks request caching | If package installed | Same | Native | M | [insertion logic][caching]; delegate absent locally |
| NemotronProgressBudgetMiddleware | Bound model/tool-result/repeat budgets | Exact profile only | Same | Native exact profile | M | [Nemotron profile][nemotron] |
| NemotronPolicyNudgeMiddleware | Prompt tool/action/task-transition follow-through | Exact profile only | Same | Native exact profile | BM, M | [Nemotron profile][nemotron] |
| NemotronToolCallShim | Repair filesystem args and empty results | Exact profile only | Same | Native exact profile | T | [Nemotron profile][nemotron] |
| ReadFileContinuationNoticeMiddleware | Mark potentially incomplete read pages | Exact profile only | Same | Native exact profile | T | [Nemotron profile][nemotron] |
| ToolRetryMiddleware | Retry thrown filesystem-tool failures once | Exact profile only | Same | Native exact profile | T | [tool_retry.py][toolretry] |
| ModelRateLimitRetryMiddleware | Retry provider rate limits | Exact profile only | Same | Native exact profile | M | [Nemotron profile][nemotron] |
| ChatNVIDIAMessageCompatibilityMiddleware | Mirror structured call/name metadata | Exact profile only | Same | Native exact profile | M | [Nemotron profile][nemotron] |
| NemotronReasoningTagCleanupMiddleware | Move think-tag reasoning out of visible text | Exact profile only | Same | Native exact profile | M | [Nemotron profile][nemotron] |
| NemotronTextToolCallParser | Recover available tool calls emitted as text | Exact profile only | Same | Native exact profile | M | [Nemotron profile][nemotron] |
| FollowupDisciplineMiddleware | Repair heuristic follow-up-question omissions | Exact profile only | Same | Native exact profile | AA | [Nemotron profile][nemotron] |
| EntityResolutionGuardMiddleware | Request missing branch/name resolutions | Exact profile only | Same | Native exact profile | BM, AA | [Nemotron profile][nemotron] |
| FinalAnswerGuardMiddleware | Request omitted literals/concrete outcomes | Exact profile only | Same | Native exact profile | AA | [Nemotron profile][nemotron] |

Two more cache implementations, **BedrockPromptCachingMiddleware** and
**FireworksPromptCachingMiddleware**, are conditionally inserted when their
provider packages are installed; neither package was available in the audit
environment. They are listed separately below rather than reported as active.
There are also **12 additional implementations** in the optional Nemotron
harness, described below. Codex profiles reuse TodoListMiddleware, already
counted above.

Together with the **18 MIRA/customized entries**, that gives **31 regular-path
reference entries**, plus **14 additional environment/model-dependent upstream
entries**, or **45 inventory entries across the documented conditional
paths**. This is not a single 45-layer stack. Delegates, internal helpers and
subclass bases are not extra registrations. Arbitrary project-supplied custom
middleware and dynamically registered harness profiles are open extension
points, not a finite built-in inventory.

SummarizationMiddleware is counted as the customized instance in the MIRA
table, with native child instances of the same implementation. The three
upstream bases used through local subclasses are explained below and not
double-counted as independent registrations.

## FilesystemMiddleware

**Purpose and timing:** Contributes filesystem tools, wraps model requests,
and wraps tool results. It gives the agent backend-aware `ls`, `read_file`,
`write_file`, `edit_file`, `delete`, `glob`, `grep`, and capability-dependent
`execute`. Tools enforce filesystem permission rules and route paths through
the supplied backend. Execution requires a compatible backend; unsupported
execute/delete capabilities are filtered from requests. It adapts grep/execute
descriptions to the remaining tool capabilities and can add composite-backend
host-path routing instructions.

**Concrete behavior:** File writes create or replace; edits replace targeted
text and enforce native read-before-edit tracking. Model wrappers move media
results out of tool-result sequences where needed, hydrate stored media, and
offload oversized human content while preserving a recovery pointer. A
ModelInvalidRequestError caused by rejected file content can receive one
fallback call with replacement notices and corresponding message updates.
Tool wrappers can offload large results to backend files and return a preview
plus path instead of filling the context window.

Defaults are 20,000 approximate tokens for tool-result eviction and 50,000 for
human-message eviction; there are tool-specific exceptions, so this is not a
universal output cap. Binary-content handling and blob metadata are also
native behavior. Filesystem commands/messages can update state; backend writes
are separate external effects.

**MIRA scope and interaction:** Main modes and native/MIRA-compiled children.
Plan has write-deny permissions and hidden write tools; Act protects mounted
defaults and permits workspace reads/writes. Execute-backed Act uses the shell
backend instead of those filesystem permission lists. MIRA visibility and HITL
are additional layers. Removing this middleware removes native file tools,
their permission enforcement and their large-content handling.

The Rubric evidence instance is deliberately narrower: StateBackend, read tool
renamed `read_rubric_evidence`, both text eviction thresholds `None`, and local
evidence-reader guidance. It does not expose project files or share the main
filesystem instance. `effective_rubric_tools` also constructs a filesystem
instance merely to extract selected project-reader/execute tool objects; that
helper instance is not itself registered as verifier middleware.

**Example:** A huge project-tool result may become a preview and a backend
`/large_tool_results/...` pointer. The model can page through that file later.
This is result offload, distinct from summarizing conversation history.

**Source/tests:** [upstream implementation][fs],
[MIRA compatibility tests](../../tests/test_deepagents_compat.py),
[Rubric offload tests](../../tests/test_rubric_offload.py).

## SubAgentMiddleware

**Purpose and timing:** Adds the synchronous `task` tool and an optional
model-wrapper prompt fragment. Its constructor prepares native task dispatch;
`create_deep_agent` supplies base middleware/defaults to raw specs before this
step. The default prompt fragment is absent in the audited factory; available
worker names/descriptions are in the task tool schema/description.

**Concrete behavior:** A normal isolated task starts a child conversation from
the assigned task; a fork instead takes parent conversation/state under native
inheritance rules. Compiled runnables are used as supplied. The task call
invokes/streams the chosen graph and returns its final result through the tool
route; permitted child state updates are merged via native Command handling.
Isolated child inputs and child output merges exclude private channels and
rubric. Declarative fork inputs retain private channels except the smaller
native fork-exclusion set; rubric is still excluded.
Per-call response schemas use native declarative construction; already compiled
children reject unsupported schema replacement.

**MIRA scope and interaction:** Main modes whenever synchronous children exist,
normally including configured general-purpose. It is not blanket middleware
inheritance: isolated specs receive their own stack, forks merge the parent's
custom entries, and precompiled children retain their graph. QuickJS's task
bridge calls this native tool. Without it, native synchronous delegation and
the task tool disappear. Permissions/HITL remain properties of the child stack.

**Example:** Parent `task(subagent_type='reviewer', description='Review a.py')`
gets a reviewer result as its ToolMessage rather than importing all of the
reviewer's chat into parent history.

**Source/tests:** [implementation][sub],
[construction tests](../../tests/test_subagent_compilation.py),
[tool allowlist tests](../../tests/test_subagent_tools.py).

## AsyncSubAgentMiddleware

**Purpose and timing:** Only installed when remote `graph_id` specs are supplied.
Adds `start_async_task`, `check_async_task`, `update_async_task`,
`cancel_async_task` and `list_async_tasks`; has before-agent, before-model,
after-model and model wrappers. Async lifecycle hooks poll unfinished tasks;
sync before/after-model hooks deliberately do not poll.

**Concrete behavior:** Tracks remote thread/run/status in `async_tasks`, records
notifications and avoids reporting the same outcome twice. Before an async
model call it can append newly observed outcomes; after a no-tool final answer
a newly observed status can redirect to the model. Follow-ups either interrupt
the remote run or queue, according to the spec's `update_strategy`. A configured
prompt fragment can be appended; MIRA's default construction does not add one.

**Scope/removal:** This manages remote work from a main graph, not the remote
graph's middleware or approvals. Removing it removes this tool surface and
automatic notification/polling updates. No MIRA setting alone creates a remote
agent; an enabled discovered remote spec is needed.

**Example:** A background task changes from running to completed; the next
async poll records the new status and presents its result to the model.

**Source:** [implementation][async],
[MIRA discovery](../../agent/subagents/discovery.py).

## _ParentNotifierMiddleware

**Purpose and timing:** Attached by `create_deep_agent` to every main deep graph.
On async after-agent it reports final success; on async model-wrapper failure
it reports an error and re-raises. GraphInterrupt is re-raised without failure
notification. Sync methods pass through.

**Concrete behavior:** Only reports if invocation config identifies a launching
agent and this is the top-level graph of that task run. Deduplicates run IDs
with a bounded process-local map (1,000 entries); does not edit graph state.
For an ordinary local MIRA run without parent metadata it has no effect.
Removing it loses automatic wake-up notifications when this deep graph is run
as a remote task, not ordinary local model execution.

**Example:** A server-hosted deep agent started by another agent completes and
wakes that parent once. Its nested verifier does not report as another finished
top-level task.

**Source:** [implementation][parent].

## _ForkTaskToolMiddleware

**Purpose and timing:** A native fork-only tool/schema contribution. It has no
before/after/model wrapper hook. DeepAgents inserts it immediately after the
fork's FilesystemMiddleware.

**Concrete behavior:** Supplies the real task tool to a fork and declares
`_deepagents_forked_context` so the native recursion guard can read a tracked
state field. Ordinary isolated workers do not get this contribution.
Without it a fork lacks the task capability/declared fork marker expected by
this native dispatch path. It does not grant unrestricted recursive delegation.

**Example:** A fork carries parent context while task dispatch can recognize
it is already in a fork and enforce native recursion rules.

**Source/tests:** [implementation][sub],
[fork compilation tests](../../tests/test_subagent_compilation.py).

## PatchToolCallsMiddleware

**Purpose and timing:** Repairs incomplete historical tool pairs in
`before_agent`. The installed class implements that sync hook only; the graph
factory's RunnableCallable adapts it for async invocation.

**Concrete behavior:** Collect IDs already answered by ToolMessages. For each
AIMessage valid or invalid tool call with a non-`None` unanswered ID, insert a
matching error ToolMessage immediately after the AIMessage. Invalid calls say
arguments were malformed/truncated; valid calls say no result was recorded and
execution may have been cancelled/interrupted. Return a message-channel
replacement using REMOVE_ALL_MESSAGES plus the patched history. It does not
execute, retry or claim success for the missing call.

**Scope/removal:** Native main and raw/precompiled child stacks. Removing it
leaves provider-invalid dangling pairs after incomplete execution. This startup
repair differs from MIRA's repeated guard, which returns actual errors at the
current call boundary.

**Example:** Historical AI `read_file` call `c1` without a result gets an error
ToolMessage for `c1` before another model request uses that history.

**Source:** [implementation][patch].

## SkillsMiddleware

**Purpose and timing:** Before-agent loads a skill index; before-model handles
explicit pins; model wrappers add the index/instructions and disclose tools
from read skills; tool wrappers resolve previously disclosed skill tools.

**Concrete behavior:** Load metadata when `skills_metadata` is missing/None;
later source paths override same-name skills. Persist metadata/load errors and
surface warnings. A populated list, even empty, is reused until reload requests
None. Pinned skill names cause SKILL.md bodies without frontmatter to be added
as messages and the pin queue cleared. The request gets a system index, not
every skill's full text. After a successful skill read, include-tool resolution
can add model-appropriate tool disclosures to messages or tool schemas and
record `_skill_tools_disclosed`; tool wrappers can then attach the disclosed
tool object for execution.

**Scope/removal:** Main when skills argument is not None (MIRA passes sources),
isolated children when declared, native forks with inherited sources. MIRA's
workflow specialization fills inherited sources when absent. Removing it loses
skill discovery/pinning/disclosure; ordinary file readers alone can still read
a known path but do not create this skill lifecycle.

**Example:** Index says a `review` skill exists; pinning it appends its actual
instructions. Merely listing it does not inject all instruction bodies.

**Interaction:** Runs after MIRA's ContextReport and stage/visibility wrappers,
so later disclosures should not be described as already observed by those
earlier request transforms. MIRA handles first-turn skill refresh after reload
outside the middleware.

**Source/tests:** [implementation][skills],
[MIRA resource tests](../../tests/test_subagent_tools.py).

## MemoryMiddleware

**Purpose and timing:** Before-agent reads configured instruction files into
`memory_contents`; model wrappers append their formatted contents to the
system message.

**Concrete behavior:** Skip loading when the channel already exists; ignore
missing files but raise for other download errors. Store source-path/content
mapping in graph state. On requests, format memory with source context. When
requested by the factory and the actual model is Anthropic, apply an ephemeral
cache breakpoint to the final system block after appending memory.

**Scope/removal:** Main with memory supplied, native forks with inherited
memory; no automatic MemoryMiddleware in isolated child/precompiled stacks.
Removing it stops automatic instruction loading/injection, not filesystem
access to those files. Existing memory state does not itself become a system
prompt without the wrapper.

**Example:** A configured AGENTS.md is read once into state and appears in
subsequent model system prompts without being repeated as user chat messages.

**Source:** [implementation][memory],
[MIRA memory-source selection](../../agent/resources/memories.py).

## HumanInTheLoopMiddleware

**Purpose and timing:** Suspend execution for human review of selected tool
calls. MIRA uses the native default batch mode: after-model creates one
interrupt containing matching actions/configurations. Tool wrappers additionally
apply reviewer edits and annotate results. Upstream supports per-call interrupt
mode, but MIRA's constructors do not select it.

**Concrete behavior:** Resume consumes a decision per action in original order.
Approve keeps the call; edit substitutes the reviewed action and records edit
bookkeeping for execution/result annotation; reject removes it from execution
and supplies a ToolMessage. Wrong decision counts raise ValueError. This uses
LangGraph interruption/checkpoint resume, not an ordinary exception-to-string
retry or a separate MIRA tool runner.

**Scope/configuration:** Main with merged policy, native/MIRA-precompiled
children with their effective policy, and Rubric verifier for selected tools
requiring approval. MIRA supplies approve/edit/reject decisions for enabled
non-read-only tools unless always allowed, filtering by mode/tool policy.
Filesystem-generated ask rules merge with explicit policy, explicit entries
winning. Precompiled/remote external graphs do not inherit approvals afterward.

**State/removal:** Saves edited/rejected call effects and private edit tracking
in graph state. Removing it loses this human approval gate even if tools remain
visible; independent visibility/permissions still apply.

**Example:** Model requests a write; execution pauses. Editing its content
runs the reviewed write and returns an annotated result, rather than silently
executing the original content.

**Source/tests:** [implementation][hitl],
[MIRA compatibility tests](../../tests/test_deepagents_compat.py).

## UnsupportedContentMiddleware

**Purpose and timing:** Late model wrapper replaces input blocks the actual
request model cannot accept with explanatory text.

**Concrete behavior:** For HumanMessage and ToolMessage, consult model.profile
capabilities for images/audio/video and inline PDF files, including separate
image/PDF-in-tool-message flags. Missing flags mean supported; explicit False
rejects. Non-PDF inline file payloads require a supported MIME type and native
ChatOpenAI/AzureChatOpenAI using Responses API; provider-managed URL/file-ID
references pass through that inline check. A replacement identifies the read
path from message metadata when available. Other roles are unchanged.

**State/removal:** Only request copies change; original multimedia stays in
thread state for a later capable model. Removing it can make every subsequent
request fail on retained unsupported content. ModelCompatibility's image
format conversion is a separate earlier operation, not this capability gate.

**Scope/example:** Native main/child and MIRA-precompiled stacks, not plain
Rubric graphs. A text-only model gets “[read_file: ... was not attached ...]”
where an image would have been; switching to a capable model can send it again.

**Source:** [implementation][unsupported].

## TodoListMiddleware

**Purpose and timing:** Adds `write_todos`, a model-wrapper system reminder,
and an after-model check against multiple todo writes in one model response.

**Concrete behavior:** The tool replaces the entire `todos` state list using a
Command. More than one parallel write_todos declaration gets matching error
ToolMessages rather than competing list replacements. It does not create
formal Plans, Goals or Success Criteria and does not judge task completion.

**Scope/configuration:** MIRA explicitly adds it to both main modes when
`system.planning_todos.enabled=true` (default false), and to MIRA-precompiled/
workflow children under that switch. Native isolated workers do not inherit
novel parent middleware. Forks inherit it. A matching upstream Codex harness
can independently add it; “todos are opt-in” describes MIRA's own builder,
not every possible externally selected harness profile.

**State/removal/example:** Request reminder is temporary, list updates persist.
Calling `write_todos` with two new items replaces the prior list; it does not
append two items. Removing it loses that tool and reminder, while formal Plan/
Goal workflows remain distinct.

**Source:** [implementation][todo],
[MIRA compilation tests](../../tests/test_subagent_compilation.py).

## Provider prompt caching

DeepAgents appends `_LazyAnthropicPromptCachingMiddleware`, whose public name
is **AnthropicPromptCachingMiddleware**, to native stacks. Its sync/async model
wrapper passes through non-Anthropic model objects and lazily imports/delegates
to the real provider middleware only when needed. The delegate copies the
system/tool cache breakpoints and request model settings; it does not save a
new conversation or summarize history. Memory appends its own later system
breakpoint when applicable. Without caching middleware, generation remains
possible but those provider caching hints are absent.

**Example:** A native ChatAnthropic request gets ephemeral cache-control hints
on the system/tool prefix and automatic conversation caching settings. Merely
using an Anthropic endpoint through ChatAnyLLM is not sufficient for that
native-model type check.

| Additional middleware | Enablement verified in DeepAgents | Role and limits |
| --- | --- | --- |
| BedrockPromptCachingMiddleware | Successful import of `langchain_aws.middleware.prompt_caching`; constructed with `unsupported_model_behavior='ignore'` | Delegate adds Bedrock-specific request caching behavior for supported models; not installed/audited internally in this environment |
| FireworksPromptCachingMiddleware | Successful import of `langchain_fireworks.middleware.prompt_caching`; same ignore option | Delegate adds Fireworks-specific request caching behavior for supported models; not installed/audited internally here |

The latter delegates are **conditional possibilities**, not confirmed active
MIRA transformations. Their exact provider payload changes require inspecting
the installed provider versions if those packages are added. MIRA precompiled
workers and plain Rubric graphs do not get this tail automatically.

**Source:** [DeepAgents insertion/delegation][caching],
[installed Anthropic delegate source][anthropic].

## _ToolExclusionMiddleware

**Purpose and timing:** Final profile-dependent model/tool wrappers remove
names from a harness profile's excluded_tools and return native errors for
attempted excluded calls. DeepAgents inserts it after other tool contributors.

**Concrete behavior/state:** Request tools are filtered and denied calls get
ToolMessages; no separate persistent policy channel is added. It follows the
selected model's profile, not `.mira/settings.yml` tool policy. Upstream itself
describes it as advertising/execution consistency, not a security boundary.

**Scope/removal/example:** Native main/isolated/fork/fallback-general-purpose
stacks only when the profile has exclusions. A profile excluding `execute`
hides/rejects it even if MIRA otherwise enabled it. Without this wrapper,
profile exclusions can be bypassed by middleware-injected tools. MIRA's earlier
ModelToolVisibility remains a distinct configured restriction.

**Source:** [implementation][exclusion].

## SummarizationMiddleware and its LangChain helper

The registered DeepAgents implementation is
`_DeepAgentsSummarizationMiddleware` with public name **SummarizationMiddleware**.
Its `_lc_helper` is a LangChain SummarizationMiddleware used for summary text,
cutoff policy and token checks. That helper is **not separately registered**:
its ordinary `before_model` message-deletion hook does not run in MIRA's stack.

Native DeepAgents uses model wrappers and `_summarization_event` projection,
preserving the raw conversation log. It archives older messages at
`/conversation_history/<session_id>.md`, adds a summary referring to that path,
retains a suffix without breaking tool-call/result pairs, and can truncate
large old write/edit argument text before full summarization. It accounts for
tools/system input at its boundary and accepts valid reported usage where
applicable. The factory sets no trim budget on the older messages passed to
summary generation (`trim_tokens_to_summarize=None`); summary input is not
automatically limited to the standalone helper's 4,000-token default.

If a normal request hits a recognized provider context limit, native recovery
can summarize and offload oversized trailing tool results before at most one
smaller retry. The full prepared input at that boundary must reduce and fit
the input budget; irreducible input/recovery exhaustion raises
ContextOverflowError. Long system prompts/tool schemas alone can therefore
still fail even after conversation reduction.

Main MIRA instances add archive/replay safeguards and optional summary-model
assignment; isolated children normally use the factory unchanged. The
[MIRA summarization reference](mira.md#summarizationmiddleware-customized-instance)
documents all thresholds and those customizations. The separate compact-tool
adapter calls the same engine, not the LangChain helper's independent graph hook.

**Source/tests:** [DeepAgents implementation][summary],
[LangChain helper][lcsummary],
[MIRA integration tests](../../tests/test_compaction_native.py).

## Bases used through MIRA subclasses

| Upstream base | Native responsibility retained | Registered MIRA implementation |
| --- | --- | --- |
| SummarizationToolMiddleware | `compact_conversation`, half-trigger eligibility, native summary event/tool result; optional model-request guidance is disabled by MIRA's default `None` | [MiraCompactionMiddleware](mira.md#miracompactionmiddleware) |
| langchain_quickjs.CodeInterpreterMiddleware | JavaScript Eval tool; per-thread REPL lifecycle/snapshots, PTC binding and prompt API; optional native task bridge | [InspectableCodeInterpreterMiddleware](mira.md#inspectablecodeinterpretermiddleware) |
| deepagents RubricMiddleware | Rubric attempt reset, after-agent evaluation, criterion coverage retry, status/evaluation channels and revision routing | [MiraRubricMiddleware](mira.md#mirarubricmiddleware) |

These bases are not extra middleware layers installed beside their subclasses.
The native Rubric statuses include `satisfied`, `needs_revision`, `failed`,
`max_iterations_reached` and `grader_error`; reaching a cap is not success.
The first usable criterion enumeration is model-generated, so later frozen
coverage checks do not prove that the initial enumeration captured every
requirement in free-form rubric prose.

**Source:** [summarization/tool source][summary], [Rubric source][rubric],
[QuickJS source][quickjs].

## Model-specific harness middleware

DeepAgents resolves profiles from provider/model identity, including exact
provider-qualified identifiers; merely importing these classes does not enable
them. MIRA's model factory creates ChatAnyLLM. Ordinary unqualified identifiers
under provider `anyllm` do not select native OpenAI/NVIDIA exact profiles.
Custom raw subagent model strings/objects, qualified identifiers, or externally
registered profiles can select them during **native** compilation. MIRA's own
`compile_raw_subagent` does not assemble this harness tail.

The installed built-in profiles adding middleware are:

- Codex keys `openai:gpt-5.1-codex`, `openai:gpt-5.2-codex`,
  `openai:gpt-5.3-codex`: append TodoListMiddleware through the profile slot and
  add model-specific prompt guidance. MIRA's own same-name Todo instance can
  replace that slot; it is not duplicated.
- Nemotron 3 Ultra keys listed below: add the following **12 entries in exactly
  this order** through the profile-extra slot, before Skills/caching/Memory.
  They are upstream behavior, not MIRA-owned guards or default AnyLLM behavior.

| Middleware | Hooks | Actual effect, example, persistence and removal consequence |
| --- | --- | --- |
| NemotronProgressBudgetMiddleware | M | Since the latest external user turn, short-circuit with an AI fallback at 16 AI messages, 48 tool results or 3 consecutive repeated tool calls. Fallback includes available result context/reason metadata and enters graph messages. Removing it loses this model-specific budget; it differs from MIRA's tool-boundary execute/block/recovery phases. |
| NemotronPolicyNudgeMiddleware | BM, M | Before-model adds one-shot named HumanMessages/private flags for long-task transitions, actionable follow-ups, lookup-then-action chains and unproductive filesystem searches after domain-tool work. First-request wrappers can add temporary file/domain-tool preferences. For example, after a lookup for a chained send request, nudge the requested action. Removing it loses those heuristic prompts, not permission checks. |
| NemotronToolCallShim | T | For read/write/edit/delete, rename `path` to `file_path` when missing; add read limit 500 when absent; normalize empty ToolMessage content to `(empty tool result)`. Changes executed args/returned history, not host-path conversion. Without it those native-model quirks reach tools unchanged. |
| ReadFileContinuationNoticeMiddleware | T | Count read source rows from header or numbered format; when at least the requested/default limit was returned, append a notice with next offset. Does not read another page. Result text persists. Without it a 500-line page lacks this explicit continuation hint. |
| ToolRetryMiddleware | T | One immediate retry for exceptions from `ls/read_file/write_file/edit_file/delete/glob/grep`, zero delays/jitter, `on_failure='continue'`. Returns an error after exhaustion; does not retry successful error-status returns as exceptions. Without it those thrown tool failures do not get this extra attempt. |
| ModelRateLimitRetryMiddleware | M | Retry rate-limit errors twice, waiting 4 then 12 seconds: exception class contains `ratelimit`/`rate_limit`, status 429, or text contains `rate limit`/`rate-limit`. No graph counter. Without it those errors propagate on the first attempt. |
| ChatNVIDIAMessageCompatibilityMiddleware | M | Request-copy AI tool calls into OpenAI-style `additional_kwargs.tool_calls` when missing, and ToolMessage name into additional kwargs. No saved request rewrite. Without it an adapter expecting that metadata can omit call/name information. |
| NemotronReasoningTagCleanupMiddleware | M on response | Strip closed `<think>...</think>` text blocks from visible assistant content and move reasoning into missing `additional_kwargs.reasoning_content`; preserve structured response/Command wrapping. Corrected responses enter state. Without it reasoning markup can remain visible content. |
| NemotronTextToolCallParser | M on response | If no structured calls exist, parse supported XML-like function/parameter forms or JSON tool payloads; shell aliases map to execute; accept available names only, assign call IDs and retain leftover prose. Corrected response calls can execute. Without it a text-form tool request remains prose. |
| FollowupDisciplineMiddleware | AA | Once per private fired flag, use regex checks on final answers to nudge redundant/missing follow-up fields (cadence/source/scope/analysis goal/support domain/delivery), append a named HumanMessage and jump to model. Skips loop-risk/exact-answer cases. Without it those heuristically detected questions are accepted unchanged. |
| EntityResolutionGuardMiddleware | BM, AA | Track `get_current_<entity>_id`, relation results and name/title lookups by branch. One pre-answer nudge and one final-answer repair can request missing current-branch/display resolution; append named HumanMessages/private flags. Without it the model can stop with the same unresolved or mixed IDs. |
| FinalAnswerGuardMiddleware | AA | Once per private flag, nudge a final answer omitting mutation argument literals (such as versions) or a vague completion lacking the latest domain-mutation outcome. Append a named HumanMessage and jump to model; skip exact-answer and loop-risk cases. Without it these heuristic omissions pass through. |

These hooks have sync/async forms. Nudge flags are private graph channels, not
MIRA's turn-scoped mode-transition metadata; unlike ModeTransition's notice,
their named HumanMessages can become persisted conversation. They can affect
repetition scans and add independent retry/stop behavior if combined. Their
tests/guarantees are upstream; MIRA's focused tests do not exhaust these profiles.

Exact installed Nemotron profile keys:

```text
NVIDIA:nvidia/nemotron-3-ultra-550b-a55b
nvidia:nvidia/nemotron-3-ultra-550b-a55b
baseten:nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B
fireworks:accounts/fireworks/models/nemotron-3-ultra-nvfp4
fireworks:accounts/fireworks/models/nemotron-3-ultra-bf16
openrouter:nvidia/nemotron-3-ultra-550b-a55b
nebius:nvidia/Nemotron-3-Ultra-550b-a55b
together:nvidia/nemotron-3-ultra-550b-a55b
```

**Source:** [profile resolution][profiles], [Codex profile][codexprofile],
[Nemotron implementations and registration][nemotron], [tool retry][toolretry].

## Source locations

Upstream files are installed in the active environment's `site-packages`, not
vendored in MIRA. Source links below use the verified release-tag commits for
DeepAgents 0.7.32 (`e85c6b9`), LangChain 1.4.7 (`fbb8a35`), QuickJS 0.3.7
(`6c89fe2`) and the optional installed Anthropic 1.7.7 delegate (`6eb169b`).
Optional Bedrock/Fireworks internals were unavailable and are explicitly marked
as such above.

| Package | Installed source paths to inspect |
| --- | --- |
| DeepAgents | `deepagents/graph.py`; `middleware/{filesystem,subagents,async_subagents,_parent_notifier,patch_tool_calls,skills,memory,unsupported_content,_prompt_caching,_tool_exclusion,summarization,rubric}.py`; `profiles/harness/{harness_profiles,_openai_codex,_nvidia_nemotron_3_ultra}.py` |
| LangChain | `langchain/agents/factory.py`; `agents/middleware/{types,todo,human_in_the_loop,summarization,tool_retry}.py` |
| QuickJS | `langchain_quickjs/middleware.py` and its tool/subagent bridges |
| Anthropic | `langchain_anthropic/middleware/prompt_caching.py` |

For example, get the exact installed source without calling a provider:

```powershell
conda run -n mira python -c "import inspect, deepagents.middleware.summarization as s; print(inspect.getfile(s)); print(inspect.getsource(s.create_summarization_middleware))"
```

MIRA construction links: [factory](../../agent/factory.py),
[builder](../../agent/middleware/builder.py),
[subagent compilation](../../agent/subagents/compilation.py),
[Rubric](../../agent/rubric/middleware.py),
[dependency declarations](../../pyproject.toml).

[fs]: https://github.com/langchain-ai/deepagents/blob/e85c6b98087861512d2c5240305e16cfac68a746/libs/deepagents/deepagents/middleware/filesystem.py
[sub]: https://github.com/langchain-ai/deepagents/blob/e85c6b98087861512d2c5240305e16cfac68a746/libs/deepagents/deepagents/middleware/subagents.py
[async]: https://github.com/langchain-ai/deepagents/blob/e85c6b98087861512d2c5240305e16cfac68a746/libs/deepagents/deepagents/middleware/async_subagents.py
[parent]: https://github.com/langchain-ai/deepagents/blob/e85c6b98087861512d2c5240305e16cfac68a746/libs/deepagents/deepagents/middleware/_parent_notifier.py
[patch]: https://github.com/langchain-ai/deepagents/blob/e85c6b98087861512d2c5240305e16cfac68a746/libs/deepagents/deepagents/middleware/patch_tool_calls.py
[skills]: https://github.com/langchain-ai/deepagents/blob/e85c6b98087861512d2c5240305e16cfac68a746/libs/deepagents/deepagents/middleware/skills.py
[memory]: https://github.com/langchain-ai/deepagents/blob/e85c6b98087861512d2c5240305e16cfac68a746/libs/deepagents/deepagents/middleware/memory.py
[unsupported]: https://github.com/langchain-ai/deepagents/blob/e85c6b98087861512d2c5240305e16cfac68a746/libs/deepagents/deepagents/middleware/unsupported_content.py
[caching]: https://github.com/langchain-ai/deepagents/blob/e85c6b98087861512d2c5240305e16cfac68a746/libs/deepagents/deepagents/middleware/_prompt_caching.py
[exclusion]: https://github.com/langchain-ai/deepagents/blob/e85c6b98087861512d2c5240305e16cfac68a746/libs/deepagents/deepagents/middleware/_tool_exclusion.py
[summary]: https://github.com/langchain-ai/deepagents/blob/e85c6b98087861512d2c5240305e16cfac68a746/libs/deepagents/deepagents/middleware/summarization.py
[rubric]: https://github.com/langchain-ai/deepagents/blob/e85c6b98087861512d2c5240305e16cfac68a746/libs/deepagents/deepagents/middleware/rubric.py
[profiles]: https://github.com/langchain-ai/deepagents/blob/e85c6b98087861512d2c5240305e16cfac68a746/libs/deepagents/deepagents/profiles/harness/harness_profiles.py
[codexprofile]: https://github.com/langchain-ai/deepagents/blob/e85c6b98087861512d2c5240305e16cfac68a746/libs/deepagents/deepagents/profiles/harness/_openai_codex.py
[nemotron]: https://github.com/langchain-ai/deepagents/blob/e85c6b98087861512d2c5240305e16cfac68a746/libs/deepagents/deepagents/profiles/harness/_nvidia_nemotron_3_ultra.py
[todo]: https://github.com/langchain-ai/langchain/blob/fbb8a35da7023ac0382c3b22f047337861bd3486/libs/langchain_v1/langchain/agents/middleware/todo.py
[hitl]: https://github.com/langchain-ai/langchain/blob/fbb8a35da7023ac0382c3b22f047337861bd3486/libs/langchain_v1/langchain/agents/middleware/human_in_the_loop.py
[lcsummary]: https://github.com/langchain-ai/langchain/blob/fbb8a35da7023ac0382c3b22f047337861bd3486/libs/langchain_v1/langchain/agents/middleware/summarization.py
[toolretry]: https://github.com/langchain-ai/langchain/blob/fbb8a35da7023ac0382c3b22f047337861bd3486/libs/langchain_v1/langchain/agents/middleware/tool_retry.py
[quickjs]: https://github.com/langchain-ai/deepagents/blob/6c89fe2197a2dfe4f3851cda38565bcadba6066b/libs/partners/quickjs/langchain_quickjs/middleware.py
[anthropic]: https://github.com/langchain-ai/langchain/blob/6eb169b56cea0b535b9af0ca504bb367f6407455/libs/partners/anthropic/langchain_anthropic/middleware/prompt_caching.py
