# Testing MIRA through ACP

The full stdio and HTTP clients are MIRA's canonical ACP reference consoles.
They show the friendly interpretation of each public ACP object; use `/raw on`
to display the same object as structured data as well.

Start the stdio console directly:

```text
conda run -n mira --no-capture-output python examples/acp/stdio/full_client.py
```

For HTTP, start the server and client in separate terminals:

```text
conda run -n mira --no-capture-output mira --acp --listen 127.0.0.1:8765
conda run -n mira --no-capture-output python examples/acp/http/full_client.py
```

## Practical scenarios

Exact model wording varies. Prompts involving `execute`, file writes, MCP, or
formal artifacts also depend on workspace settings and enabled tools.

| Scenario | Suggested prompt or action | Expected ACP-visible behavior | Dependency |
| --- | --- | --- | --- |
| Assistant message | `Reply only with PONG` | `MIRA` followed by an `AgentMessageChunk` | None beyond a configured model |
| Reasoning | Ask for a comparison that normally elicits visible reasoning | Separate `THOUGHT` / `AgentThoughtChunk`, if the model emits it | Model-dependent |
| Read file | `Read README.md and report the title` | `TOOL [read]`, completion, then assistant text | `read_file` available |
| Search | `Search for MiraApplication in Python files` | `TOOL [search]` with ACP-provided input and result | Search tool available |
| Execute | `Run python --version` | Permission choices when required, then `TOOL [execute]` and result | Execute enabled and approved |
| Edit/write and diff | `Create acp_probe.txt containing OK` | Edit tool call containing readable ACP diff content | Write tool available; approval policy |
| Approve/reject HITL | Request a protected write, then choose each option on separate runs | Exact selected ACP option ID is returned; invalid input cancels | HITL enabled |
| Tool failure | Ask MIRA to read a definitely missing file | `TOOL RESULT` with `status: failed` and returned error | Tool/model-dependent |
| AskUser option | Ask MIRA to call AskUser with choices `A` and `B` | Generic permission surface with both choices and `Reply in chat` | AskUser tool available |
| AskUser free reply | Select `Reply in chat`, then type the answer at the next `>` prompt | Interrupted turn ends; next text is a new ACP prompt | AskUser tool available |
| ACT / PLAN mode | Run `/mode plan`, `/session`, then `/mode act` | Real ACP mode calls; console reports the active mode | None |
| Formal MIRA Plan | Enter PLAN and request a formal Plan | Ordinary MIRA Plan message, then `Implement`, `Keep`, `Revise in chat` | Planning model/config |
| Formal MIRA Goal | Ask to create a formal Goal | Ordinary MIRA Goal message, then review permission choices | Model-dependent |
| `write_todos` | Ask MIRA to use `write_todos` with several statuses | Distinct `ACP PLAN / write_todos` (`AgentPlanUpdate`) | Planning todos enabled |
| Subagent delegation | Ask MIRA to delegate a bounded investigation | Only ACP-visible underlying tool/message activity; no direct Subagent lifecycle update | Subagents configured; model-dependent |
| MCP approval | Configure an untrusted MCP server and connect | Generic permission options: allow once, always allow, deny | MCP configured |
| MCP tool call | Invoke a configured MCP tool | Generic ACP tool call and completion | MCP connected and tool available |
| Stdio replay | In stdio run `/session`, restart, then `/load <id>` and prompt again | Replayed user/assistant/tool updates, then continued durable session | Saved session exists |
| HTTP live multi-turn | Send two prompts at the same HTTP console `>` | Both prompts reuse one live ACP session | HTTP server running |
| HTTP replay/load | Enter `/load <id>` in HTTP | Console explains that load is replay-only and remains disabled | ACP 0.12.1 limitation |
| Cancellation | `/cancel-after 3 Write a long explanation` | ACP cancel is sent after three seconds; turn reports cancellation | Timing/model-dependent |

## What MIRA currently exposes through ACP

This table describes the existing adapter; the examples do not add or infer
hidden MIRA state.

| MIRA capability | ACP representation | Exposure |
| --- | --- | --- |
| Assistant message | `AgentMessageChunk` | Direct |
| Reasoning | `AgentThoughtChunk` | Direct; model-dependent |
| Informational text | `AgentMessageChunk` | Direct |
| Tool start/result/failure | `ToolCallStart` / `ToolCallProgress` | Direct |
| File edit/write | Tool call with `FileEditToolCallContent` diff | Direct |
| HITL approval | Generic ACP permission request | Direct |
| AskUser | Generic permission options plus `Reply in chat` | Indirect interaction mapping |
| `write_todos` | `AgentPlanUpdate` | Direct |
| Formal Goal/Plan | Ordinary message plus review permission | Indirect specialized lifecycle |
| Durable transcript replay | User, assistant, reasoning, and tool updates | Direct through stdio `session/load`; HTTP replay cannot continue |
| Subagent lifecycle | No `SubagentEvent` projection | Not directly exposed; related tool/message activity may remain visible |
| Rubric lifecycle | No `RubricEvent` projection | Not directly exposed |
| Usage/context | No `UsageEvent` projection | Not directly exposed |
| Compaction | No `CompactionEvent` projection | Not directly exposed |
| Runtime lifecycle | No `RuntimeEvent` projection | Not directly exposed |
| Artifact lifecycle event | No `ArtifactEvent` projection | Specialized Goal/Plan request mapping only |
| MCP lifecycle | No `MCPEvent` projection | Approval and MCP tool calls are exposed through permissions/tools |

MIRA PLAN mode, a formal retained MIRA Plan, and ACP `AgentPlanUpdate` are three
different concepts. PLAN is the session mode. A formal Plan is emitted as an
ordinary MIRA message followed by review choices. Only an actual `write_todos`
tool call produces `AgentPlanUpdate`.
