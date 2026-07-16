# Manual Test Prompts

Use these prompts for manual smoke testing while developing MIRA.

## HITL File Write

```powershell
mira --prompt "write a file called test.txt with the content 'hello world'"
```

Expected:

- MIRA shows a `write_file` tool call.
- MIRA shows an approval prompt.
- Approving the action writes `test.txt` with `hello world`.

## Subagent Delegation

```powershell
mira --prompt "use 2 subagents. look for the readme file. after that, tell me a joke"
```

Expected:

- MIRA delegates work to two subagents.
- The subagents inspect or locate the README file.
- MIRA finishes with a joke after the README task.

## TUI Subagents Panel

Run the interactive TUI:

```powershell
mira
```

Enter:

```text
Use two subagents in parallel: have one summarize README.md and one inspect pyproject.toml, then compare their findings.
```

Expected:

- A bottom `subagents` panel opens while the subagents run.
- Regular subagents appear as flat task rows with status and elapsed time; no
  group labels or ids are shown.
- Rows use generated subagent names plus compact inline task hints.
- `TASK`, `STATUS`, and `TIME` remain fixed and aligned when the terminal is
  resized; task text stays on one line and visible truncation ends in `...`.
- While work is running, the close control is hidden. Collapsing the panel keeps
  an animated summary visible, and starting another subagent reopens the panel.
- After all rows finish, the close control returns and dismisses the panel.
- After completion, the panel remains visible. Submitting the next prompt
  collapses it to the header summary.
- Closing the panel hides it without deleting the just-finished rows; starting
  another subagent workflow resets and reopens it.
- Restarting or reopening the session shows durable transcript subagent blocks,
  not the old live panel.

## TUI Dynamic Eval Subagent Groups

In `/settings`, enable dynamic subagents, then enter:

```text
Use eval to generate 8 haikus about breakfast food, then run a small tournament with subagent judges to pick the best one.
```

Expected:

- A bottom `dynamic subagents` panel opens while eval-created subagents run.
- The left list shows `Group 1`, `Group 2`, and so on for eval batches; raw
  eval ids are not displayed.
- The right task table follows the active group and shows generated subagent
  names, compact inline hints, status, and elapsed time.
- The durable session history contains the eval tool call/result and assistant
  summary, not separate replayed panel rows for each eval-created subagent.

## Dynamic Eval Response Schemas

In `/settings`, enable **Dynamic subagents**. First leave its nested **Response
schemas** setting enabled and enter:

```text
Use eval to ask a general-purpose subagent to judge which is better, "quiet pond" or "bright market". Require a responseSchema with string fields winner and reason, then return the result.
```

Expected:

- Eval may dispatch the structured request using the model/provider's normal
  structured-output behavior.
- Existing behavior is unchanged when Response schemas is `yes`.

Then set **Response schemas** to `no` and repeat the same prompt.

Expected:

- DeepAgents reports that `response_schema` cannot be used with the compiled
  `general-purpose` subagent.
- No child model starts for the rejected schema-bearing dispatch.
- MIRA remains responsive and does not enter a child todo or generation loop.

With Response schemas still set to `no`, enter:

```text
Use eval to ask a general-purpose subagent to inspect the workspace and judge which is better, "quiet pond" or "bright market". Do not pass responseSchema. Return its text answer.
```

Expected:

- The compiled full subagent starts normally.
- It can use todos, filesystem tools, project tools, and skills available to
  the current agent.
- Eval receives and returns the final text answer.

## Cancelled TUI Bubble Boundaries

Run the interactive TUI:

```powershell
mira
```

Enter this reasoning-heavy prompt, cancel the turn while `thinking` is still
streaming, then submit `continue`:

```text
Think out loud, then write three different short stories about a dog chasing a cat; use subagents in parallel and judge which is funniest.
```

Expected:

- The cancelled turn keeps its partial `thinking` block as history.
- The next turn starts a new `thinking` block instead of appending to the old
  one.
- Any running subagent blocks become `CANCELLED` and stop animating.
- Transient `working...` or `preparing tool call...` status blocks disappear.

Enter this tool/delegation prompt, cancel while tool or task setup is visible,
then submit `continue`:

```text
Search this repo for cancellation handling and summarize every file involved. Use subagents if helpful.
```

Expected:

- Incomplete tool-call or task-draft bubbles from the cancelled turn are not
  reused by the next turn.
- New tool, task, reasoning, or assistant output appears in fresh bubbles.

Then verify active plans survive unrelated cancellation:

```text
/plan
Plan a small change to improve transcript rendering after interrupted turns.
```

After a plan bubble appears, enter this separate prompt and cancel it while it
is running:

```text
Now do a separate long reasoning task about how cancellation should work in terminal UIs.
```

Expected:

- The existing plan bubble still shows Implement, Revise, and Discard.
- The cancelled unrelated turn does not resolve, discard, or rewrite the plan.

## LM Studio Tool Calling And Reasoning

Use LM Studio with a loaded reasoning-capable model and the OpenAI-compatible
server enabled at `MIRA_LLM_BASE_URL`, usually `http://localhost:1234/v1`.

```powershell
conda run -n ai_agents python -m cli.main -p "Use a tool to inspect README.md, then answer briefly with the project name."
```

Expected:

- MIRA starts with the model displayed as `lmstudio:<model>`.
- MIRA shows a filesystem search/read tool call such as `read_file`, `glob`,
  or `grep`.
- MIRA answers briefly using information from `README.md`.
- The turn does not fail with an LM Studio native SDK tool-calling error.

## One-Shot Markdown File Prompt

```powershell
conda run -n ai_agents python -m cli.main -f tests/manual/file_prompt.md
```

Expected:

- MIRA reads the Markdown file as the one-shot prompt.
- MIRA inspects `README.md`.
- MIRA answers with exactly two bullet points.
- Running `conda run -n ai_agents python -m cli.main -p tests/manual/file_prompt.md`
  treats the path as literal prompt text, not as a file to read.

## Neat And Tidy Architecture Audit

Run this periodically when the repo starts feeling messy or before starting a
cleanup pass:

```powershell
conda run -n ai_agents python -m cli.main -f tests/manual/neat_tidy_audit_prompt.md
```

Expected:

- MIRA does not edit files.
- MIRA reviews the full repo for architectural neatness and maintainability.
- Findings are grouped by module or functional area, such as CLI/startup,
  runtime streams, TUI, sessions, resources, diagnostics, and tests.
- Each group includes risk, likely files/symbols, refactor difficulty, testing
  scope, and a smallest safe cleanup target.
- MIRA ends by asking which group to target first instead of starting a broad
  refactor.

Then run an interactive reasoning check:

```powershell
mira
```

Enter:

```text
Think briefly about whether README.md describes MIRA as educational, inspect README.md if needed, then answer yes or no.
```

Expected:

- A `thinking` block appears if the loaded model emits reasoning through LM
  Studio's OpenAI-compatible endpoint.
- MIRA can still use read-only tools.
- If no `thinking` block appears but tool calling works, record the model name
  and LM Studio version; that model/server path is not emitting reasoning
  fields through the OpenAI-compatible endpoint.

## Planning Mode Blocks Writes

```powershell
mira
```

Then enter:

```text
/plan
explain how planning mode prevents project mutations
```

Expected:

- MIRA says it is in planning mode.
- MIRA reports `write_file`, `edit_file`, `execute`, `task`, and `eval` as
  disabled.
- MIRA answers normally without creating a plan bubble or asking a follow-up.

Then enter:

```text
find all dead code for refactoring
```

Expected:

- MIRA inspects the repository without using a disabled planning tool.
- MIRA's reasoning classifies the request as implementation intent before it
  begins repository research.
- If a material scope decision is needed, MIRA calls `ask_user` with concise
  choices instead of asking an open-ended chat question.
- MIRA shows a structured plan bubble with Implement, Revise, and Discard.
- Implement, Revise, and Discard are compact, borderless one-row buttons that
  match the prompt-panel button treatment.
- Implement receives focus when the plan appears; Left/Right wraps across the
  actions, `i`/`r`/`d` activates the matching action, and Escape returns focus
  to the prompt.
- After clicking the prompt, clicking the active plan body restores the last
  focused action and makes the plan shortcuts active again.
- Discarding a plan returns focus to the prompt immediately.
- The plan bubble includes Summary, Key Changes, Test Plan, and Assumptions.
- The Test Plan names an exact command/check to run and an expected result.
- MIRA does not write or edit `test.txt` until Implement is chosen.

Then choose Revise on the plan bubble and enter:

```text
include a testing plan
```

Expected:

- MIRA opens a focused Revise Plan prompt before resolving the current plan.
- MIRA shows a visible `Revise plan: include a testing plan` turn.
- MIRA understands the feedback refers to the previous plan.
- MIRA presents a replacement plan bubble and the old plan is inactive history.

Then choose Implement on the plan bubble.

Expected:

- MIRA resolves the plan bubble as approved for implementation.
- MIRA shows a `write_file` tool call.
- MIRA shows an approval prompt.
- After implementation, MIRA runs the planned check or names the skipped check
  and explains why it could not be run.
- Approving writes `test.txt` with `hello world`.

Then enter:

```text
/plan
show me the previous plan
```

Expected:

- MIRA stays in planning mode.
- MIRA recalls the previously saved structured plan from session context.
- MIRA does not say there is no previous plan.
- MIRA does not recreate an active plan bubble unless it is explicitly proposing a new/revised plan.

## Structured Plan Recall

```powershell
mira
```

Then enter:

```text
/plan
can you write a simple palindrome function to a file in the root directory
```

Expected:

- MIRA presents a structured palindrome plan bubble.

Then choose Revise and enter:

```text
add docstring and typing hints
```

Expected:

- MIRA presents a revised palindrome plan that keeps the original task context.

Then choose Discard or Implement, then enter:

```text
/plan
show me the previous palindrome plan
```

Expected:

- MIRA recalls the saved palindrome plan, including Summary, Key Changes, Test Plan, and Assumptions.
- MIRA includes the plan status such as discarded, revision requested, or approved for implementation.

## Ask User Prompt Layout

### Autonomous Planning Decisions

Run each prompt in a fresh `/plan` thread. MIRA should call `ask_user` before
showing alternatives in prose. Select the recommended or first option; the
resumed turn should finish with `present_plan`.

1. `Plan making the codebase neater. The work can focus on runtime architecture, code-quality standardization, or UI cleanup; none has been selected.`
2. `Plan replacing session storage. JSON Lines and SQLite are both acceptable, and the persistence tradeoff has not been decided.`
3. `Plan adding authentication to the API. API keys and OAuth are both viable, and the intended client type is not established.`
4. `Plan renaming the public CLI flags. We have not decided whether backward-compatible aliases are required.`
5. `Plan migrating persisted settings to a new schema. The acceptable choice between automatic migration and explicit user migration is unresolved.`
6. `Plan changing the runtime event API. We have not decided whether compatibility with third-party consumers outweighs a cleaner breaking design.`
7. `Plan redesigning plan-bubble shortcuts. Automatic focus and modifier-based global shortcuts are both viable, and the desired interaction has not been chosen.`
8. `Plan adding diagnostics telemetry. Whether collection is disabled, opt-in, or enabled by default is a product decision.`
9. `Plan adding a cache. An external dependency and a small built-in implementation have different maintenance tradeoffs, and no preference is established.`
10. `Plan parallelizing repository analysis. Threads, processes, and asyncio have materially different constraints, and the workload assumptions are unknown.`
11. `Plan changing API error responses. A clean new envelope conflicts with preserving the current wire format.`
12. `Plan supporting multiple Python versions. The minimum supported version and willingness to use newer language features have not been decided.`

For every case verify that the initial tool call is `ask_user`, its question
does not enumerate its 1-3 concise choices, the selected answer remains in the
same planning thread, the resumed outcome is `present_plan`, and no disabled
planning tool is called.

Final broad-goal regression (this exact wording is intentionally test-only):

```text
find a way to make the code base neater
```

Expected: MIRA recognizes that the intended outcome is subjective, calls
`ask_user` before research to choose among distinct directions, then calls
`present_plan` after the choice is selected.

```text
Use the ask_user tool to ask me which implementation path to take. Use exactly these options: minimal change (Recommended), focused refactor, planning only. Put only the question in the question field and only the answers in options.
```

Expected:

- The prompt panel shows the question once.
- Three concrete options plus `Tell MIRA what to do differently` fit vertically
  without scrolling.
- The recommended option is visible as `(Recommended)`.

```text
Use the ask_user tool to ask me to choose between 10 numbered test targets. Do not proceed until I choose.
```

Expected:

- The choices remain accessible with a scrollbar.
- The TUI does not overflow, hide the fallback, or crash.

```text
Use ask_user to give me 10 unique lunch options.
```

Expected:

- MIRA calls `ask_user`.
- The prompt shows 10 lunch choices plus `Tell MIRA what to do differently`.
- The choices remain accessible with a scrollbar.

```text
Use the ask_user tool with three deliberately long option labels about testing database initialization, email ingestion, and processing/extraction.
```

Expected:

- The choice buttons are equal-width and vertical.
- Long labels truncate cleanly.
- `Tell MIRA what to do differently` remains visible.

## One-Shot Implementation Runs Planned Checks

Use a disposable workspace with `execute` enabled and always-allowed in that
workspace's `.mira/settings.yml`.

```powershell
conda run -n ai_agents python -m cli.main --workspace .tmp_plan_followthrough_manual -p "Create hello_check.py that defines greet(name) returning 'hello, ' plus the name. After creating it, run python -m py_compile hello_check.py. In your final answer, report whether the check ran."
```

Expected:

- One-shot output shows a `write_file` tool call for `hello_check.py`.
- One-shot output shows an `execute` tool call for
  `python -m py_compile hello_check.py`.
- The final answer reports that the check ran successfully.
- If the check cannot run, the final answer names
  `python -m py_compile hello_check.py` and explains why it was skipped or
  failed.

## Execute Virtual Workspace Paths

Use a disposable workspace. In the TUI, enable `execute` from `/settings`
before running these checks.

```powershell
mira --workspace .tmp_execute_manual
```

Then enter:

```text
write a Python file at /tmp.py that prints "mira execute path ok", then run it
```

Expected:

- MIRA writes the file using the virtual file-tool path `/tmp.py`.
- MIRA shows an `execute` approval prompt.
- The proposed shell command runs the workspace file as `python tmp.py`,
  `python .\tmp.py`, or an equivalent workspace-relative command.
- The proposed shell command does not run `python /tmp.py`.
- Approving the command prints `mira execute path ok`.

Then try the one-shot surface in a disposable Git-initialized workspace with
`execute` already enabled in that workspace's `.mira/settings.yml`:

```powershell
conda run -n ai_agents python -m cli.main --workspace .tmp_execute_manual -p "Create a Python file at /tmp.py that prints exactly EXECUTE_PATH_OK, then run it with execute and report the output."
```

Expected:

- One-shot output shows the write and execute flow.
- The `execute` command uses a workspace-relative script path, not `/tmp.py`.
- The `execute` command uses `python tmp.py`, `python .\tmp.py`, or an
  equivalent workspace-relative command.
- The final output includes `EXECUTE_PATH_OK`.

## Execute Nested Workspace Paths

Use a disposable workspace with `execute` enabled.

```powershell
mira --workspace .tmp_execute_manual
```

Then enter:

```text
create /scripts/check_path.py that prints "nested path ok", then run it
```

Expected:

- MIRA writes `/scripts/check_path.py`.
- The `execute` command uses `python scripts/check_path.py`,
  `python .\scripts\check_path.py`, or an equivalent workspace-relative path.
- The `execute` command does not use `python /scripts/check_path.py`.
- Approving the command prints `nested path ok`.

## Manual Context Compaction

Use a disposable workspace and start the TUI:

```powershell
conda run -n ai_agents python -m cli.main --workspace .tmp_compact_manual
```

Build a conversation with several substantial prompts and replies, then enter:

```text
/compact
```

Expected:

- MIRA shows a compaction status without displaying a model-made
  `compact_conversation` tool call.
- If older messages exceed DeepAgents' retention window, the status finishes as
  `context compacted` and the saved session gains a compaction event.
- If the conversation is already within the retention window, the status
  finishes as `nothing to compact`.
- `/session` reports the same turn count as before `/compact`.
- A subsequent topic-switch prompt starts a normal turn and retains relevant
  information from the generated summary.
- Summary-model reasoning and generated summary text never appear as reasoning
  or assistant bubbles while compaction is running.

Afterward, enter a normal prompt that explicitly asks MIRA to discuss the words
"compact conversation" and "summarize" without invoking compaction.

Expected:

- The ordinary reasoning and reply remain visible; wording alone does not make
  MIRA classify the model call as compaction.

## Goal-Driven Rubric Grading

Use a disposable workspace and keep its session files for replay checks. Use
mock models for deterministic criteria revision and iteration-cap behavior, and
also run scenario 2 once with a real locally configured MIRA model where
practical.

### 1. Disabled compatibility

Leave Rubric Middleware disabled in `/settings`, then enter:

```text
/goal add a palindrome helper
```

Expected: MIRA directs you to Rubric Middleware in `/settings`; no model call or
proposal event is created. Enter `/plan` and plan the same task. The legacy plan
bubble and single planning flow remain unchanged, with no Definition of Done or
visible `prepare_goal` control.

### 2. Goal proposal and implementation

Enable Rubric Middleware, leave maximum iterations at 3, then enter:

```text
/goal create palindrome.py with a typed is_palindrome function and focused tests
```

Expected: a goal bubble separately shows the original objective, Markdown
Definition of Done, and `Rubric iterations: 3`. Choose Implement. The normal
action agent receives the effective objective plus rubric criteria, rubric
activity updates one TUI block in place, passes are numbered from 1, and no
`rubric_grader` tool appears. While criteria are being generated, an animated
`drafting Definition of Done...` block remains visible.

### 3. Goal criteria-only revision

Create another `/goal`, choose Revise, and enter:

```text
Make the plan shorter.
```

Expected with the deterministic mock: criteria are returned exactly unchanged,
the previous plan is absent from the criteria-revision request, and the real
feedback is recorded as a user event. Revise again with `Require Unicode
examples in the tests.` Only the Definition of Done changes. Discard the
proposal and verify prompt focus is restored.

### 4. Rubric-enabled planning order

Enter `/plan`, then:

```text
Plan a searchable notes index. SQLite and JSON are both acceptable, but choose
with me before finalizing the design.
```

Expected: the planning agent uses `ask_user`; the answer remains structured,
read-only research completes, the hidden `prepare_goal` interrupt generates
criteria, and the same planning thread then calls `present_plan`. Before
`prepare_goal`, `/tools` and diagnostics should not show `present_plan` in
the model-visible research surface. After the resume, `present_plan` is the only
model-visible tool and a tool call is required; `prepare_goal`, `ask_user`, and
research tools cannot be selected. The TUI changes its animated status from
`drafting Definition of Done...` to `drafting plan...` across the two model
calls. The bubble shows normal plan sections followed by the
Definition of Done. Session JSON keeps original objective, resolved decision,
effective objective, criteria, and plan as separate proposal fields.

### 5. Plan revision and approval

Choose Revise on the rubric-enabled plan and enter:

```text
Keep the same outcomes, but make the plan shorter.
```

Expected: criteria revision completes first and may return criteria unchanged;
plan revision then receives identical feedback, original objective, previous
plan, and revised criteria, and returns a complete plan. Choose Implement. The
action context contains the approved plan while invocation state contains the
separate criteria.

### 6. Cap reconciliation, ordinary clearing, and replay

Set maximum iterations to 1. With a deterministic mock grader, implement a goal
whose final event says `needs_revision` while checkpoint state says
`max_iterations_reached`.

Expected: TUI, one-shot/trace output, `TurnResult`, and the durable rubric event
finish as `max_iterations_reached`, showing failed criteria and gaps. Reopen the
session: completed proposal and rubric blocks replay, start activity does not,
and no synthetic grader feedback appears as user chat. Submit an ordinary
action prompt while rubric remains enabled; MIRA sends explicit `rubric=None`
and performs no grading. Changing either rubric setting while a proposal is
pending resolves it as `settings changed` and rebuilds both agents.

## Windows TUI Keyboard And Copy Matrix

Run the current checkout from a disposable Git workspace with:

```powershell
conda run --no-capture-output -n ai_agents python -m cli.main --workspace <workspace>
```

Repeat the checks in each terminal host with both `cmd.exe` and Windows
PowerShell where available:

- Classic Windows Console Host (`conhost.exe`)
- Windows Terminal
- VS Code integrated terminal

Record the terminal host separately from the shell, along with its version and
the active Textual driver.

### Enter and Shift+Enter

1. Type `line one`, press left Shift+Enter, and confirm the prompt is not
   submitted.
2. Type `line two` and confirm the prompt contains exactly two lines.
3. Press ordinary Enter and confirm the complete prompt submits exactly once.
4. Submit a separate one-line prompt with Enter.
5. Repeat the multiline check with right Shift+Enter.
6. Confirm Ctrl+Enter has no MIRA-specific newline behavior.

Expected: Enter always submits, each Shift+Enter inserts one newline and never
submits, and both physical Shift keys behave identically. Classic Console Host
should use MIRA's raw `VK_RETURN`/`SHIFT_PRESSED` normalization; an already
encoded VT Shift+Enter sequence should remain unchanged.

### Ctrl+C selections

1. Generate multiline user and assistant bubbles.
2. With PromptBox still focused, select part of each bubble, press Ctrl+C, and
   paste into an external editor.
3. Click a bubble so PromptBox loses focus, select bubble text, and repeat.
4. Repeat with the chat container focused and with no widget focused.
5. Select across multiple rendered chat widgets and verify Textual's combined
   selected text is copied in display order.
6. Select prompt text with no chat selection and verify prompt copying.
7. Leave both a prompt selection and a chat selection; verify chat text wins.
8. Press Ctrl+C with no selection while idle and during an active turn.
9. Verify Ctrl+X and Ctrl+V still cut and paste prompt text.

Expected: the exact selected text appears in the external editor, each command
performs one clipboard write, no terminal-native selection shortcut is needed,
and Ctrl+C with no selection does not change the prompt, cancel work, or quit.
