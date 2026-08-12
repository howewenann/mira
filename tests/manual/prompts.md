# Manual Test Prompts

Use these prompts for manual smoke testing while developing MIRA.

## TUI Autocomplete And File References

Run the interactive TUI and verify `/he`, `/help`, and `Compare @auth`. Change
`@lang` to `@rec` at the same mention, dismiss a pending file search with
Escape, mouse-select a file, and select a path containing spaces.

Submit multiple references, then a manually edited missing reference:

```text
Compare @README.md with @tests/test_textual_app.py
Use read_file to inspect @README.md
Inspect @does/not/exist.py and report what happens
```

Expected:

- The popup displays at most five matching commands or files.
- Up/Down changes the highlighted completion; Enter or a mouse click inserts
  it without submitting. Escape dismisses the popup.
- Paths containing spaces are inserted as `@"path with spaces"`.
- Changing the query at the same `@` reuses its candidate list; a new or
  dismissed interaction performs fresh discovery and stale results stay hidden.
- Multiple visible references supply exact normalized paths without embedding
  their contents or forcing a particular reader.
- An explicitly named reader is used when available; a missing path fails only
  through that reader's ordinary tool result.

## HITL File Write

```powershell
mira --prompt "write a file called test.txt with the content 'hello world'"
```

Expected:

- MIRA shows a `write_file` tool call.
- MIRA shows an approval prompt.
- Approving the action writes `test.txt` with `hello world`.
- Editing the content before approval updates the existing tool bubble and the
  saved session event to the edited content; it does not leave a second call.

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
- Each group clock starts with its first visible subagent, continues through
  staggered launches, and freezes only after the final row finishes, fails, or
  is cancelled. Concurrent row times are not summed.
- The right task table follows the active group and shows generated subagent
  names, compact inline hints, status, and elapsed time.
- The durable session history contains the eval tool call/result and assistant
  summary, not separate replayed panel rows for each eval-created subagent.

## Dynamic Eval Response Schemas

In `/settings`, enable **Dynamic eval subagents**. First leave its nested **Response
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
- Transient `working...` status blocks disappear; no standalone
  `preparing tool call...` bubble appears.

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

- The existing Plan bubble still shows Implement, Revise, and Close.
- The cancelled unrelated turn does not resolve, discard, or rewrite the plan.

## LM Studio Tool Calling And Reasoning

Use LM Studio with a loaded reasoning-capable model and a `.mira/models.yml`
profile whose `api_base` is usually `http://localhost:1234/v1`. Select it as
Main through `/models`.

```powershell
conda run -n ai_agents python -m cli.main -p "Use a tool to inspect README.md, then answer briefly with the project name."
```

Expected:

- MIRA starts with the model displayed as `lmstudio:<model>`.
- MIRA shows a filesystem search/read tool call such as `read_file`, `glob`,
  or `grep`.
- MIRA answers briefly using information from `README.md`.
- The turn does not fail with an LM Studio native SDK tool-calling error.

## Immediate Ordinary Tool Results

Use a disposable Git workspace so the timing tools and session files can be
removed after the check. Add `.mira/tools/result_timing.py`:

```python
import time

from langchain_core.tools import tool


@tool
def timing_result(label: str, delay_seconds: float = 0.0) -> str:
    """Return a labeled result after a short configurable delay."""
    time.sleep(delay_seconds)
    return f"{label} finished"
```

Enable and always allow only this disposable tool in `.mira/settings.yml`:

```yaml
hitl:
  tools:
    timing_result:
      enabled: true
      always_allow: true
```

Initialize the workspace and launch the current checkout:

```powershell
git init .tmp_tool_results_manual
conda run --no-capture-output -n ai_agents python -m cli.main --workspace .tmp_tool_results_manual
```

### Parallel completion and original-block updates

Enter:

```text
Call timing_result twice in parallel. Start the slow call first with label slow
and delay_seconds 4, then the fast call with label fast and delay_seconds 0.2.
Wait for both and report their completion order.
```

Expected:

- Both ordinary tool-call bubbles appear without waiting for the slow call.
- Although the slow call was requested first, `fast finished` appears while the
  slow call is still running, followed later by `slow finished`.
- Each output updates its own existing bubble by call identity. No result bubble
  is appended at the current bottom or moved to completion-time order.
- The final assistant answer appears only after both results and names the fast
  completion first.
- Any active assistant or thinking bubble remains one continuous bubble when an
  older tool block updates. Waiting/model activity is not cleared by that update.
- Cancelling the turn while the slow call is running leaves no later result,
  orphan watcher, or `Task exception was never retrieved` warning.

Repeat with three calls whose delays are 3, 1, and 2 seconds. Verify every
result attaches to the matching label and appears in completion order without
blocking discovery of later calls.

### One-shot terminal ordering

Run the same two-call prompt in one-shot mode:

```powershell
conda run --no-capture-output -n ai_agents python -m cli.main --workspace .tmp_tool_results_manual -p "Call timing_result twice in parallel. Start slow with delay_seconds 4, then fast with delay_seconds 0.2. Wait for both and report their completion order."
```

Expected:

- `fast finished` prints before `slow finished`, and both print before the final
  answer.
- Streamed assistant or reasoning text is never corrupted or interleaved inside
  a line. A result may wait for the next safe terminal boundary.
- There is one readable output line per tool result and no duplicate recovered
  result at turn end.

### Persistence and replay

After the parallel TUI case finishes, note the session id, close MIRA, and
resume that session.

Expected:

- Each tool bubble still contains exactly one matching result.
- Results remain grouped with their original calls rather than replaying as
  completion-time bubbles.
- Session JSON contains one `tool_result` event per `call_id`; the final-state
  recovery path did not persist a duplicate.

### Native failure and retry regression

When reproducing a provider or ordinary tool that emits a native `tool-error`,
request two failed attempts followed by one successful retry.

Expected:

- Each real attempt has exactly one tool bubble; provisional argument chunks
  do not leave an additional `draft` bubble when the stable call id arrives.
- Both failures remain visible in their own bubbles under a red `error:` label,
  and the successful retry uses the normal `output:` label.
- A handled failure does not add a turn-level error. If the graph also aborts,
  the inline tool error appears before the existing turn-level error report.
- After resuming the session, failed `tool_result` events retain
  `status: "error"` and replay with the same error labels.
- In one-shot mode, failed completions print as `<tool> error: ...` at a safe
  terminal boundary without splitting streamed assistant text.

### Plan and goal isolation

Run these checks in the same build, but treat them as regressions only: ordinary
tool-result timing must not change their event sequence or visible surfaces.

Enter `/plan`, then:

```text
Plan a small improvement to README navigation without editing files yet.
```

Expected:

- The existing actionable Plan bubble appears with Implement, Revise, and
  Close.
- `prepare_plan` and `finalize_plan` use the same stable call/result lifecycle
  as other control tools while retaining the dedicated Plan surface.
- No partial plan content or new plan status block appears.

Choose Revise and enter `Keep the same scope but add an exact verification
command.` Verify one replacement plan appears and the old plan becomes inactive.

With Rubric Middleware either enabled or disabled, enter:

```text
/goal create a small typed slug helper with focused tests
```

Expected:

- The Success Criteria indicator is followed by one actionable Goal bubble
  containing Objective and Success Criteria but no Plan.
- `prepare_goal` and forced `finalize_goal` retain stable call/result identity
  without duplicating the dedicated Goal surface.
- No partial criteria, partial goal, or additional goal status block appears.

Choose Revise and enter `Require Unicode examples.` Verify the existing goal
revision path produces one complete replacement Goal and leaves the old bubble
as inactive history.

Finally, enter `/plan`, then:

```text
Plan a searchable notes index with focused tests. Use SQLite unless repository
inspection proves it incompatible.
```

Expected:

- The normal Plan flow remains approach + Success Criteria, with its existing
  finalized Plan controls, ordering, and revision behavior.
- Formal Plan construction uses `prepare_plan` and `finalize_plan`, never
  `prepare_goal` or `finalize_goal`.

Delete `.tmp_tool_results_manual` after completing the checks. Do not copy its
`.mira/_sessions` or timing tool into the repository.

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
- If a required scope decision cannot be discovered, MIRA calls `ask_user` with concise
  choices instead of asking an open-ended chat question.
- When decision-complete, the visible tool sequence is `prepare_plan`, Success
  Criteria generation, forced `finalize_plan`.
- MIRA shows a Plan bubble with Implement, Revise, and Close.
- Implement, Revise, and Close are compact, borderless one-row buttons that
  match the prompt-panel button treatment.
- Implement receives focus when the plan appears; Left/Right wraps across the
  actions, `i`/`r`/`c` activates the matching action, and Escape returns focus
  to the prompt.
- After clicking the prompt, clicking the active plan body restores the last
  focused action and makes the plan shortcuts active again.
- Closing a Plan returns focus to the prompt without clearing `current_plan`.
- The Plan bubble includes Objective, Context and Constraints, Key Changes,
  Test Plan, Assumptions, and a separate Success Criteria section.
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

- MIRA marks `current_plan` active and switches to Act.
- MIRA shows a `write_file` tool call.
- MIRA shows an approval prompt.
- After implementation, MIRA runs the planned check or names the skipped check
  and explains why it could not be run.
- Approving writes `test.txt` with `hello world`.

Then enter:

```text
/plan-show
```

Expected:

- MIRA renders the exact retained Plan bubble without a model call.
- The current status and concise automatic-evaluation result are muted.
- Asking `Show me the previous plan.` causes `show_plan` to render that same
  exact bubble rather than paraphrasing it.

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

Then choose Close or Implement, then enter:

```text
/plan-show
```

Expected:

- MIRA renders the exact current Plan and Success Criteria.
- Historical replaced bubbles remain inactive transcript items.

## Ask User Prompt Layout

### Autonomous Planning Decisions

Run each prompt in the persistent `/plan` conversation. MIRA should call
`ask_user` before showing required choices in prose. Select the recommended or
first option; the resumed turn should call `prepare_plan`, generate Success
Criteria, and finish through forced `finalize_plan`.

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
same planning thread, the resumed outcome is `finalize_plan`, and no disabled
planning tool is called.

Final broad-goal regression (this exact wording is intentionally test-only):

```text
find a way to make the code base neater
```

Expected: MIRA recognizes that the intended outcome is subjective, calls
`ask_user` before research to choose among distinct directions, then calls
`finalize_plan` after the choice is selected.

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

## Durable Criteria-Only Goal Lifecycle

Use a disposable workspace and retain its session file for replay checks.

1. With Rubric Middleware disabled, run `/goal Write a short professional event
   announcement.` from Act, then again from Plan mode. Expected: both use
   `prepare_goal` -> Success Criteria -> forced `finalize_goal`, keep the original
   mode, and show the same criteria-only GoalBubble with no Plan.
2. Enable rubrics and create the same Goal. Expected: construction is unchanged;
   only the snapshotted automatic-evaluation policy differs.
3. Create a contextual Goal that benefits from repository inspection. Expected:
   only read-only discovery and `ask_user` are available before `prepare_goal`;
   evidence clarifies but does not expand the exact visible objective.
4. Select Revise and provide outcome-changing feedback. Expected:
   `SuccessCriteriaService.revise()` preserves still-valid criteria, may update
   the Objective, and a complete replacement Goal becomes current.
5. Select Close, then run `/goal-show`. Expected: Close retains lifecycle state;
   `/goal-show` renders the exact Goal again with Implement, Revise, and Close.
6. Ask `Show me the current Goal.` Expected: `show_goal` uses the same bubble,
   preserves its tool-call id, produces no duplicate output, and changes no state.
7. Implement a rubric-disabled Goal. Expected: only this explicit attempt gets
   Goal context, no Plan fields are injected, and success completes as
   `agent-declared`; an error or cancellation pauses it.
8. Implement a rubric-enabled Goal. Verify separate rubric bubbles,
   `needs_revision` continuation, `satisfied` -> `rubric-verified`, and resumable
   `max_iterations_reached` via `/goal-resume`.
9. Reopen a completed Goal with `/goal-show` and select Implement. Expected: the
   same Goal starts a new attempt and increments its attempt count.
10. Replace a completed Plan with a Goal and a completed Goal with a Plan.
    Expected: replacement is automatic only after successful presentation.
11. Attempt both replacement directions with incomplete formal work. Expected:
    the structured Replace/Keep choice appears; Keep preserves the old artifact,
    and accepting replacement still preserves it if generation later fails.
12. Reload a session with an exact `current_goal`, then try a retired
    `active_goal` session. Expected: the exact current Goal replays and the
    retired session is not migrated.
13. Check `/plan-show` while a Goal is current and `/goal-show` while a Plan is
    current. Expected: deterministic guidance to the matching command.

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

### Solid scrollbars

1. Make chat history, `/settings`, the prompt panel, and the subagent panel
   overflow vertically by shrinking the terminal or adding enough content.
2. Scroll each panel from top to bottom with the mouse wheel and keyboard.
3. Drag each visible scrollbar thumb and click above and below it.
4. Inspect both ends of every thumb at several positions.

Expected: Windows scrollbars use solid colored cells with no boxed, replacement,
or fractional-block glyphs. Wheel and keyboard scrolling, thumb dragging, and
page-region clicks retain their normal behavior.

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

## Resilient Custom Tools Matrix

Run every scenario from a disposable Git workspace with the current checkout:

```powershell
conda run --no-capture-output -n ai_agents python -m cli.main --workspace <workspace>
```

### Toast behavior

Run these checks alongside the scenarios below:

1. Start with one missing project-tool dependency. Confirm startup produces one
   `Custom tools unavailable` warning containing `Open Issues or run /issues.`,
   the toast is not clickable, and both `Issues 1` and `/issues` open the repair
   screen. Several broken files must still produce only one grouped toast.
2. Leave one failure unresolved and run `/reload` twice. Confirm each explicit
   reload produces `Reload completed` and `1 custom tool file is still
   unavailable.` without adding the warning to chat history.
3. Fix the only failure outside Issues, then wait and continue using MIRA.
   Confirm the tool does not appear automatically and `Issues 1` remains. Run
   `/reload`; confirm there is no recovery toast, `Issues 1` disappears,
   `/tools` shows the recovered tool, and the rebuilt agent can call it.
4. Start with two failures, repair one externally, and run `/reload`. Confirm one
   warning reports one recovered file and one still unavailable, while the
   indicator changes from `Issues 2` to `Issues 1`.
5. Repair all failures with Install All and Reload. Confirm progress remains in
   the modal, the modal closes, the indicator disappears, and no toast appears.
6. Leave a syntax error after Install All and Reload. Confirm the modal stays
   open with refreshed failures and package input, and no toast appears.
7. Add a new broken tool before `/reload`. Confirm one warning reports the new
   failure and current unresolved count, while paths and tracebacks remain only
   in Issues.

### 1. Missing MIRA dependency does not block startup

Create `local_packages/mira_manual_dep/pyproject.toml` for a setuptools project
named `mira-manual-dep`, and add
`mira_manual_dep/__init__.py` containing a `decorate(text)` function. Create
`.mira/tools/manual_mira_tool.py` with a module-scope `import mira_manual_dep`
and a normal LangChain `@tool` named `manual_mira_tool` that calls it.

Expected: startup succeeds, one warning and `Issues 1` appear, and `/tools`
does not expose the tool. Open Issues, confirm the source/import and project-tool
guidance, replace the package input with `./local_packages/mira_manual_dep`, and
choose Install All and Reload. The UI remains responsive, the issue disappears,
`/tools` exposes the tool, and invocation returns `manual:<text>`. Uninstall
`mira-manual-dep` from `ai_agents` afterward.

### 2. Multiple packages plus a syntax error

Create two analogous local packages, `mira_manual_alpha` and
`mira_manual_beta`, and two normal `@tool` files that import them. Add
`.mira/tools/manual_syntax_error.py` containing `def broken(` followed by an
invalid return statement.

Expected: one screen lists all three files and shows project-runtime guidance
once. The install target names MIRA Python and shows its exact interpreter path.
Its input contains both modules, not the syntax error. Replace them with both
local package paths and install once. Both tools recover and appear in `/tools`;
only the syntax file remains and contributes no agent tool.

### 3. Shared dependency is deduplicated

Create two tool files that import the same missing local module.

Expected: `Issues 2` counts files, both files are listed, the package input has
one module name, and one installation/reload recovers both.

### 4. Close and repair later

With one unresolved dependency, open Issues, use the top-right `x`, continue
chatting, reopen from `Issues 1`, use the footer Close, then reopen with
`/issues` and repair it.

Expected: both close controls create no chat bubble or session event, the
indicator remains, all normal chat works, each open reflects current failures,
and repair removes the indicator.

### 5. Broken tools are invisible

Create one successful tool file and one import-failing file. Inspect `/tools`,
then ask the model to call the expected broken tool name. Repair and `/reload`.

Expected: only the successful tool schema is available before repair; the model
cannot call the broken name. The repaired tool appears only after successful
loading.

### 6. Project-runtime tool

Create `project_only_module.py` with `decorate(text)` returning
`project:<text>`. Create `.mira/tools/manual_project_tool.py`:

```python
from mira_tool_api import project_tool

@project_tool
def manual_project_tool(text: str) -> str:
    """Test execution in the configured project environment."""
    from project_only_module import decorate
    return decorate(text)
```

Configure an Execute Environment, `/reload`, inspect `/tools`, and invoke it.
Expected: the public name, Runtime `Project`, and selected environment appear;
the result is `project:<text>`, the internal proxy name never appears, and the
project environment needs neither LangChain nor MIRA. Move the project-only
import to module scope where it is unavailable to MIRA: discovery then fails
and Issues explains the inside-function rule and example path.

### 7. Workspace tool exceptions

Create a normal LangChain `@tool` named `read_file_as_bytes` that raises
`FileNotFoundError(path)`, reload, and ask it to read a missing `@file`.

Expected: the call receives one red `status: error` result with the same call
id, the model explains or repairs the failure in the same turn, and no
turn-level error report or next-turn synthetic cancellation appears.

Then change the project-runtime function above to
`raise RuntimeError("manual project failure")`.

Expected: the tool remains available, invocation becomes a normal tool error
identifying Runtime `Project`, MIRA stays open, and no MIRA package install is
offered.

### 8. Pip failure

Enter an obviously nonexistent requirement in Issues and install.

Expected: captured pip failure details appear in the scrollable body, reload is
not called, the input and footer actions are restored, both close controls work,
and the issue remains.

### 9. Cascading dependency

Create one normal tool file that imports missing local packages A then B.
Install A through Issues, then install B after reload discovers it.

Expected: the same modal refreshes from A to B and the tool appears after the
second install/reload.

### 10. Narrow terminal

Shrink the terminal while a mixed Issues screen is open.

Expected: the body scrolls, both footer action buttons stay together, the
top-right `x` remains visible, Escape closes only while idle, and `/issues`
remains a reliable fallback.

### 11. Issues keyboard navigation

Open Issues with at least one repairable missing dependency. Confirm the
package input receives initial focus; Tab and Shift+Tab traverse enabled
controls; Up and Down wrap through the input and actions; and Left and Right
switch between Install and Close without moving focus out of the package input
while editing. Type `i` and `c` in the input, then focus an action and confirm
`i` starts installation and `c` closes. Confirm Enter submits the input, the
Install label shows `(i)`, the footer Close and top-right `x` both dismiss the
idle modal, disabled controls are skipped, and no close control or shortcut
dismisses the modal while installation is active.

### 12. One-shot mode

Add one successful tool file, one missing-dependency file, and one syntax-error
file, then run:

```powershell
conda run -n ai_agents python -m cli.main --workspace <workspace> -p "use the available custom tool"
```

Expected: one grouped warning names both failed files and deduplicated missing
modules, states normal `@tool` ownership and the project example path, performs
no installation, creates no optional-resource crash report, and continues with
the successful tool available.

## Conversational Durable Plan Lifecycle

Use one disposable Git-protected workspace.

1. Enter `/plan`, then `Explain how current session persistence works.`
   Expected: read-only prose discussion; no forced Plan.
2. Enter `/plan Check the current implementation and design a reliable current
   Plan lifecycle.` Expected: the suffix is recorded as a normal Plan-mode user
   message on the same persistent Plan thread.
3. In Plan and Act modes, submit a request with one genuine preference choice.
   Expected: `ask_user`, never a prose question.
4. Request a final implementation-ready Plan with automatic evaluation disabled.
   Expected: `prepare_plan` -> Success Criteria -> forced `finalize_plan`; the
   bubble shows Plan, Success Criteria, muted disabled policy, and Implement,
   Revise, Close.
5. Repeat an equivalent request with automatic evaluation enabled. Expected:
   the same Plan construction and comparable Plan/criteria content; only muted
   policy text adds the configured iteration cap.
6. Close the Plan, run `/plan-show`, then ask `Show me the previous plan.`
   Expected: both display the exact retained Plan; the latter calls `show_plan`.
7. Revise only the implementation approach. Expected: a complete replacement
   Plan with byte-identical Success Criteria. Revise the required outcome next;
   expected criteria are regenerated to reflect that change.
8. With evaluation disabled, Implement. Expected: `active` then `completed`
   with agent-declared completion. `/plan-resume` rejects the completed Plan.
   Reopen it and Implement again; expected: a new attempt on the same Plan id.
9. With evaluation enabled, exercise `needs_revision`, `satisfied`, and
   `max_iterations_reached`. Expected: separate rubric bubbles; satisfied is
   rubric-verified completion; the maximum state remains resumable.
10. Close an incomplete Plan, exit, resume the session, and run `/plan-show`.
    Expected: exact Plan fields, Success Criteria, status, rubric summary, and
    actions return. Run `/plan-resume`; expected: immediate Act execution.
11. Run `/plan-clear`. Expected: `current_plan` is removed while historical
    Plan and rubric bubbles remain in the transcript.

## Current Persistence Rejection

In a disposable workspace, confirm invalid settings and malformed or conflicting
Plan/Goal sessions are rejected. Raw compaction summary aliases must not persist.

## Unified Plan/Goal Response-Status Protocol

Use current-checkout MIRA with LM Studio model `google/gemma-4-12b` and a
disposable workspace. Enter continuous Plan mode with `/plan`; use `/goal
<prompt>` only for the Goal scenarios. Inspect the saved session and trace when
the scenario requests persistence evidence.

1. With retry cap `2`, ask for a high-level explanation of session storage
   without code changes. Expected: complete prose ending visibly with
   `RESPONSE_STATUS: COMPLETE`, with no forced Plan.
2. Ask Plan mode to inspect session loading, normalization, updates, and saving
   before explaining them. Expected: tool-call responses omit a status, read-only
   calls occur, and the later answer visibly ends with `RESPONSE_STATUS: COMPLETE`.
3. Ask Plan mode to investigate session architecture and propose the smallest
   coherent session-duplication change. Repeat several times. Expected:
   false-progress prose remains immediately before a `Response check` bubble that
   shows the failed check and exact retry prompt; actual research follows, and
   the outcome is `ask_user`, `prepare_plan`, or a complete grounded answer with
   its terminal response status visible.
4. Use a deterministic fake response ending in
   `RESPONSE_STATUS: NEEDS_RESEARCH` without a call, then one with no status.
   Expected: specific then general correction; rejected prose, its exact status,
   and both correction bubbles remain in visible/model history.
5. Ask for a session-duplication design while explicitly leaving full-transcript
   versus metadata-only behavior undecided. Expected: `ask_user`; after choosing,
   request an implementation-ready Plan and verify `prepare_plan` followed by
   the unchanged forced `finalize_plan` bubble.
6. Run `/goal Prepare a durable criteria-only Goal for the same duplication
   outcome.` Expected: Goal research can use only `PREPARE_GOAL`/`prepare_goal`;
   finalization forces only `finalize_goal`. A Plan-only status from a fake response
   is rejected as malformed.
7. Disable rubrics and repeat scenario 3, then enable them and run an existing
   Goal rubric scenario. Expected: protocol behavior is unchanged, no grader is
   called during research recovery, and rubric state/history is unchanged.
8. Ask in Chinese to inspect and explain session loading and saving. Expected:
   Chinese prose remains intact and the exact ASCII status remains visible.
9. After success, inspect session JSON, run `/reload`, and reopen the transcript.
   Expected: the accepted answer and exact status appear once; rejected prose,
   its status, and its technical correction bubble remain paired and ordered in
   session, reload, one-shot, and trace output without appearing as user-authored
   text.
10. Test caps `1` and `2` with a fake model that always ends with
    `RESPONSE_STATUS: NEEDS_RESEARCH` and no call. Expected: respectively two and three
    total model attempts, retained rejected candidates and correction bubbles,
    followed by the explicit incomplete response. Enter
    `0` or `21` in Settings. Expected: the previous normalized value is restored
    and the UI displays the `1`-`20` range.
11. Create a Plan, then ask `show me the plan again` while forcing the model to
    call `finalize_plan`. Expected: `finalize_plan` returns a visible native tool
    error before any interrupt, the error directs the model to `show_plan`, and
    the model can call `show_plan` in the same turn without a MIRA exception or
    pending interrupt. Repeat symmetrically for `finalize_goal`/`show_goal`.
12. Create a Plan after several `ls` and `read_file` calls, then submit `Show,
    reopen, and review the retained Plan.` Expected: Gemma immediately calls
    `show_plan` without new research, prose reproduction, preparation, or
    finalization. The exact Plan bubble returns and no earlier tool call/result
    bubble is recorded again. Repeat with a retained Goal and `show_goal`.
    Test paused, `max_iterations_reached`, and completed Goals; natural-language
    recall and `/goal-show` must both leave Implement, Revise, and Close visible
    and enabled.
13. Trigger one response-status correction and one ordinary warning. Expected:
    the bubble is titled `Response check`, includes `Workflow: Plan` or
    `Workflow: Goal`, and uses the exact system/status blue palette. The warning
    uses the distinct orange palette; neither resembles the user's gold bubble.
14. Inspect `/tools` during Plan research, Goal research, and both finalization
    stages. Expected: the applicable `show_plan` or `show_goal` is listed first
    during research; finalization exposes only `finalize_plan` or
    `finalize_goal`; retired model-facing names are absent.
15. During `plan_finalize`, force malformed `finalize_plan` arguments. Pause the
    graph after its native error is produced. Expected: the original
    `finalize_plan` call bubble already shows the schema error and the session
    JSON already contains its `tool_result` with `status: error`; resuming with a
    corrected call creates a second, separately ordered call. Repeat with
    `goal_finalize` and `finalize_goal`.
16. During `plan_finalize`, force calls to `ask_user`, `read_file`,
    `finalize_goal`, and an unregistered stale name. Expected: every call returns
    a visible native error on its original call bubble, none reaches its handler
    or opens an interrupt, and every error says that only `finalize_plan` is
    permitted. Repeat symmetrically for `goal_finalize`/`finalize_goal`.
17. Replay cumulative root `values` snapshots containing an old error, two new
    errors with distinct call ids, and repeated copies of the final snapshot.
    Expected: the old error is ignored, each new call/error pair appears once in
    graph order before the stream finishes, and final-output fallback adds no
    duplicate. Reload the session and inspect a trace for the same order.
18. Complete a valid Plan and Goal through their finalizers. Expected: each
    retains its dedicated artifact surface and no raw `Interrupt(...)`,
    `Command`, empty completion, or ordinary success result leaks into the
    transcript. Response-status retry counters remain unchanged throughout
    tool-error repair.
19. Run `/goal write me a story of about 20 words. save it to story.md in the
    root directory`. Expected: the Goal Objective is concise and polished while
    retaining the approximate length, filename, Markdown format, and root
    location. The newly proposed Goal immediately shows enabled Implement,
    Revise, and Close actions; `/reload` preserves the polished Objective and
    the same current Goal schema.
20. Create each incomplete source artifact, then request each destination kind:
    Plan -> Plan, Plan -> Goal, Goal -> Plan, and Goal -> Goal. Expected: the
    title and both buttons name the current source kind, while the body names
    both the new and current kinds. Decline preserves the current artifact.
    Accept, then cancel or fail before finalization; expected: the current
    artifact is still preserved. Complete finalization; expected: only then is
    the old artifact superseded.
21. Revise the story Goal first with wording-only feedback, then with `Change
    the deliverable to poem.md and make it 40 words.` Expected: wording-only
    feedback preserves the original outcome and constraints; the explicit
    scope-changing feedback permits the revised Objective and Success Criteria
    to adopt the new filename and length.

Record actual Gemma behavior here after each real run: model/provider, cap,
scenario, tool sequence, retry count, final outcome, and whether any status or
provisional text appeared or persisted.

Observed 2026-08-02 with `lmstudio:google/gemma-4-12b` through the real planning
agent:

- Plan discussion, cap `1`: one no-tool response stated that Plan mode is
  read-only and ended exactly with visible `RESPONSE_STATUS: COMPLETE`. No
  correction retry occurred.
- Goal discussion, cap `2`: one no-tool response answered the Goal-resume
  question and ended exactly with visible `RESPONSE_STATUS: COMPLETE`. No
  correction retry occurred.
- README-navigation scenario, cap `1`: Gemma inspected the requested context
  and called `prepare_plan` with a concrete navigation objective and constraints.
  The tool-call response correctly omitted `RESPONSE_STATUS`. The bare graph
  smoke harness reached the normal control-tool interrupt but did not drive the
  interactive finalization UI, so the resulting Plan bubble was not inspected
  in this run.
- Invalid value `0` against previous cap `5`: the setter restored `5`. The
  focused Textual test verifies the visible `1`-`20` range message and exactly
  one agent rebuild after a valid submission.
- Deterministic graph tests cover paths Gemma did not naturally exercise here:
  caps `1` and `2` exhaustion, specific and general correction prompts,
  cross-workflow statuses, and recoverable `finalize_plan`/`finalize_goal` stage
  errors followed by `show_plan`/`show_goal`.

Observed 2026-08-03 with `lmstudio:google/gemma-4-12b` through the real planning
graph in a disposable workspace:

- Plan recall prompt `Show, reopen, and review the retained Plan` selected
  `show_plan` immediately with `{}` arguments and reached its normal interrupt.
- The symmetric Goal recall prompt selected `show_goal` immediately with `{}`
  arguments and reached its normal interrupt.
- Neither probe called research, preparation, or finalization tools. Both
  expected results printed before the bounded smoke process timed out during
  runtime cleanup; no repository or session artifact was written.

Observed 2026-08-03 with the deterministic live-stream and real LangGraph test
harnesses after completing the unified lifecycle:

- A gated root-values stream paused after a malformed `finalize_plan`, a hidden
  `ask_user`, and a second malformed finalizer with a distinct call id. Each
  matching error was rendered before its gate was released, stayed in arrival
  order, persisted through the live recorder, and was not duplicated by the
  repeated values snapshot or final-output fallback. Its historical baseline
  error was not replayed.
- The finalization integration graph rejected a valid hidden `ask_user` before
  its raising handler could run, returned the original call id and exact
  `finalize_plan` guidance, then accepted the model's corrected finalizer. A
  malformed `finalize_goal` used ToolNode's native schema error and a distinct
  corrected Goal call succeeded. Neither route incremented correction retries.
- Dedicated Plan and Goal control tests completed `prepare_plan`,
  `finalize_plan`, and `finalize_goal` through their normal interrupt surfaces;
  no raw control-stream result replaced those surfaces.
- The separate real-runtime cleanup hang was not changed or retested as part of
  these deterministic lifecycle scenarios.

## DeepAgents 0.7 Upgrade

Use a disposable Git-protected workspace and current-checkout invocation:

```powershell
conda run -n ai_agents python -m cli.main --workspace <workspace>
```

1. Ask a normal non-coding question in action mode. Expected: a concise,
   general-purpose answer with no unsolicited coding workflow or todo list.
2. Enter `/plan` and request a read-only investigation. Expected: MIRA keeps the
   existing Plan intent, can read/search, and cannot write, edit, delete,
   execute, delegate, or evaluate code.
3. Add several arbitrarily named Markdown files under `.mira/memories`, then
   inspect the action and planning runtime. Expected: both receive the same
   resources in their configured deterministic order, without interpreting
   filenames as roles.
4. Inspect `/tools` with Planning todos off, toggle it on in `/settings`,
   inspect again, then toggle it off. Expected: `write_todos` appears exactly
   once only while enabled, both agents rebuild, and no stale entry remains.
5. Enable Planning todos plus schema-free dynamic subagents. Expected: the main
   action agent, planning agent, and compiled general-purpose worker each have
   one todo capability, with no duplicate tool calls.
6. Approve `write_file` for a missing path. Expected: the approval says it will
   create a file and the requested complete content is written.
7. Approve `write_file` for an existing file containing extra lines. Expected:
   the approval says the entire file will be replaced and no old suffix remains.
8. Approve `edit_file` for one exact substring. Expected: the approval describes
   a targeted change and unmatched content remains intact.
9. Request recursive deletion of a test directory in action mode. Expected:
   `/tools` lists `delete` only for a supporting backend; the warning names
   recursive, destructive behavior, reject keeps the tree, and approve removes
   it. With `always_allow` enabled, the same request runs without a dialog.
10. Repeat the delete request in planning mode. Expected: the tool is absent and
    the filesystem permission backstop denies mutation.
11. Run a rubric goal whose final pass reaches its cap, then provoke a grader
    provider error if a safe test endpoint is available. Expected:
    `max_iterations_reached` terminates without an extra review; explanations
    and model/strategy/HTTP diagnostics remain visible.
12. Ask `eval` to compute `1 + 1`, retain a variable across calls, evaluate
    invalid syntax, throw an error, and inspect `typeof process`, `require`, and
    `fetch`. Expected: async results and thread persistence work, errors are
    readable, and ambient host/network APIs are unavailable. Separately run a
    bounded infinite-loop probe and cancel an active eval; both must return
    control without terminating MIRA.

## Tool And Formal-Construction Timing

In the TUI, run a tool whose arguments stream and whose execution lasts several
seconds, then create both a Plan and a Goal.

- The tool uses one bubble throughout: `Preparing · MM:SS elapsed` while a real
  draft is available, `Running · MM:SS elapsed` when execution begins, and
  `Completed in MM:SS` or `Failed after MM:SS` with the final output. The clock
  does not reset, and the normal bottom-right timestamp remains unchanged.
- Cancel a turn while the parent `eval` bubble says `Running`. Its clock freezes
  as `Cancelled after MM:SS`; every eval-subagent row and Group clock freezes at
  the same cancellation boundary. Reopen the session and confirm the parent
  tool still shows the frozen status and duration.
- No standalone `Preparing tool call...` bubble appears during a text-only turn
  or after the final response.
- Success Criteria generation and Plan/Goal creation show temporary spinner
  bubbles with elapsed time; those bubbles disappear when the existing final
  Plan or Goal UI takes over.
- Resume the session and confirm completed tool durations remain visible.

## Rubric Model Profiles And Live Progress

Use a disposable Git-protected workspace and enable rubric grading. Keep the
native Goal Success Criteria visible for comparison with the grader-authored
criterion names.

1. Configure an LM Studio profile for `google/gemma-4-12b` with
   `model_kwargs: {reasoning_effort: none}`, select it as Main, and run a small Goal
   whose result can be verified from the transcript. Expected: the native
   DeepAgents grader completes within the configured generation limit; the TUI
   remains responsive and shows its spinner, `[profile] lmstudio:google/gemma-4-12b`, and
   a once-per-second elapsed clock.
2. Add a second LM Studio profile for `prism-ml/bonsai-27b` with
   `max_tokens: 4096` and `model_kwargs: {reasoning_effort: none}`, select it
   explicitly for Rubric, and
   repeat the Goal. Expected: the action agent remains Gemma, the progress block
   identifies `[profile] lmstudio:prism-ml/bonsai-27b`, and grading completes through the
   same DeepAgents middleware.
3. Inspect the completed bubble. Expected: it replaces live activity with the
   grader identity, duration, `N of N criteria satisfied`, every native
   model-generated criterion name marked `✓` or `✗`, each failed criterion's
   exact gap, and the final explanation/verdict. Names may differ from the Goal
   Success Criteria because the grader authors them.
4. Reload the session and inspect the trace. Expected: the same completed
   identity, duration, criteria, gaps, and verdict reappear. No animation ticks
   are saved; the trace contains only rubric start and completion records.
5. Redirect a one-shot rubric run to a file. Expected: one start block and one
   completion block with no per-second output. Repeat in an interactive terminal;
   the elapsed line updates in place.
6. Stop or provoke a safe provider failure during grading. Expected: no spinner
   continues after interruption, and native grader errors/revision/max-iteration
   behavior remains visible and unchanged.
7. Configure a complete cross-provider rubric profile (provider, model, key, and
   any endpoint/parameters it needs), then repeat. Expected: no main-provider
   credential, endpoint, sampling value, or JSON kwarg leaks into the grader.

Observed 2026-08-04 against LM Studio in a disposable workspace:

- `google/gemma-4-12b` acting and grading with
  `reasoning_effort:"none"` and a 2,048-token generation cap returned `READY`,
  completed grading in 8 seconds, and rendered a clean 1-of-1 native checklist.
  The process took 114 seconds overall because the already-known runtime cleanup
  delay occurred after the visible result.
- `google/gemma-4-12b` acting with `prism-ml/bonsai-27b` grading under the same
  bounded settings selected the dedicated grader correctly and completed in 14
  seconds (28 seconds overall). Bonsai returned a satisfied verdict, but its
  native structured response was malformed: DeepAgents supplied no normalized
  criteria and its explanation retained model tool-syntax fragments. MIRA did
  not reinterpret that provider output or add a custom rubric parser.

## Standards-Compliant MCP OAuth

Use a disposable workspace and manually add Linear's remote endpoint to
`.mira/mcp/mcp.json` as an ordinary HTTP server with no `auth` field. Keep a normal
unauthenticated HTTP MCP and a static-header HTTP MCP available for comparison.

1. Launch the TUI and approve the configured OAuth server. Expected: it becomes
   `Login required`, the filled teal indicator reads `! MCP 0/1`, no browser
   opens, the state does not appear under Issues, and comparison servers behave
   as before.
2. Expand the server and select `Login`. Expected: only its row shows the
   authenticating spinner, the TUI stays responsive, and the browser opens for
   consent. Successful consent changes its labelled badge to Available and the
   top indicator to `✓ MCP 1/1`.
3. Restart MIRA and run `/reload-runtime`. Expected: the stored credential under
   `~/.mira/_state/mcp-tokens/` is reused or refreshed without opening a browser.
   A one-shot invocation also never opens a browser or waits for a callback.
4. Select `Forget login`. Expected: only this server stops, only its token
   directory is removed, `.mira/mcp/mcp.json` is unchanged, and the enabled server
   returns to `Login required`; a disabled server remains disabled.
5. Configure at least two available remote servers, open MCP, and restart one.
   Expected: each server card has a name/transport/status header, three separate
   count cells, and a right-aligned action row. Only the selected card
   transitions; it returns to Available without `CancelledError`, while every
   other server keeps its session, capabilities, and status. The MCP and Issues
   title bars use the same `x` close style as Settings.
6. Restart MIRA with one server that advertises prompts and resources, and one
   that advertises neither. Expected: each server remains Starting until its
   advertised discovery completes. Counts and final health are already settled
   before opening or expanding MCP; unadvertised capabilities show zero and the
   server remains Available without a Needs attention section.
7. Scroll to a lower server and repeatedly expand and collapse it while another
   server is starting or restarting. Expected: no network discovery occurs,
   health, counts, errors, and registries do not change, and the clicked header
   retains focus and the exact scroll position. Selecting Restart then refreshes
   all advertised capabilities for only that server.
8. Configure a static HTTP header as
   `"Authorization": "Bearer ${MCP_TEST_TOKEN}"`, set `MCP_TEST_TOKEN`
   before launching MIRA, and approve the server. Expected: the server receives
   the resolved value while the approval preview, panel, logs, and `mcp.json`
   never show it. Remove the variable and run `/reload-runtime`; only that server becomes
   Failed with an error naming `MCP_TEST_TOKEN` and explaining how to define it.
   Restore the variable in `.env`, run `/reload-runtime`, and confirm the server starts.

## Model Registry, No-Main Startup, And SUBA Autocomplete

Use a disposable workspace with no existing `.mira/`.

1. Launch MIRA. Expected: `.mira/models.yml` and an empty `.mira/prompts/` are
   created; the splash and footer show `unset`; the Models tab shows Main
   `unset` and inherited controls `unset (default)`.
2. Run `/help`, `/tools`, `/subagents`, `/issues`, and `/models`. Expected: all
   local commands work without Main. A normal prompt, Goal, Plan execution, or
   `/compact` instead says `Main model is not configured. Run /models.` without
   creating an error report.
3. Add two valid profiles to `models.yml`, run `/reload`, and select one as Main.
   Expected: the footer and splash show `[profile] provider:model`; null Rubric,
   Summarization, and raw subagents immediately show `<profile> (default)` while
   their stored YAML values remain null. Pin one secondary assignment, change
   Main, and confirm only inherited labels change. While MCP servers are
   connected, change Main, Rubric, Summarization, the context limit, a subagent
   enable toggle, and a raw subagent model. Expected: each change reports
   `settings saved; agents rebuilt`, takes effect immediately, and does not
   disconnect or restart any MCP server. The bright footer button reads
   `model: [profile] provider:model`, stays left-aligned, and truncates with `…`
   only when the terminal cannot fit the complete identity.
4. Add a raw subagent with a long name and `description`, and use a long profile
   name. Expected: Context, the assignment dropdowns, and Subagent Enable share
   one left edge; the Subagents header has no visible `name`; and names and model
   selections remain on one line with `…` instead of wrapping. Enable keeps the
   standard width and a two-cell gap follows each name. Enable the subagent, then
   type `@general-` and a fragment of the new name. Expected: enabled matches
   appear as single-line `SUBA` rows in alphabetical order with ellipsis
   overflow. Selecting general-purpose inserts exactly `general-purpose subagent`.
5. Add nested prompt files whose flattened names collide, plus malformed model,
   MCP, and tool definitions. Expected: colliding prompts are all excluded and
   `/issues` shows one flat, initially collapsed list ordered STARTUP, MODEL,
   MCP, TOOL. Expanded rows show location, details, and guidance; there are no
   package inputs or install controls.
