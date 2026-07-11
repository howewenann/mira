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
discuss how to write a file called test.txt with the content 'hello world'
```

Expected:

- MIRA says it is in planning mode.
- MIRA says write/edit tools are disabled or not allowed.
- MIRA discusses the change without editing `test.txt`.
- MIRA does not write or edit `test.txt`.

Then enter:

```text
give me the plan
```

Expected:

- MIRA shows a structured plan bubble with Implement, Revise, and Discard.
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
