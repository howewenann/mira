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

## Planning Mode Blocks Writes

```powershell
mira
```

Then enter:

```text
/plan
write a file called test.txt with the content 'hello world'
```

Expected:

- MIRA says it is in planning mode.
- MIRA says write/edit tools are disabled or not allowed.
- MIRA produces a plan for creating `test.txt`.
- MIRA saves the clean plan in memory.
- MIRA does not write or edit `test.txt`.

Then enter:

```text
/plans
```

Expected:

- MIRA shows at least one saved plan in a panel.
- MIRA does not repeat the same plan as both a list item and plain text.

Then enter:

```text
/act
write a file called test.txt with the content 'hello world'
```

Expected:

- MIRA leaves planning mode.
- MIRA shows a `write_file` tool call.
- MIRA shows an approval prompt.
- Approving writes `test.txt` with `hello world`.

Use the ask_user tool to ask me which implementation path to take: minimal change, full refactor, or planning only. Do not proceed until I choose.