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
- MIRA does not write or edit `test.txt` until Implement is chosen.

Then choose Implement on the plan bubble.

Expected:

- MIRA resolves the plan bubble as approved for implementation.
- MIRA shows a `write_file` tool call.
- MIRA shows an approval prompt.
- Approving writes `test.txt` with `hello world`.

Use the ask_user tool to ask me which implementation path to take: minimal change, full refactor, or planning only. Do not proceed until I choose.
