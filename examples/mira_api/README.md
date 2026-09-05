# MIRA Python API frontend examples

These examples embed MIRA directly through the supported `mira` and `mira.api`
packages. They do not use ACP.

## Minimal frontend

[`minimal_frontend.py`](minimal_frontend.py) implements the two-method frontend
contract, prints assistant text, rejects unsupported blocking interactions, and
sends one fixed prompt:

```text
conda run -n mira --no-capture-output python examples/mira_api/minimal_frontend.py
```

## Full frontend

[`full_frontend.py`](full_frontend.py) is a conservative interactive terminal
frontend. It renders streamed events and handles approvals, AskUser prompts,
Goal/Plan reviews and display, MCP approval, and application confirmations:

```text
conda run -n mira --no-capture-output python examples/mira_api/full_frontend.py "Summarize this project"
```
