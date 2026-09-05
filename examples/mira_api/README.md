# MIRA Python API examples

Use these examples when MIRA should run inside your Python process. Install
MIRA, then start with the minimal lifecycle:

```text
python examples/mira_api/minimal_frontend.py
python examples/mira_api/full_frontend.py
```

`minimal_frontend.py` shows the frontend callbacks, application ownership,
session creation, one prompt, and cleanup. `full_frontend.py` adds streamed
events, approvals, AskUser, Goal/Plan review, MCP approval, and a prompt loop.

For an external protocol client instead, use the ACP examples.
