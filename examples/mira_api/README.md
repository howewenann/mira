# MIRA Python API examples

Use the Python API when embedding MIRA in the same Python process. Start with
the minimal lifecycle:

```text
python .mira/examples/api/minimal_frontend.py
python .mira/examples/api/full_frontend.py
```

`minimal_frontend.py` shows frontend callbacks, application ownership, session
creation, one prompt, and cleanup. `full_frontend.py` demonstrates streaming,
approvals, AskUser, Goal and Plan review, MCP approval, and an interactive flow.

If the client will run as a separate process, use ACP instead.
