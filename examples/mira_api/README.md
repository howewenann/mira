# MIRA Python API examples

Use the Python API when embedding MIRA in the same Python process. Start with
the minimal lifecycle:

```text
python .mira/examples/api/minimal_frontend.py
python .mira/examples/api/full_frontend.py
```

`minimal_frontend.py` shows frontend callbacks, application ownership, session
creation, one prompt, and cleanup. `full_frontend.py` demonstrates streaming,
approvals, AskUser, Goal and Plan review, MCP approval and elicitation, and an
interactive flow. Form elicitation accepts a JSON object; it is a teaching
frontend rather than a schema-generated form UI.

If the client will run as a separate process, use ACP instead.

Read the [integration guide](../../docs/frontend-api.md) first, then the
[architecture walkthrough](../../docs/frontend-architecture.md) for execution
flows and ownership. These examples need a configured Main model and may ask
for MCP connection approval. The minimal example deliberately refuses requests;
use the full example when interaction is needed.

In the full frontend, `/inspections` lists saved run summaries and `/inspect ID`
retrieves one live or historical transcript. IDs are announced as capture
starts. Capture does not require that a transcript is displayed.

```text
python .mira/examples/api/full_frontend.py --resume --session SAVED_ID
python .mira/examples/api/full_frontend.py --cancel-after 1 "Inspect the project"
```

Opening saved state does not execute it. `/resume-pending` answers a saved
request; `/continue-incomplete` asks before retrying uncertain work. Cancellation
waits for the turn-start notification, requests cancellation, and joins cleanup.
The full example prints inspection snapshots on selection; a GUI can apply
incremental inspection notifications to an already open view instead.
