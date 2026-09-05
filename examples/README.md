# MIRA examples

Examples are grouped by integration boundary and complexity:

| Integration | Minimal | Full |
| --- | --- | --- |
| Direct MIRA Python API frontend | [`mira_api/minimal_frontend.py`](mira_api/minimal_frontend.py) | [`mira_api/full_frontend.py`](mira_api/full_frontend.py) |
| ACP stdio client | [`acp/stdio/minimal_client.py`](acp/stdio/minimal_client.py) | [`acp/stdio/full_client.py`](acp/stdio/full_client.py) |
| ACP Streamable HTTP client | [`acp/http/minimal_client.py`](acp/http/minimal_client.py) | [`acp/http/full_client.py`](acp/http/full_client.py) |

Minimal examples show only the essential connection and one prompt. Full
examples add user-supplied prompts, richer output or interaction handling,
follow-up turns, and the session behavior supported by that integration.

See [`mira_api/README.md`](mira_api/README.md) or
[`acp/README.md`](acp/README.md) for setup and commands.
