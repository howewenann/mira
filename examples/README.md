# MIRA examples

Choose the integration boundary first, then grow from the minimal lifecycle to
the full reference implementation:

| Integration | Minimal | Full |
| --- | --- | --- |
| Direct MIRA Python API frontend | [`mira_api/minimal_frontend.py`](mira_api/minimal_frontend.py) | [`mira_api/full_frontend.py`](mira_api/full_frontend.py) |
| ACP stdio client | [`acp/stdio/minimal_client.py`](acp/stdio/minimal_client.py) | [`acp/stdio/full_client.py`](acp/stdio/full_client.py) |
| ACP Streamable HTTP client | [`acp/http/minimal_client.py`](acp/http/minimal_client.py) | [`acp/http/full_client.py`](acp/http/full_client.py) |

```text
Embed MIRA directly                 Integrate through ACP
        |                                    |
        v                                    v
mira_api/minimal_frontend.py        choose stdio or HTTP
        |                                    |
        v                                    v
mira_api/full_frontend.py           minimal_client.py
                                             |
                                             v
                                    full_client.py
```

Minimal examples teach the smallest understandable lifecycle. Full examples
are realistic reference implementations with interaction handling and richer
output while keeping the same public lifecycle visible.

```text
Direct API: Frontend -> MiraApplication.start -> open_session -> prompt
ACP:        Client -> initialize -> new_session -> prompt -> session_update

Client --stdio--> spawned MIRA
Client --HTTP---> running MIRA server
```

See [`mira_api/README.md`](mira_api/README.md) or
[`acp/README.md`](acp/README.md) for setup and commands.
