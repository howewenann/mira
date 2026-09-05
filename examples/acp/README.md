# ACP client examples

ACP clients are the frontend side of an ACP connection. The stdio and HTTP
examples use only public APIs from the stock `agent-client-protocol` SDK.

## Stdio

Install `mira[acp]`. The client launches `python -m cli.main --acp` as its child
process, so no MIRA server is started separately.

```text
python examples/acp/stdio/minimal_client.py
python examples/acp/stdio/full_client.py "Summarize this project"
```

The minimal client sends one fixed prompt and denies permission requests. The
full client accepts a prompt, `--follow-up`, `--workspace`, and `--session`; it
shows permission choices interactively and denies them when no choice is made.

## Experimental Streamable HTTP

Install `mira[acp-http]` and start the loopback-only MIRA server first:

```text
mira --acp --listen 127.0.0.1:8765
python examples/acp/http/minimal_client.py
python examples/acp/http/full_client.py "Summarize this project"
```

The full HTTP client accepts `--url`, `--workspace`, and `--follow-up`. ACP
0.12.1 cannot continue a loaded session on a new HTTP connection, so the HTTP
examples intentionally demonstrate new sessions and live follow-up turns only.
