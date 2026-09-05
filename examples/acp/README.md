# ACP client examples

ACP clients are the frontend side of an ACP connection. The stdio and HTTP
examples use only public APIs from the stock `agent-client-protocol` SDK.

## Stdio

Install `mira[acp]`. The client launches `python -m cli.main --acp` as its child
process, so no MIRA server is started separately.

```text
conda run -n mira --no-capture-output python examples/acp/stdio/minimal_client.py
conda run -n mira --no-capture-output python examples/acp/stdio/full_client.py
conda run -n mira --no-capture-output python examples/acp/stdio/full_client.py "Summarize this project"
```

The minimal client sends one fixed prompt and denies permission requests. The
full client is MIRA's canonical ACP stdio reference console. Without a prompt it
starts an interactive REPL; with a prompt it remains useful for scripted tests.
It accepts `--follow-up`, `--workspace`, and `--session`, shows generic ACP
permission choices, and denies them when no valid choice is made.

## Experimental Streamable HTTP

Install `mira[acp-http]` and start the loopback-only MIRA server first:

```text
conda run -n mira --no-capture-output mira --acp --listen 127.0.0.1:8765
conda run -n mira --no-capture-output python examples/acp/http/minimal_client.py
conda run -n mira --no-capture-output python examples/acp/http/full_client.py
conda run -n mira --no-capture-output python examples/acp/http/full_client.py "Summarize this project"
```

The full HTTP client provides the same reference-console presentation and
commands where the protocol permits them. It accepts `--url`, `--workspace`,
and `--follow-up`. ACP 0.12.1 cannot continue a loaded session on a new HTTP
connection, so `/load` is deliberately disabled and explains the replay-only
limitation.

Both full clients support `/help`, `/new`, `/session`, `/mode act`, `/mode plan`,
`/raw on`, `/raw off`, `/cancel-after`, and `/quit`. The stdio client also
supports `/load <session-id>` for durable replay and continuation.

Use [`TESTING.md`](TESTING.md) for practical prompts, expected ACP-visible
updates, and the exact MIRA-to-ACP coverage boundary.
