# ACP client examples

These clients use only the public `agent-client-protocol` SDK. Install
`mira[acp]` for stdio or `mira[acp-http]` for Streamable HTTP.

## Stdio: the client starts MIRA

```text
python examples/acp/stdio/minimal_client.py
python examples/acp/stdio/full_client.py
```

The minimal example sends one prompt and denies permissions. The full example
adds multiple prompts, permission choices, ACT/PLAN modes, cancellation, raw
updates, and durable `/load <session-id>` continuation.

## HTTP: the client connects to MIRA

Start the loopback-only server, then run a client in another terminal:

```text
mira --acp --listen 127.0.0.1:8765
python examples/acp/http/minimal_client.py
python examples/acp/http/full_client.py
```

The HTTP full client resembles the stdio version, but it does not own the MIRA
process. ACP 0.12.1 HTTP session loading is replay-only and cannot continue on
the new connection, so this example deliberately disables `/load`. Live
multi-turn prompts on one HTTP connection are supported.

In either full client, use `/help` to see its small set of local commands.
