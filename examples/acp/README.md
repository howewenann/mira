# Use MIRA through ACP

An ACP frontend launches MIRA as an external agent. Configuration formats vary,
but look for fields equivalent to:

```text
Name:    MIRA
Command: mira
Args:    --acp
```

For MIRA installed in a Conda environment:

```text
Name:    MIRA
Command: conda
Args:    run -n mira --no-capture-output mira --acp
```

`--no-capture-output` is required because ACP uses stdout for its protocol.
For Zed's exact configuration, see [`zed.md`](zed.md).

These client examples use only the public `agent-client-protocol` SDK. Install
`mira[acp]` for stdio or `mira[acp-http]` for Streamable HTTP.

## Stdio: the client launches MIRA

```text
python .mira/examples/acp/stdio/minimal_client.py
python .mira/examples/acp/stdio/full_client.py
```

The minimal example sends three scripted prompts through one session. The full
example retains the interactive loop, permission choices, ACT/PLAN modes,
cancellation, raw updates, and durable `/load <session-id>` continuation.

Stdio also supports `load_session()`: a later client process can load a durable
session ID and continue the conversation.

## HTTP: the client connects to MIRA

Start the loopback-only server, then run a client in another terminal:

```text
mira --acp --listen 127.0.0.1:8765
python .mira/examples/acp/http/minimal_client.py
python .mira/examples/acp/http/full_client.py
```

The minimal HTTP example sends three scripted prompts through one session on
the live connection. The full client retains the interactive behavior but does
not own the MIRA process.

Current ACP HTTP `session/load` behavior on a new connection is replay-only and
cannot reliably continue that session. Multiple prompts may reuse one session
ID on the same live HTTP connection; the ID alone does not provide continuity
across HTTP connections. Stdio does not have this limitation.

In either full client, use `/help` to see its small set of local commands.
