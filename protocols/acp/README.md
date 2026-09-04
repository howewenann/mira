# MIRA ACP adapter

ACP is for external ACP-compatible clients. For direct Python embedding, use
MIRA's public Python API instead.

## Stdio

Install the lightweight extra and start MIRA as a stdio ACP agent:

```text
pip install "mira[acp]"
mira --acp
```

The client owns and spawns the MIRA child process. This is the normal choice
for editors, CODA, and other local ACP hosts; stdio is not an attachment point
for an already-running server.

## Experimental Streamable HTTP

Install the HTTP extra and start a persistent local process:

```text
pip install "mira[acp-http]"
mira --acp --listen 127.0.0.1:8765
```

Clients connect to `http://127.0.0.1:8765/acp`. HTTP uses the stock SDK's
`create_asgi_app()` with Hypercorn and is localhost-only. MIRA rejects wildcard,
LAN, and other non-loopback bind addresses because this experimental transport
does not yet provide authentication or TLS. WebSocket is not a supported MIRA
product surface in this release.

The SDK creates one `MiraAgent` and `ACPFrontend` per HTTP connection, so live
frontend state is never shared across independent clients. The public ACP 0.12.1
HTTP API does not expose a per-connection agent teardown callback. MIRA retains
the agents it creates and deterministically shuts down all MIRA applications,
sessions, frontend senders, and MCP-backed resources when the HTTP server stops.
Connection close alone therefore does not immediately release MIRA-owned
resources.

There is one further stock-SDK limitation: `session/load` on a new HTTP
connection loads and replays a durable MIRA transcript, but ACP 0.12.1 does not
associate that loaded session with the new HTTP connection for subsequent
session-scoped requests. Continuing the loaded session then receives HTTP 404.
MIRA does not use private transport hooks to bypass this limitation. New HTTP
sessions and repeated prompts on the same live connection are supported; stdio
continues to support full durable load-and-continue behavior.

The bare-minimum stock-SDK examples send one prompt and deny permission requests
instead of auto-approving tool use:

```text
python examples/acp_stdio_client.py
python examples/acp_http_client.py
```

Start `mira --acp --listen 127.0.0.1:8765` before running the HTTP example. For
custom prompts, follow-up turns, and durable stdio session loading, use
`examples/acp_client.py`. Its `--session` option is deliberately rejected with
HTTP while the ACP 0.12.1 limitation above remains.

## Shared behavior

MIRA implements the stock `agent-client-protocol` Python `Agent` interface
directly. The adapter translates ACP lifecycle calls into
`MiraApplication`/`MiraSession` operations and translates MIRA's public
frontend events and requests back into ACP updates. It does not execute or
stream a LangGraph graph, and DeepAgents is not part of the protocol adapter.

MIRA owns workspace normalization, application reuse, durable sessions,
transcript replay, ACT/PLAN mode, cancellation, permissions, MCP trust, and
formal artifacts. Client-provided MCP servers and additional workspace
directories are rejected. MIRA ACP currently accepts text prompt content only;
resources, images, and audio fail with an ACP invalid-params response.

AskUser and review interactions use stable ACP permission buttons. The AskUser
question is the permission surface's title, so it is not appended to the
preceding ordinary assistant message. Suggested answers are buttons and
`Reply in chat` is always available for open text. Selecting it ends the
current interrupted turn without supplying that label—or any invented
answer—to the model; the next ordinary prompt carries the user's free-text
response. Goal and Plan reviews similarly offer `Implement`, `Keep`, and
`Revise in chat`.

Formal MIRA Goals and Plans are rendered as ordinary visible agent messages.
The finalizer tool is completed and flushed first, establishing a message
boundary before the artifact, and the artifact is flushed before its review
buttons. These remain MIRA-owned artifacts, not ACP todo plans. Only state
emitted by the actual MIRA `write_todos` tool is projected as an ACP
`AgentPlanUpdate`.

Both transports reuse the same `MiraAgent`, `ACPFrontend`, mapper, and public
`mira`/`mira.api` boundary. Neither invokes MIRA internals or the LangGraph graph
directly. The stdio runner uses only the stable ACP protocol path and reports
the pinned SDK's supported protocol version. On Windows it completes optional
NumPy/native initialization before starting the SDK's stdin reader; HTTP does
not apply that stdio-specific preload.
