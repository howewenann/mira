# MIRA ACP adapter

Install the optional dependency and start MIRA as a stdio ACP agent:

```text
pip install "mira[acp]"
mira --acp
```

MIRA implements the stock `agent-client-protocol` Python `Agent` interface
directly. The adapter translates ACP lifecycle calls into
`MiraApplication`/`MiraSession` operations and translates MIRA's public
frontend events and requests back into ACP updates. It does not execute or
stream a LangGraph graph, and DeepAgents is not part of the protocol adapter.

MIRA owns workspace normalization, application reuse, durable sessions,
transcript replay, ACT/PLAN mode, cancellation, permissions, MCP trust, and
formal artifacts. Client-provided MCP servers and additional workspace
directories are rejected. This integration serves stdio only and accepts text
prompt blocks.

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

The runner uses only the stable ACP protocol path. On Windows it completes
optional NumPy/native initialization before starting the SDK's stdin reader.
