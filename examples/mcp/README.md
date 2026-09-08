# Connect an MCP server

MIRA reads active servers from `.mira/mcp/mcp.json`. The adjacent
`schema.json` provides editor help. After every configuration change, run:

```text
/reload-runtime
```

## Project-local stdio server

This example installs the real `mcp-server-fetch` server inside the current
project, isolated from both MIRA and the host project's Python dependencies:

```text
uv init .mira/mcp/servers/fetch --bare --no-workspace
uv add --project .mira/mcp/servers/fetch mcp-server-fetch
```

Add this server under `mcpServers` in `.mira/mcp/mcp.json`:

```json
{
  "$schema": "./schema.json",
  "mcpServers": {
    "fetch": {
      "type": "stdio",
      "command": "uv",
      "args": [
        "run",
        "--project",
        ".mira/mcp/servers/fetch",
        "mcp-server-fetch"
      ]
    }
  }
}
```

`uvx` is intentionally not used: this recipe demonstrates a persistent,
project-local installation whose dependencies do not modify either Python
environment.

## HTTP server and environment values

```json
{
  "$schema": "./schema.json",
  "mcpServers": {
    "remote": {
      "type": "http",
      "url": "https://example.com/mcp",
      "headers": {
        "Authorization": "Bearer ${REMOTE_MCP_TOKEN}"
      }
    }
  }
}
```

Put `REMOTE_MCP_TOKEN` in the project-root `.env`. MIRA keeps `${ENV_VAR}`
references unresolved in the saved configuration and approval display.

On reload, approve the server before MIRA launches or connects to it. **Server
Always Allow** trusts that exact server configuration/fingerprint on later
reloads. **Tool Always Allow** separately skips the approval normally shown
when that individual MCP tool is invoked. Review changed server commands, URLs,
arguments, environment, and headers before trusting a new fingerprint.
