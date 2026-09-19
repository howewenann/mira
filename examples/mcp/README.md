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

## Removable on-demand launcher caches

On-demand launchers can keep their package and runtime files under a directory
that is easy to remove with the server configuration. The recommended layout
is `.mira/cache/mcp/<server>/`. MIRA does not inspect commands or add these
options automatically.

For `uvx`, pass its cache directory explicitly:

```json
{
  "command": "uvx",
  "args": [
    "--cache-dir",
    ".mira/cache/mcp/fetch/uv",
    "mcp-server-fetch"
  ]
}
```

This was manually verified with `mcp-server-fetch`: a cold launch took about
4.0 seconds, a warm launch about 0.7 seconds, and deleting the configured
directory caused it to be recreated on the next cold launch in about 4.2
seconds.

For `pipx run`, keep its home and both installer caches beneath one parent:

```json
{
  "command": "pipx",
  "args": [
    "run",
    "--spec",
    "mcp-server-fetch",
    "mcp-server-fetch"
  ],
  "env": {
    "PIPX_HOME": ".mira/cache/mcp/fetch/pipx/home",
    "PIP_CACHE_DIR": ".mira/cache/mcp/fetch/pipx/pip-cache",
    "UV_CACHE_DIR": ".mira/cache/mcp/fetch/pipx/uv-cache"
  }
}
```

The installed `pipx` resolves `PIPX_VENV_CACHEDIR` to
`<PIPX_HOME>/.cache` and `PIPX_STANDALONE_PYTHON_CACHEDIR` to
`<PIPX_HOME>/py`. Cold, warm, delete-parent, and cold-recreate launches were
manually verified.

For `npx`, use its supported cache argument:

```json
{
  "command": "npx",
  "args": [
    "-y",
    "--cache",
    ".mira/cache/mcp/filesystem/npm",
    "@modelcontextprotocol/server-filesystem",
    "D:/some/path"
  ]
}
```

Cold, warm, delete, and cold-recreate behavior was manually verified.

For `pnpm dlx`, configure both cache and content-addressable store through the
environment. Do not put `--cache-dir` or `--store-dir` before `dlx`.

```json
{
  "command": "pnpm",
  "args": [
    "dlx",
    "@modelcontextprotocol/server-filesystem",
    "D:/some/path"
  ],
  "env": {
    "PNPM_CONFIG_CACHE_DIR": ".mira/cache/mcp/filesystem/pnpm/cache",
    "PNPM_CONFIG_STORE_DIR": ".mira/cache/mcp/filesystem/pnpm/store"
  }
}
```

Both directories are required. This was manually verified at about 3.5
seconds cold, 0.2 seconds warm, and 3.1 seconds after deleting the parent.

`bunx` supports a removable cache through `BUN_INSTALL_CACHE_DIR`, although
this example was not manually verified because `bunx` was unavailable in the
probe environment:

```json
{
  "command": "bunx",
  "args": [
    "@modelcontextprotocol/server-filesystem",
    "D:/some/path"
  ],
  "env": {
    "BUN_INSTALL_CACHE_DIR": ".mira/cache/mcp/filesystem/bun"
  }
}
```

Docker uses daemon-managed image storage rather than a per-MCP installation
directory. Remove an unwanted image by identity with `docker image rm <image>`.

MIRA displays stderr that a launcher actually emits while connecting. Some
launchers suppress download progress when stderr is not a terminal; those
servers still show honest elapsed startup time, never manufactured percentages.

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
