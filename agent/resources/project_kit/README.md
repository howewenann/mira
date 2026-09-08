# MIRA in this project

## Start here

1. Add a model profile to `models.yml`. Put secrets in the project-root `.env`
   and reference them as `${NAME}`.
2. Start MIRA, run `/models`, and select the profile for **Main**.
3. Enter a normal prompt.

Use `/help` for commands, `/settings` for runtime and permissions, and `/models`
for model assignments. `/plan` starts read-only planning, `/goal` creates
success-checked work, `@` references a file, and `/` opens commands.

**MIRA is usable once a Main model is configured. Everything below is optional.**

## Choose what you want to do next

- Add project instructions or memory: [`examples/memories/AGENTS.md`](examples/memories/AGENTS.md)
- Add a skill: [`examples/skills/example-skill/SKILL.md`](examples/skills/example-skill/SKILL.md)
- Add a custom tool: [`examples/tools/`](examples/tools/)
- Add a subagent: [`examples/subagents/example_subagent.py`](examples/subagents/example_subagent.py)
- Connect MCP: [`examples/mcp/README.md`](examples/mcp/README.md)
- Use an ACP client: [`examples/acp/README.md`](examples/acp/README.md)
- Build a Python frontend: [`examples/api/README.md`](examples/api/README.md)
- Enable tracing: [`examples/tracing/README.md`](examples/tracing/README.md)
- Configure command execution: open `/settings`, enable `execute`, and choose
  System, Conda name, Conda prefix, or venv. Execution is disabled by default;
  approvals still apply.
- Understand permissions: read the next section, then inspect `/settings`.

Copy an example into its matching active directory before editing it. Everything
under `examples/` is inert and may be refreshed when MIRA is upgraded.

## Permissions in about 60 seconds

- **Enable** puts a tool on the normal agent surface.
- **Always Allow** skips its normal human approval prompt.
- **Plan** makes it available in Plan mode where policy permits.
- **PTC** makes it callable programmatically through `eval`.
- **Rubric** makes it available to the Rubric verifier.

Hard policy blocks still win: read-only Plan mode cannot be turned into an
action surface, and settings do not bypass built-in safety restrictions. For
MCP, server approval trusts that server configuration; per-tool Always Allow
controls invocation approval. See the [MCP recipe](examples/mcp/README.md).

## Project files and runtime state

`models.yml`, `tracing.yml`, `mcp/mcp.json`, and files in the active resource
directories are yours; MIRA creates missing templates but never overwrites
them. This README, `.env.example`, `mcp/schema.json`, and `examples/` describe
the installed MIRA version and refresh on launch when MIRA changes.

Underscore-prefixed directories are runtime state: `_sessions/`, `_logs/`,
`_errors/`, and `_phoenix/`. Commit the configuration and active resources your
team shares. Usually ignore underscore-prefixed runtime state and real `.env`
secrets. MIRA does not edit your project `.gitignore`.

## Troubleshooting

- `/issues` — startup and resource problems
- `/runtime` — active models and runtime state
- `/tools` — loaded tools and permissions
- `/mcp` — MCP servers and tools
- `/reload` — refresh lightweight project resources and settings
- `/reload-runtime` — rebuild models, MCP, tracing, and the full runtime

After editing models, MCP, tracing, or runtime dependencies, use
`/reload-runtime`. For ordinary memories, skills, prompts, subagents, and tool
files, use `/reload`.
