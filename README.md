# MIRA

Minimal Iterative Reasoning Agent.

MIRA is a small, educational general-purpose Python agent with a Textual
terminal UI, one-shot prompting, project-specific tools and context, planning,
approvals, and resumable sessions.

## Install

MIRA requires Python 3.11 or newer.

```bash
pip install mira
```

For development from this checkout, create the repository's Conda environment
and activate it:

```bash
conda env create -f environment.yml
conda activate mira
```

The environment file installs the current checkout in editable mode.

## Configure a model

Run MIRA once to create `.mira/models.yml`, then add a named model profile using
the schema guide and examples in that file. Reference secrets as `${NAME}` and
provide those variables in the environment that starts MIRA.

Open `/models` to select the Main model and optionally assign Rubric,
Summarization, context limits, and subagent models. Workspace settings live in
`.mira/settings.yml` and are managed with `/settings`.

## Start MIRA

Open the interactive TUI:

```bash
mira
```

Run one task and exit:

```bash
mira --prompt "summarize this project"
mira --file task.txt
mira --prompt "add focused tests" --rubric "The requested tests pass."
```

Use `mira --help` for startup options, including workspace and session
selection. `--direct` bypasses proxy settings and disables TLS verification for
the current process, so use it only with a trusted local model endpoint.

To expose MIRA to an ACP-compatible client, install the optional stock SDK
adapter. Stdio is the default child-process transport:

```bash
pip install "mira[acp]"
mira --acp
```

Experimental Streamable HTTP connects clients to an already-running,
localhost-only MIRA process:

```bash
pip install "mira[acp-http]"
mira --acp --listen 127.0.0.1:8765
```

The client supplies the workspace for each session. See the
[ACP adapter guide](protocols/acp/README.md) for supported capabilities.

## Python API

Run MIRA headlessly or build a frontend using the same supported API as the
owned Textual and terminal consumers:

```python
from mira import MiraApplication
from mira.api import Frontend, MessageEvent
```

See the [MIRA Python API guide](docs/frontend-api.md) for lifecycle, events,
blocking requests, snapshots, and a complete runnable frontend.

## TUI basics

| Action | Shortcut or command |
| --- | --- |
| Submit / insert newline | Enter / Shift+Enter |
| Copy selected text | Ctrl+C |
| Cancel active work or quit | Alt+Q |
| Show commands and usage | `/help` |
| Change settings / models | `/settings` / `/models` |
| Enter Plan mode / return to Act | `/plan` / `/act` |
| Create an outcome-focused Goal | `/goal <prompt>` |
| Reload agents / full runtime | `/reload` / `/reload-runtime` |
| View configuration problems | `/issues` |
| Open MCP status and controls | `/mcp` |

Type `/` to autocomplete commands and `@` to autocomplete local files, tools,
subagents, and MCP resources. Sessions are saved automatically; start with
`--resume` to reopen the latest one.

## Optional local tracing

MIRA preserves LangChain/DeepAgents trace topology with LangSmith and enriches
the same OpenTelemetry spans with OpenInference semantics before generic
OTLP/HTTP export. To try a local Phoenix backend:

```bash
pip install "mira[tracing]"
pip install arize-phoenix
```

Open `MIRA → Settings → General → Tracing → Yes → Config`, select the
bootstrapped `Phoenix` profile, and choose **Reload Runtime**. Then open the
Phoenix UI at `http://127.0.0.1:6006`. Add any other compatible OTLP/HTTP
profiles to `.mira/tracing.yml`; no MIRA code changes are required.

### Local tracing storage

Phoenix and MLflow manage their own local state independently of MIRA. Keeping
that state inside the project's `.mira/` directory is optional. Add these
backend settings to `.env`:

```dotenv
# Phoenix
PHOENIX_WORKING_DIR=.mira/phoenix

# MLflow
MLFLOW_BACKEND_STORE_URI=sqlite:///.mira/mlflow/mlflow.db
MLFLOW_ARTIFACTS_DESTINATION=.mira/mlflow/mlartifacts
```

An `.env` file does not configure independently launched CLI processes by
itself. Install the optional `python-dotenv` CLI and use `dotenv run --` to
inject those values when starting each backend:

```bash
pip install "python-dotenv[cli]"
dotenv run -- phoenix serve
dotenv run -- mlflow server
```

The `python-dotenv` CLI is a convenience for launching these servers, not a
required MIRA dependency. Phoenix uses `PHOENIX_WORKING_DIR` for its local
working data and may maintain multiple files there. Modern MLflow uses SQLite
as its default local backend and otherwise creates `mlflow.db` in the current
working directory. The settings above move that database and the tracking
server's served artifact destination under `.mira/mlflow/`; MLflow creates the
missing parent directories.

The resulting layout is roughly:

```text
.mira/
├── phoenix/
│   └── ...
└── mlflow/
    ├── mlflow.db
    └── mlartifacts/
```

These variables configure Phoenix and MLflow, not MIRA. MIRA still only
exports traces to the OTLP endpoints configured in its tracing profile.

## More information

See [docs/advanced.md](docs/advanced.md) for project resources, MCP, execution
environments, advanced tracing, storage, safety, and diagnostics. Design
rationale lives in [ARCHITECTURE_DECISIONS.md](ARCHITECTURE_DECISIONS.md).

Do not commit credentials, session data, or diagnostic logs.
