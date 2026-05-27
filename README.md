# MIRA

Minimal Iterative Reasoning Agent.

MIRA is an educational Python CLI coding agent. The code is intentionally small
and direct so it is easy to read, change, and learn from.

## Quick Start

```powershell
pip install -e .
mira --help
mira
```

MIRA defaults to an LMStudio-compatible OpenAI endpoint at
`http://localhost:1234/v1`.

## Configuration

MIRA reads configuration from environment variables. You can set them in your
shell before running `mira`:

```powershell
$env:MIRA_LMSTUDIO_MODEL = "your-loaded-model-name"
$env:MIRA_LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
$env:MIRA_LMSTUDIO_API_KEY = "lm-studio"
mira
```

Or put them in a `.env` file in the directory where you run `mira`:

```dotenv
MIRA_LMSTUDIO_MODEL=your-loaded-model-name
MIRA_LMSTUDIO_BASE_URL=http://localhost:1234/v1
MIRA_LMSTUDIO_API_KEY=lm-studio
MIRA_TOOL_OUTPUT_CHARS=240
```

These values are loaded in `config/loader.py` and passed to `ChatOpenAI` in
`agent/llm.py`. `MIRA_TOOL_OUTPUT_CHARS` controls how many characters of each
tool result are shown in the terminal, including the final tool output shown
for subagents. Tool output is shown on one line; set the value to `0` to show
full output.

If you do not set them, MIRA uses:

```text
MIRA_LMSTUDIO_MODEL=local-model
MIRA_LMSTUDIO_BASE_URL=http://localhost:1234/v1
MIRA_LMSTUDIO_API_KEY=lm-studio
MIRA_TOOL_OUTPUT_CHARS=240
```

## Tool Calls And Subagents

When MIRA delegates work, the main agent calls the `task` tool. In the terminal
MIRA keeps this compact: each delegated worker appears once under its own
colored block with a readable suffix, for example
`subagent - general-purpose [ember]`. MIRA first shows a short delegation line
under `mira [main]`, then each subagent block shows the request and an animated
`RUNNING` status. When it finishes, the block switches to `DONE` and shows only
that subagent's final tool call, args, and output. The final output follows
`MIRA_TOOL_OUTPUT_CHARS`; set it to `0` when you want the full result. When the
main agent finishes a turn, the terminal prints `mira done`.
