# Use MIRA from Zed

Add a custom external ACP agent to Zed's settings:

```json
{
  "agent_servers": {
    "MIRA": {
      "type": "custom",
      "command": "mira",
      "args": ["--acp"],
      "env": {}
    }
  }
}
```

For a Conda installation, use:

```json
{
  "agent_servers": {
    "MIRA": {
      "type": "custom",
      "command": "conda",
      "args": ["run", "-n", "mira", "--no-capture-output", "mira", "--acp"],
      "env": {}
    }
  }
}
```

Zed launches MIRA, so do **not** separately run `mira --acp`. Select MIRA as
the external agent in Zed and start a thread. `--no-capture-output` is required
in the Conda form because ACP uses stdout for its protocol stream.
