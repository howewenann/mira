# Trace MIRA with Phoenix

## Install

```text
pip install "mira[tracing]" arize-phoenix
```

`mira[all]` also includes MIRA's tracing dependencies.

## Start Phoenix

Without project environment values:

```text
phoenix serve
```

MIRA loads the project `.env` itself, but a separate Phoenix process does not.
Phoenix itself consumes values such as `PHOENIX_WORKING_DIR=.mira/_phoenix`.
To apply them, start it with:

```text
dotenv run -- phoenix serve
```

Success: Phoenix reports that its UI is listening on port 6006.

## Enable and view traces

Open `/settings`, select the existing `phoenix` tracing profile, enable tracing,
save, and run `/reload-runtime`. Send a prompt, then open:

```text
http://127.0.0.1:6006
```
