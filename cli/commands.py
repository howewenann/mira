import asyncio
import warnings
from pathlib import Path


def run(prompt: str | None, resume: bool, workspace: Path, session: str | None) -> None:
    _suppress_known_warnings()
    asyncio.run(_run(prompt=prompt, resume=resume, workspace=workspace, session=session))


def _suppress_known_warnings() -> None:
    from langchain_core._api import LangChainBetaWarning

    warnings.filterwarnings("ignore", category=LangChainBetaWarning)


async def _run(prompt: str | None, resume: bool, workspace: Path, session: str | None) -> None:
    from runtime.runner import run_turn
    from ui.repl import start_repl

    app = _bootstrap(workspace=workspace, session=session, resume=resume)

    if prompt:
        await run_turn(
            agent=app["agent"],
            text=prompt,
            renderer=app["renderer"],
            thread_id=app["session"]["id"],
        )
        app["store"].save(app["session"])
        return

    await start_repl(
        agent=app["agent"],
        renderer=app["renderer"],
        store=app["store"],
        session=app["session"],
        model_name=app["model_name"],
    )


def _bootstrap(workspace: Path, session: str | None, resume: bool) -> dict:
    from agent.factory import build_agent
    from agent.llm import get_model_name
    from config.loader import load_config
    from session.checkpoint import make_checkpointer
    from session.store import SessionStore
    from ui.renderer import Renderer

    workspace = workspace.expanduser().resolve()
    config = load_config(workspace)
    store = SessionStore(Path(config["session_dir"]))
    record = store.load(session, resume=resume, workspace=workspace)
    checkpointer = make_checkpointer()
    renderer = Renderer(tool_output_chars=config["tool_output_chars"])
    agent = build_agent(config=config, workspace=workspace, checkpointer=checkpointer)

    return {
        "agent": agent,
        "config": config,
        "model_name": get_model_name(config),
        "renderer": renderer,
        "session": record,
        "store": store,
    }
