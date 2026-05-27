from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory

from runtime.runner import run_turn


async def start_repl(agent, renderer, store, session: dict, model_name: str) -> None:
    renderer.splash(model_name=model_name, session_id=session["id"])

    history_path = Path(session["workspace"]) / ".mira" / "history.txt"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    prompt = PromptSession(history=FileHistory(str(history_path)))

    while True:
        try:
            text = await prompt.prompt_async(HTML("<b><ansicyan>you&gt;</ansicyan></b> "))
        except (EOFError, KeyboardInterrupt):
            renderer.newline()
            break

        text = text.strip()
        if not text:
            continue

        if await handle_command(text, renderer, store, session, model_name):
            if text in {"/exit", "/quit"}:
                break
            continue

        await run_turn(agent=agent, text=text, renderer=renderer, thread_id=session["id"])
        session["turns"] += 1
        store.save(session)


async def handle_command(text: str, renderer, store, session: dict, model_name: str) -> bool:
    if not text.startswith("/"):
        return False

    if text in {"/exit", "/quit"}:
        renderer.console.print("[dim]bye[/dim]")
        return True

    if text == "/help":
        renderer.console.print("Commands: /help, /exit, /clear, /session, /model")
        return True

    if text == "/clear":
        renderer.console.clear()
        return True

    if text == "/session":
        renderer.console.print(f"session: {session['id']}")
        renderer.console.print(f"workspace: {session['workspace']}")
        renderer.console.print(f"turns: {session['turns']}")
        return True

    if text == "/model":
        renderer.console.print(f"model: {model_name}")
        return True

    renderer.console.print(f"[dim]unknown command:[/dim] {text}")
    return True
