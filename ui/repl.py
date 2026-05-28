from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory

from agent.plan_policy import PLAN_BLOCKED_RESULT_MARKERS, PLAN_PROJECT_WRITE_TOOLS, project_write_tools_text
from runtime.runner import run_turn

PLAN_CONTEXT_TEMPLATE = """Previous planning context:
{plan}

You are now in action mode. Write/edit tools are available again, subject to normal approval prompts. Do not assume planning-mode permission errors still apply.

User request:
{text}"""

PLAN_REQUEST_TEMPLATE = """You are in planning mode.
Do not call write_file, edit_file, or any other tool that modifies files.
Do not attempt the requested change.
Respond only with a concrete plan for how the change should be done later in action mode.
If the user asks to create, write, edit, or delete files, describe the exact file operation that should happen after /act.

User request:
{text}"""


async def start_repl(agent, plan_agent, renderer, store, session: dict, model_name: str) -> None:
    renderer.splash(model_name=model_name, session_id=session["id"])
    mode = {"planning": False, "last_plan": "", "plan_pending": False, "plans": [], "plan_runs": 0}

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

        if await handle_command(text, renderer, store, session, model_name, mode):
            if text in {"/exit", "/quit"}:
                break
            continue

        if mode["planning"]:
            result = await run_turn(
                agent=plan_agent,
                text=plan_request_text(text),
                renderer=renderer,
                thread_id=mode["plan_thread_id"],
            )
            save_clean_plan(mode, result, renderer)
        else:
            action_text = action_request_text(mode, text)
            await run_turn(agent=agent, text=action_text, renderer=renderer, thread_id=session["id"])

        session["turns"] += 1
        store.save(session)


async def handle_command(text: str, renderer, store, session: dict, model_name: str, mode: dict | None = None) -> bool:
    if not text.startswith("/"):
        return False

    mode = mode if mode is not None else {"planning": False}

    if text in {"/exit", "/quit"}:
        renderer.console.print("[dim]bye[/dim]")
        return True

    if text == "/help":
        renderer.console.print("Commands: /help, /exit, /clear, /session, /model, /plan, /act, /plans")
        return True

    if text == "/plan":
        mode["planning"] = True
        mode["last_plan"] = ""
        mode["plan_pending"] = False
        mode["plan_runs"] = mode.get("plan_runs", 0) + 1
        mode["plan_thread_id"] = plan_thread_id(session, mode["plan_runs"])
        renderer.console.print(f"[cyan]planning mode[/cyan]: {project_write_tools_text()} disabled; use /act to leave")
        return True

    if text == "/act":
        mode["planning"] = False
        if mode.get("last_plan"):
            mode["plan_pending"] = True
            renderer.console.print("[cyan]action mode[/cyan]: last plan will be included in your next request")
        else:
            renderer.console.print("[cyan]action mode[/cyan]")
        return True

    if text == "/clear":
        renderer.console.clear()
        return True

    if text == "/plans":
        print_plans(renderer, mode)
        return True

    if text == "/session":
        renderer.console.print(f"session: {session['id']}")
        renderer.console.print(f"mode: {'planning' if mode['planning'] else 'action'}")
        renderer.console.print(f"saved plans: {len(mode.get('plans', []))}")
        renderer.console.print(f"workspace: {session['workspace']}")
        renderer.console.print(f"turns: {session['turns']}")
        return True

    if text == "/model":
        renderer.console.print(f"model: {model_name}")
        return True

    renderer.console.print(f"[dim]unknown command:[/dim] {text}")
    return True


def plan_thread_id(session: dict, run_id: int | None = None) -> str:
    if run_id is None:
        return f"{session['id']}:plan"

    return f"{session['id']}:plan:{run_id}"


def save_clean_plan(mode: dict, result, renderer) -> None:
    if not has_clean_plan(result):
        mode["last_plan"] = ""
        mode["plan_pending"] = False
        if write_was_blocked(result) or write_tool_was_used(result):
            renderer.console.print("[yellow]planning mode[/yellow]: write/edit was blocked; no plan was saved")
        return

    plan_text = result.final_text.strip()
    plan = {"id": len(mode.setdefault("plans", [])) + 1, "text": plan_text}
    mode["plans"].append(plan)
    mode["last_plan"] = plan_text
    mode["plan_pending"] = False


def has_clean_plan(result) -> bool:
    final_text = getattr(result, "final_text", "").strip()
    if not final_text:
        return False

    if any(marker in final_text.lower() for marker in PLAN_BLOCKED_RESULT_MARKERS):
        return False

    if write_tool_was_used(result):
        return False

    if write_was_blocked(result):
        return False

    return True


def write_tool_was_used(result) -> bool:
    return bool(set(PLAN_PROJECT_WRITE_TOOLS).intersection(getattr(result, "tool_calls", [])))


def write_was_blocked(result) -> bool:
    tool_results = getattr(result, "tool_results", [])
    return any(
        marker in value.lower()
        for value in tool_results
        for marker in PLAN_BLOCKED_RESULT_MARKERS
    )


def print_plans(renderer, mode: dict) -> None:
    plans = mode.get("plans", [])
    if not plans:
        if hasattr(renderer, "no_plans"):
            renderer.no_plans()
        else:
            renderer.console.print("[dim]no saved plans[/dim]")
        return

    for plan in plans:
        if hasattr(renderer, "plan"):
            renderer.plan(plan["id"], plan["text"])
        else:
            renderer.console.print(plan["text"])


def plan_request_text(text: str) -> str:
    return PLAN_REQUEST_TEMPLATE.format(text=text)


def action_request_text(mode: dict, text: str) -> str:
    if not mode.get("plan_pending") or not mode.get("last_plan"):
        return text

    plan = mode["last_plan"]
    mode["last_plan"] = ""
    mode["plan_pending"] = False
    return PLAN_CONTEXT_TEMPLATE.format(plan=plan, text=text)
