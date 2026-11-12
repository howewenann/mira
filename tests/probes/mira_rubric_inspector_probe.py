#!/usr/bin/env python
"""MIRA live Rubric offload probe using the actual MIRA Textual Inspector widget.

Run from the MIRA repository in its environment:

    python tests/probes/mira_rubric_inspector_probe.py
    python tests/probes/mira_rubric_inspector_probe.py --session-id SESSION_ID

1: Verifier Inspector, 2: Grader Inspector, Ctrl+Q: exit.
The existing source session is backed up read-only, never resumed or modified.
Candidate offloading is isolated in temporary files and cleaned up on exit.
The displayed events come from actual nested MIRA Verifier/Grader executions.
"""


from __future__ import annotations

import argparse
import asyncio
import re
import sqlite3
import sys
import tempfile
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, default=None, help="MIRA repository root; default: current directory or script ancestor")
    ap.add_argument("--workspace", type=Path, default=None, help="MIRA workspace with .mira settings/sessions (default: --repo)")
    ap.add_argument("--session-id", default=None, help="Saved session id; default: search recent saved sessions for an oversized rubric message")
    ap.add_argument("--thread-id", default=None, help="Optional exact LangGraph thread id within --session-id (Act/Plan/Goal)")
    ap.add_argument("--scan-sessions", type=int, default=25, help="How many recent sessions to inspect if --session-id is omitted (default: 25)")
    ap.add_argument("--rubric-file", type=Path, default=None, help="Optional rubric text; otherwise use saved rubric, or a diagnostic rubric")
    ap.add_argument("--synthetic", action="store_true", help="Append one 6k-char synthetic message IN MEMORY if necessary (no checkpoint mutation)")
    ap.add_argument("--dry-run", action="store_true", help="Read snapshot, offload and verify files; make NO model calls")
    ap.add_argument("--max-total-chars", type=int, default=250_000, help="Fail safely above this total offloaded text size (default: 250000)")
    return ap.parse_args()


def _repo(path: Path | None) -> Path:
    candidates = [path] if path else [Path.cwd(), *Path(__file__).resolve().parents]
    for p in candidates:
        if p is None:
            continue
        root = p.expanduser().resolve()
        if (root / "agent" / "rubric" / "middleware.py").is_file() and (root / "config" / "loader.py").is_file():
            return root
    raise RuntimeError("MIRA source not found. Use --repo D:\\Projects\\mira or run from its repo root.")


def _session_candidates(config: dict[str, Any], requested: str | None, scan_limit: int) -> list[tuple[str, Path]]:
    """Discover session *metadata*, without touching or resuming any session."""
    from session.store import SessionStore

    store = SessionStore(Path(config["session_dir"]).expanduser().resolve())
    if requested:
        path = store.path(requested)  # validates the id
        if not path.is_file():
            raise RuntimeError(f"Saved session metadata not found: {path}")
        db = store.database_path(requested)
        if not db.is_file():
            raise RuntimeError(f"Checkpoint database missing: {db}")
        return [(requested, db)]

    paths = sorted(store.session_paths(), key=store._updated_at_timestamp, reverse=True)
    candidates = [
        (p.parent.name, store.database_path(p.parent.name))
        for p in paths if store.database_path(p.parent.name).is_file()
    ]
    if not candidates:
        raise RuntimeError(f"No saved checkpoint databases in {store.root}")
    return candidates[:max(1, scan_limit)]


def _copy_db_read_only(src: Path, dest: Path) -> None:
    """Online SQLite backup: never open source read/write, safe while MIRA runs."""
    with sqlite3.connect(src.resolve().as_uri() + "?mode=ro", uri=True, timeout=15) as connection:
        with sqlite3.connect(dest) as cloned:
            connection.backup(cloned, pages=128, sleep=0.05)


async def _checkpoint_messages(
    db: Path, session_id: str, *, requested_thread: str | None = None,
) -> list[tuple[str, list[Any], dict[str, Any]]]:
    """Read all of this session's Act/Plan/Goal threads from a DB backup.

    DeepAgents stores messages in LangGraph DeltaChannel. Most head
    checkpoints intentionally OMIT `messages` from `channel_values`.
    Reconstruct using the checkpointer's native ancestor history; NEVER treat
    the missing head value as an empty conversation.
    """
    import aiosqlite
    from deepagents._messages_reducer import _messages_delta_reducer
    from langgraph._internal._typing import MISSING
    from langgraph.channels.delta import DeltaChannel
    from langgraph.checkpoint.serde.types import _DeltaSnapshot
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    snapshots: list[tuple[str, list[Any], dict[str, Any]]] = []
    async with aiosqlite.connect(str(db)) as connection:
        async with connection.execute(
            "SELECT thread_id, MAX(checkpoint_id) AS last_id FROM checkpoints "
            "WHERE checkpoint_ns = '' GROUP BY thread_id ORDER BY last_id DESC"
        ) as cursor:
            rows = await cursor.fetchall()
        thread_ids = [
            str(row[0]) for row in rows
            if (str(row[0]) == requested_thread if requested_thread else (
                str(row[0]) == session_id or str(row[0]).startswith(session_id + ":")
            ))
            and not str(row[0]).endswith(":session")
        ]
        if requested_thread and requested_thread not in thread_ids:
            raise RuntimeError(
                f"Thread {requested_thread!r} has no root checkpoints in session {session_id!r}."
            )
        saver = AsyncSqliteSaver(connection)
        for thread_id in thread_ids:
            cfg = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
            checkpoint = await saver.aget_tuple(cfg)
            if checkpoint is None:
                continue
            saved = checkpoint.checkpoint
            values = dict(saved.get("channel_values") or {})
            raw = values.get("messages")
            if isinstance(raw, _DeltaSnapshot):
                messages = list(raw.value)
            elif raw is not None:
                messages = list(raw)
            elif "messages" in (saved.get("channel_versions") or {}):
                # This is the typical case: a DeltaChannel's latest checkpoint
                # stores no full messages blob. Native history provides its
                # nearest snapshot + ordered ancestor writes.
                if not hasattr(saver, "aget_delta_channel_history"):
                    raise RuntimeError(
                        "Installed LangGraph saver lacks native DeltaChannel history; "
                        "cannot safely reconstruct messages from this checkpoint."
                    )
                histories = await saver.aget_delta_channel_history(
                    config=checkpoint.config or cfg,
                    channels=["messages"],
                )
                history = histories.get("messages", {})
                channel = DeltaChannel(
                    _messages_delta_reducer, list, snapshot_frequency=50,
                ).from_checkpoint(history.get("seed", MISSING))
                channel.replay_writes(history.get("writes", []))
                messages = list(channel.get())
            else:
                messages = []
            if messages:
                values["messages"] = messages
                snapshots.append((thread_id, messages, values))
    return snapshots


def _effective_messages(messages: list[Any], values: dict[str, Any]) -> list[Any]:
    """Match MIRA's compaction-aware conversation whenever possible."""
    if not values.get("_summarization_event"):
        return messages
    from agent.middleware.compaction import normalize_summarization_event
    from deepagents.middleware.summarization import _DeepAgentsSummarizationMiddleware
    return list(_DeepAgentsSummarizationMiddleware._apply_event_to_messages(
        messages, normalize_summarization_event(values.get("_summarization_event"))
    ))


def _selected_indices(messages: list[Any]) -> list[int]:
    """Mirror DeepAgents 0.7.32's first-real-human plus last-30 window."""
    from deepagents.middleware import rubric as stock
    from langchain_core.messages import HumanMessage

    n = len(messages)
    count = stock._MAX_TRANSCRIPT_MESSAGES
    indices = list(range(max(0, n - count), n))
    for i, message in enumerate(messages):
        if isinstance(message, HumanMessage) and message.additional_kwargs.get("lc_source") != stock.RUBRIC_GRADER_MESSAGE_SOURCE:
            if i not in indices:
                return [i, *indices]
            break
    return indices


def _surrogate(original: Any, breadcrumb: str) -> Any:
    """Evaluation-only copy; preserve role and tool-call IDs, not tool-call text."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    if isinstance(original, HumanMessage):
        return HumanMessage(content=breadcrumb)
    if isinstance(original, ToolMessage):
        return ToolMessage(content=breadcrumb, tool_call_id=original.tool_call_id, name=original.name)
    if isinstance(original, AIMessage):
        return AIMessage(content=breadcrumb)
    return original.model_copy(update={"content": breadcrumb}, deep=True)


def _offload(messages: list[Any], backend: Any, max_total_chars: int) -> tuple[list[Any], dict[str, str]]:
    """Use stock _coerce_text and limits, only on messages actually selected."""
    from deepagents.middleware import rubric as stock

    modified = list(messages)
    selected = _selected_indices(messages)
    needed: dict[str, str] = {}
    for index in selected:
        original = messages[index]
        text = stock._coerce_text(original)
        if len(text) <= stock._MAX_TRANSCRIPT_CHARS_PER_MESSAGE:
            continue
        path = f"/offloaded/message-{index + 1}.txt"
        needed[path] = text
    total = sum(map(len, needed.values()))
    if not needed:
        raise RuntimeError("No oversized messages in the stock rubric window. Try --synthetic.")
    if total > max_total_chars:
        raise RuntimeError(f"Would offload {total:,} characters, exceeding safety cap {max_total_chars:,}; adjust --max-total-chars if intended.")
    for path, content in needed.items():
        result = backend.write(path, content)
        if result.error:
            raise RuntimeError(f"DeepAgents backend.write failed for {path}: {result.error}")
        check = backend.read(path, offset=0, limit=max(1, len(content.splitlines()) + 1))
        if check.error or check.file_data is None:
            raise RuntimeError(f"DeepAgents backend cannot read freshly written {path}: {check.error}")
        # This direct backend read checks disk integrity, not the verifier's tool read.
        stored = check.file_data["content"]
        if stored.replace("\r\n", "\n").rstrip("\n") != content.replace("\r\n", "\n").rstrip("\n"):
            raise RuntimeError(f"Offloaded file's content differs from original: {path}")
    for index in selected:
        path = f"/offloaded/message-{index + 1}.txt"
        if path in needed:
            modified[index] = _surrogate(messages[index], f"[Rubric evidence offloaded: {path}. Read this complete file to recover this message's original content.]")
    return modified, needed


_HEADER = re.compile(r"(?m)^@@ lines (\d+)-(\d+) of (\d+)(?: \| [^\n]*?)? @@\n")


def _complete_tool_pages(evidence: list[Any], required: dict[str, str]) -> dict[str, dict[str, Any]]:
    """Compare EVERY returned page to the source text, not a 240-char UI preview.

    DeepAgents 0.7.32 formats read_file results as a header followed by
    verbatim source lines. Check exact content for every paginated window and
    require every original line to be represented by at least one good read.
    """
    from langchain_core.messages import AIMessage, ToolMessage

    calls: dict[str, str] = {}
    for msg in evidence:
        if isinstance(msg, AIMessage):
            for call in msg.tool_calls:
                if call.get("name") == "read_file" and isinstance(call.get("args"), dict):
                    calls[str(call.get("id"))] = str(call["args"].get("file_path") or "")
    pages: dict[str, list[tuple[int, int, int, str]]] = defaultdict(list)
    per_file_counts: dict[str, int] = defaultdict(int)
    for msg in evidence:
        if not isinstance(msg, ToolMessage):
            continue
        path = calls.get(str(msg.tool_call_id))
        if path not in required:
            continue
        per_file_counts[path] += 1
        if str(msg.status or "") == "error" or not isinstance(msg.content, str):
            continue
        match = _HEADER.search(msg.content)
        if not match:
            continue
        start, end, total = map(int, match.groups())
        # Model-visible body contains the source data. Notices precede the header.
        body = msg.content[match.end():]
        pages[path].append((start, end, total, body))
    report: dict[str, dict[str, Any]] = {}
    for path, source in required.items():
        source_lines = source.replace("\r\n", "\n").split("\n")
        if source_lines and source_lines[-1] == "":
            source_lines.pop()
        matched_lines: set[int] = set()
        mismatched_pages: list[str] = []
        for start, end, total, body in pages[path]:
            if start < 1 or end < start or total != len(source_lines) or end > len(source_lines):
                mismatched_pages.append(f"invalid window {start}-{end} of {total}; source has {len(source_lines)} lines")
                continue
            expected = "\n".join(source_lines[start - 1:end])
            if body != expected:
                mismatched_pages.append(f"page {start}-{end} content mismatched ({len(body)} vs {len(expected)} chars)")
                continue
            matched_lines.update(range(start, end + 1))
        missing = sorted(set(range(1, len(source_lines) + 1)) - matched_lines)
        report[path] = {
            "read_calls": per_file_counts[path],
            "matched_pages": len(pages[path]) - len(mismatched_pages),
            "covered_lines": len(matched_lines),
            "total_lines": len(source_lines),
            "missing_lines": missing,
            "mismatched_pages": mismatched_pages,
            "complete": not missing and not mismatched_pages and per_file_counts[path] > 0,
        }
    return report


async def _live_probe(config: dict[str, Any], messages: list[Any], values: dict[str, Any], backend: Any, required: dict[str, str], surrogate_messages: list[Any], rubric_text: str) -> None:
    from deepagents.middleware.filesystem import FilesystemMiddleware
    from langchain_core.messages import HumanMessage, ToolMessage
    from agent.llm import get_llm, get_rubric_model_name
    from agent.rubric.middleware import MiraRubricMiddleware
    from config.settings import RUBRIC_MODEL

    print(f"Rubric LLM: {get_rubric_model_name(config)}", flush=True)
    model = get_llm(config, role=RUBRIC_MODEL)
    fs = FilesystemMiddleware(backend=backend, tools=["read_file"])
    middleware = MiraRubricMiddleware(
        model=model,
        verifier_tools=[],
        verifier_middleware=[fs],
        max_iterations=1,
    )
    state: dict[str, Any] = {"messages": list(messages), "rubric": rubric_text, "_rubric_criteria": []}
    verifier_input_base = middleware._verifier_input
    verifier_run_base = middleware._astream_verifier
    grader_run_base = middleware._astream_final_grader
    captured: dict[str, Any] = {}

    # CANDIDATE OFFLOADING: the only change to MIRA behaviour, restricted to
    # this middleware *instance* in this probe. Never writes into graph state.
    def verifier_input(state_for_verifier: dict[str, Any], iteration: int) -> dict[str, Any]:
        temporary = {**state_for_verifier, "messages": surrogate_messages}
        nested_input = verifier_input_base(temporary, iteration)
        prompt = nested_input["messages"][0].content
        prompt += (
            "\n\nPROBE REQUIREMENT (operational, not part of the quoted transcript): "
            "Every offloaded file mentioned in the transcript MUST be read in its entirety "
            "with the available read_file tool. If read_file reports a next offset, read "
            "the remaining pages until the end. Do not summarize a file in place of "
            "reading it. Do not finish verification until every line has been read."
        )
        nested_input["messages"][0] = HumanMessage(content=prompt)
        captured["verifier_input"] = nested_input
        return nested_input

    async def verifier_run(verifier: Any, nested_input: dict[str, Any], *, config: dict[str, Any], context: Any) -> dict[str, Any]:
        result = await verifier_run_base(verifier, nested_input, config=config, context=context)
        captured["verifier_result"] = result
        evidence = middleware._verification_evidence(result)
        captured["evidence"] = evidence
        report = _complete_tool_pages(evidence, required)
        captured["read_report"] = report
        print("\nACTUAL verifier read_file results:", flush=True)
        for path, item in report.items():
            print(
                f"  {'PASS' if item['complete'] else 'FAIL'} {path}: "
                f"{item['covered_lines']}/{item['total_lines']} source lines; "
                f"{item['read_calls']} tool calls; "
                f"{len(item['mismatched_pages'])} mismatched pages",
                flush=True,
            )
            if item["missing_lines"]:
                print(f"       missing line numbers: {item['missing_lines'][:30]}", flush=True)
            for mismatch in item["mismatched_pages"][:3]:
                print(f"       {mismatch}", flush=True)
        if not all(item["complete"] for item in report.values()):
            raise AssertionError("INCOMPLETE RETRIEVAL: grader intentionally BLOCKED. The verifier didn't read every original line.")
        return result

    async def grader_run(grader: Any, nested_input: dict[str, Any], *, config: dict[str, Any], context: Any) -> dict[str, Any]:
        captured["grader_input"] = nested_input
        expected = captured.get("evidence") or []
        actual_messages = nested_input.get("messages") or []
        actual_ids = {id(x) for x in actual_messages}
        missing = [x for x in expected if id(x) not in actual_ids]
        assert not missing, f"GRADER INPUT LOST {len(missing)} REAL VERIFIER EVIDENCE MESSAGES"
        forwarded_tools = [x for x in expected if isinstance(x, ToolMessage)]
        assert forwarded_tools, "Grader input has NO ToolMessage evidence"
        for tool_msg in forwarded_tools:
            assert tool_msg.content is not None, "Grader input contained an empty tool result"
        print(
            f"PASS full evidence forwarding: {len(expected)} original verifier messages "
            f"({len(forwarded_tools)} untruncated ToolMessage objects) are in grader input",
            flush=True,
        )
        print("Real grader call starting (tools=[]).", flush=True)
        return await grader_run_base(grader, nested_input, config=config, context=context)

    middleware._verifier_input = verifier_input  # type: ignore[method-assign]
    middleware._astream_verifier = verifier_run  # type: ignore[method-assign]
    middleware._astream_final_grader = grader_run  # type: ignore[method-assign]

    try:
        graded = await middleware._ainvoke_grader(state, iteration=0, context=None)
    finally:
        # The actual TemporaryDirectory context owns cleanup after grading,
        # including all errors. The source session is never involved.
        pass
    print("\nREAL structured grader result:", flush=True)
    print(f"  verdict = {graded.result}", flush=True)
    print(f"  criteria = {len(graded.criteria)}", flush=True)
    print("  (Verdict quality is model-dependent; full-evidence plumbing is the test.)", flush=True)
    assert captured.get("grader_input"), "No grader invocation was captured"


def _default_rubric(values: dict[str, Any], target: dict[str, str]) -> str:
    saved = values.get("rubric") or values.get("_active_rubric")
    if isinstance(saved, str) and saved.strip():
        return saved.strip()
    return (
        "Evaluate the latest completed assistant answer against the original user request. "
        "Assess whether the answer contains substantive, coherent work and addresses "
        "the requested topic. Use the available fresh verification evidence, especially "
        "material beyond the standard 4,000-character transcript excerpt."
    )


async def _main(opts: argparse.Namespace) -> int:
    repo = _repo(opts.repo)
    workspace = (opts.workspace or repo).expanduser().resolve()
    if not (workspace / ".mira").is_dir():
        raise RuntimeError(f"Not a MIRA workspace (missing .mira/): {workspace}")
    sys.path.insert(0, str(repo))

    from config.loader import load_config
    from deepagents.backends import FilesystemBackend
    from deepagents.middleware import rubric as stock
    from langchain_core.messages import AIMessage

    if opts.thread_id and not opts.session_id:
        raise RuntimeError("--thread-id requires --session-id")
    config = load_config(workspace)
    candidates = _session_candidates(config, opts.session_id, opts.scan_sessions)
    print("MIRA LIVE RUBRIC OFFLOAD PROBE", flush=True)
    print(f"Repository: {repo}", flush=True)
    print(f"Workspace:  {workspace}", flush=True)
    print(f"DeepAgents per-message rubric limit: {stock._MAX_TRANSCRIPT_CHARS_PER_MESSAGE:,} chars", flush=True)
    print(f"DeepAgents transcript window: {stock._MAX_TRANSCRIPT_MESSAGES} recent messages + first human", flush=True)
    print("Checkpoint access: SQLite online backups only; DeltaChannel messages reconstructed natively", flush=True)

    with tempfile.TemporaryDirectory(prefix="mira-rubric-offload-probe-") as tmp:
        temp = Path(tmp)
        selected: tuple[str, str, list[Any], dict[str, Any]] | None = None
        fallback: tuple[str, str, list[Any], dict[str, Any]] | None = None
        for idx, (session_id, db) in enumerate(candidates):
            cloned_db = temp / f"session-{idx}-backup.db"
            try:
                _copy_db_read_only(db, cloned_db)
                snapshots = await _checkpoint_messages(
                    cloned_db, session_id, requested_thread=opts.thread_id,
                )
            except Exception as exc:
                if opts.session_id:
                    raise
                print(f"  Skip session {session_id}: {type(exc).__name__}: {exc}", flush=True)
                continue
            best: tuple[str, list[Any], dict[str, Any], int] | None = None
            for thread_id, messages, values in snapshots:
                effective = _effective_messages(messages, values)
                large = sum(
                    len(stock._coerce_text(effective[i])) > stock._MAX_TRANSCRIPT_CHARS_PER_MESSAGE
                    for i in _selected_indices(effective)
                )
                if best is None or large > best[3]:
                    best = (thread_id, effective, values, large)
            if best is None:
                print(f"  {session_id}: no nonempty Act/Plan/Goal thread", flush=True)
                continue
            thread_id, effective, values, large = best
            print(f"  {session_id}: thread={thread_id}; messages={len(effective)}; oversized={large}", flush=True)
            candidate = (session_id, thread_id, effective, values)
            if fallback is None:
                fallback = candidate
            if large or (opts.synthetic and opts.session_id):
                selected = candidate
                break
            if opts.session_id:
                break
        if selected is None and opts.synthetic:
            selected = fallback
        if selected is None:
            if fallback is None:
                raise RuntimeError(
                    "No saved Act/Plan/Goal thread with messages found. "
                    "Try --session-id ID --thread-id THREAD, or inspect existing sessions."
                )
            raise RuntimeError(
                "No >4,000-character message found in the stock rubric window "
                "of the selected session(s). Try --session-id ID or --synthetic "
                "for an in-memory diagnostic message."
            )

        session_id, thread_id, effective, values = selected
        print(f"SELECTED session: {session_id} (source checkpoint READ-ONLY)", flush=True)
        print(f"SELECTED thread:  {thread_id}", flush=True)
        print(f"Effective messages: {len(effective)}", flush=True)
        oversized = [i for i in _selected_indices(effective)
                     if len(stock._coerce_text(effective[i])) > stock._MAX_TRANSCRIPT_CHARS_PER_MESSAGE]
        if not oversized and opts.synthetic:
            synthetic = (
                "[PROBE-ONLY synthetic assistant response; not saved to session]\n"
                + "Evidence: section content, bounded transcript demonstration.\n" * 105
                + "PROBE_END_SENTINEL_7F32F4D1\n"
            )
            effective.append(AIMessage(content=synthetic))
            print("No oversized saved message: appended a synthetic message to the IN-MEMORY copy only.", flush=True)
        backend = FilesystemBackend(root_dir=temp / "workspace", virtual_mode=True)
        surrogates, required = _offload(effective, backend, opts.max_total_chars)
        print(f"Offloaded {len(required)} message(s), {sum(map(len, required.values())):,} chars to isolated temp backend:", flush=True)
        for path, content in required.items():
            print(f"  {path}  ({len(content):,} chars; {len(content.splitlines())} lines)", flush=True)
        try:
            if opts.dry_run:
                print("PASS DRY RUN: all files round-tripped through DeepAgents backend; no LLM calls made.", flush=True)
            else:
                rubric = opts.rubric_file.read_text(encoding="utf-8") if opts.rubric_file else _default_rubric(values, required)
                print("Live mode: sending selected saved transcript to configured Rubric model.", flush=True)
                await _live_probe(config, effective, values, backend, required, surrogates, rubric)
        finally:
            cleanup = backend.delete("/offloaded")
            if cleanup.error:
                print(f"WARNING backend cleanup failed: {cleanup.error}; temporary workspace will still be removed.", flush=True)
            else:
                print("PASS cleanup: DeepAgents backend deleted all candidate offload files.", flush=True)
        print("PASS source checkpoint: used only a SQLite read-only backup; original session not invoked.", flush=True)
    print("PASS isolated temp directory destroyed.", flush=True)
    return 0


def main() -> int:
    options = _args()
    try:
        return asyncio.run(_main(options))
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nINTERRUPTED — temporary files are removed by the temp-directory context.", flush=True)
        return 130
    except Exception as exc:
        print(f"\nFAIL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc(limit=3)
        print("Source session was not resumed or updated. Probe temporary files are cleaned up.", file=sys.stderr)
        return 1



# ============================================================================
# Inspector UI runner (uses MIRA's actual Inspector and RubricInspectionProjector)
# ============================================================================


def _ui_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the live Rubric offloading probe inside MIRA's real Textual Inspector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo", type=Path, default=None, help="MIRA source repository, if not current directory")
    parser.add_argument("--workspace", type=Path, default=None, help="MIRA workspace with .mira settings and saved sessions")
    parser.add_argument("--session-id", default=None, help="Saved MIRA session id (source is read-only)")
    parser.add_argument("--thread-id", default=None, help="Exact Act/Plan/Goal thread within --session-id")
    parser.add_argument("--scan-sessions", type=int, default=25, help="Most recent sessions to scan")
    parser.add_argument("--rubric-file", type=Path, default=None, help="Optional rubric text file")
    parser.add_argument("--synthetic", action="store_true", help="Append synthetic in-memory oversized evidence if none is found")
    parser.add_argument("--max-total-chars", type=int, default=250000, help="Safety cap for offloaded evidence")
    parser.add_argument("--tool-output-chars", type=int, default=5000, help="Displayed tool-result preview budget; raw tool evidence stays intact")
    return parser.parse_args()


def inspector_main() -> int:
    opts = _ui_args()
    repo = _repo(opts.repo)
    workspace = (opts.workspace or repo).expanduser().resolve()
    if not (workspace / ".mira").is_dir():
        raise RuntimeError(f"Not a MIRA workspace (missing .mira/): {workspace}")
    if opts.thread_id and not opts.session_id:
        raise RuntimeError("--thread-id requires --session-id")
    sys.path.insert(0, str(repo))

    import threading
    from textual import on
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.widgets import Button, Footer, Header, Static

    from core.execution.inspection.live import LiveInspectionStore
    from core.execution.inspection.rubric import RubricInspectionProjector, rubric_inspection_id
    from ui.textual.widgets.inspector import Inspector
    from ui.textual.platform.windows.input import driver_class_for_platform

    RUN_ID = "mira-offload-probe"
    VERIFIER_ID = rubric_inspection_id(RUN_ID, 0, "verifier")
    GRADER_ID = rubric_inspection_id(RUN_ID, 0, "grader")
    styles = [
        str(repo / "ui" / "textual" / "styles" / "mira.tcss"),
        str(repo / "ui" / "textual" / "styles" / "workspace.tcss"),
    ]

    class RubricInspectorProbe(App[None]):
        TITLE = "MIRA · Rubric Offload Live Probe"
        CSS_PATH = [p for p in styles if Path(p).is_file()]
        CSS = """
        Screen { layout: vertical; }
        #probe-top { height: 3; width: 100%; align: left middle; padding: 0 1; }
        #probe-top Button { margin-right: 1; min-width: 13; }
        #probe-title { width: 1fr; content-align: right middle; text-style: bold; }
        #probe-inspector { height: 1fr; width: 100%; }
        #probe-status { height: 3; padding: 0 2; content-align: left middle; }
        #probe-coverage { height: 3; padding: 0 2; content-align: left middle; }
        """
        BINDINGS = [
            Binding("1", "verifier", "Verifier"),
            Binding("2", "grader", "Grader"),
            Binding("ctrl+q", "quit", "Quit"),
        ]

        def __init__(self) -> None:
            super().__init__(driver_class=driver_class_for_platform(sys.platform))
            self.live_inspections = LiveInspectionStore()
            self.projection = RubricInspectionProjector(self)
            self.active_phase = "verifier"
            self.ui_thread: int | None = None
            self.done = False
            self.completed = False
            self.last_error = ""
            self.stats: dict[str, Any] = {}

        def compose(self) -> ComposeResult:
            yield Header(show_clock=False)
            with Horizontal(id="probe-top"):
                yield Button("1 · Verifier", id="probe-verifier", variant="primary")
                yield Button("2 · Grader", id="probe-grader")
                yield Static("Read-only · live model calls", id="probe-title")
            yield Inspector(
                self.live_inspections,
                tool_output_chars=max(240, opts.tool_output_chars),
                embedded=True,
                id="probe-inspector",
            )
            yield Static("Preparing read-only checkpoint snapshot...", id="probe-status")
            yield Static("Coverage: not yet evaluated", id="probe-coverage")
            yield Footer()

        def on_mount(self) -> None:
            self.ui_thread = threading.get_ident()
            self.call_after_refresh(self._launch)

        def _launch(self) -> None:
            self.run_worker(self._execute_probe(), name="rubric-offload-probe", exclusive=True)

        @on(Button.Pressed, "#probe-verifier")
        def verifier_pressed(self, event: Button.Pressed) -> None:
            event.stop()
            self.action_verifier()

        @on(Button.Pressed, "#probe-grader")
        def grader_pressed(self, event: Button.Pressed) -> None:
            event.stop()
            self.action_grader()

        def action_verifier(self) -> None:
            self._show_phase("verifier")

        def action_grader(self) -> None:
            self._show_phase("grader")

        def _show_phase(self, phase: str) -> None:
            self.active_phase = phase
            inspection_id = VERIFIER_ID if phase == "verifier" else GRADER_ID
            self.query_one("#probe-verifier", Button).variant = "primary" if phase == "verifier" else "default"
            self.query_one("#probe-grader", Button).variant = "primary" if phase == "grader" else "default"
            inspector = self.query_one("#probe-inspector", Inspector)
            if not inspector.open(inspection_id):
                self._status(f"{phase.title()} has not started yet. Select 1 or 2 after the phase begins.")

        def _status(self, message: str) -> None:
            self.query_one("#probe-status", Static).update(message)

        def _coverage(self, message: str) -> None:
            self.query_one("#probe-coverage", Static).update(message)

        def _event_writer(self, value: dict[str, Any]) -> None:
            # MIRA's nested tool events may originate in a worker thread.
            # All Textual widgets and LiveInspectionStore subscriptions are
            # mutated on the UI thread only.
            if threading.get_ident() == self.ui_thread:
                self._apply_event(value)
            else:
                self.call_from_thread(self._apply_event, value)

        def _apply_event(self, value: dict[str, Any]) -> None:
            kind = str(value.get("type") or "")
            if kind == "rubric_verification_start":
                self.projection.start(value, "verifier")
                if self.active_phase == "verifier":
                    self._show_phase("verifier")
                self._status("Verifier running · real model + read_file tool calls")
            elif kind == "rubric_grading_start":
                self.projection.start(value, "grader")
                if self.active_phase == "grader":
                    self._show_phase("grader")
                self._status("Grader running · tool-free structured output")
            elif kind == "rubric_inspection_delta":
                self.projection.append_delta(value)
            elif kind == "rubric_tool_start":
                self.projection.tool_call(value)
            elif kind == "rubric_tool_end":
                self.projection.tool_completion(value)
            elif kind == "rubric_verification_end":
                self.projection.finish(value, "verifier")
                if not value.get("succeeded", False):
                    self._status("Verifier failed · select 1 to inspect the actual error")
            elif kind == "rubric_grading_end":
                self.projection.finish(value, "grader")
                if not value.get("succeeded", False):
                    self._status("Grader failed · select 2 to inspect the actual error")

        async def _execute_probe(self) -> None:
            import tempfile
            from config.loader import load_config
            from deepagents.backends import FilesystemBackend
            from deepagents.middleware import rubric as stock
            from deepagents.middleware.filesystem import FilesystemMiddleware
            from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
            from agent.llm import get_llm, get_rubric_model_name
            from agent.rubric.middleware import MiraRubricMiddleware, _RUBRIC_RUN
            from config.settings import RUBRIC_MODEL

            try:
                config = load_config(workspace)
                candidates = _session_candidates(config, opts.session_id, opts.scan_sessions)
                with tempfile.TemporaryDirectory(prefix="mira-rubric-inspector-probe-") as tmp:
                    temp = Path(tmp)
                    selected = None
                    fallback = None
                    for idx, (sid, db) in enumerate(candidates):
                        self._status(f"Scanning checkpoint {idx+1}/{len(candidates)} · {sid}")
                        backup = temp / f"session-{idx}-backup.db"
                        try:
                            _copy_db_read_only(db, backup)
                            snapshots = await _checkpoint_messages(backup, sid, requested_thread=opts.thread_id)
                        except Exception:
                            if opts.session_id:
                                raise
                            continue
                        best = None
                        for tid, messages, values in snapshots:
                            effective = _effective_messages(messages, values)
                            count = sum(
                                len(stock._coerce_text(effective[i])) > stock._MAX_TRANSCRIPT_CHARS_PER_MESSAGE
                                for i in _selected_indices(effective)
                            )
                            if best is None or count > best[3]:
                                best = (tid, effective, values, count)
                        if best is None:
                            continue
                        tid, effective, values, count = best
                        row = (sid, tid, effective, values)
                        if fallback is None:
                            fallback = row
                        if count or (opts.synthetic and opts.session_id):
                            selected = row
                            break
                        if opts.session_id:
                            break
                    if selected is None and opts.synthetic:
                        selected = fallback
                    if selected is None:
                        raise RuntimeError(
                            "No eligible >4,000-char message found in the stock Rubric window. "
                            "Try --session-id or --synthetic."
                        )

                    sid, tid, effective, values = selected
                    large = [i for i in _selected_indices(effective)
                             if len(stock._coerce_text(effective[i])) > stock._MAX_TRANSCRIPT_CHARS_PER_MESSAGE]
                    if not large and opts.synthetic:
                        effective.append(AIMessage(
                            content="[PROBE-ONLY synthetic assistant evidence]\n"
                            + "Verified line content from the session.\n" * 230
                            + "PROBE_END_SENTINEL_7F32F4D1\n"
                        ))
                    backend = FilesystemBackend(root_dir=temp / "workspace", virtual_mode=True)
                    surrogates, required = _offload(effective, backend, opts.max_total_chars)
                    self.stats = {"session": sid, "thread": tid, "required": required}
                    self._status(
                        f"Session {sid} · {len(required)} offloaded message(s) · "
                        f"{sum(map(len, required.values())):,} chars · "
                        f"Rubric model: {get_rubric_model_name(config)}"
                    )
                    self._coverage("Offloaded: " + ", ".join(f"{path} ({len(value):,} chars)" for path, value in required.items()))

                    try:
                        rubric_text = opts.rubric_file.read_text(encoding="utf-8") if opts.rubric_file else _default_rubric(values, required)
                        model = get_llm(config, role=RUBRIC_MODEL)
                        fs = FilesystemMiddleware(backend=backend, tools=["read_file"])
                        middleware = MiraRubricMiddleware(
                            model=model,
                            verifier_tools=[],
                            verifier_middleware=[fs],
                            max_iterations=1,
                        )
                        state = {"messages": list(effective), "rubric": rubric_text, "_rubric_criteria": []}
                        input_base = middleware._verifier_input
                        verify_base = middleware._astream_verifier
                        grade_base = middleware._astream_final_grader
                        evidence_holder: dict[str, Any] = {}

                        def verifier_input(actual_state: dict[str, Any], iteration: int) -> dict[str, Any]:
                            safe_state = {**actual_state, "messages": surrogates}
                            nested_input = input_base(safe_state, iteration)
                            prompt = nested_input["messages"][0].content
                            nested_input["messages"][0] = HumanMessage(content=prompt + (
                                "\n\nPROBE REQUIREMENT (operational; not part of the transcript): "
                                "Read EVERY referenced /offloaded/ file completely using read_file. "
                                "If the tool shows a next offset, read all remaining pages. "
                                "Do not finish until every file has been inspected in full."
                            ))
                            return nested_input

                        async def verifier_run(
                            verifier: Any, nested_input: dict[str, Any], *, config: dict[str, Any], context: Any
                        ) -> dict[str, Any]:
                            result = await verify_base(verifier, nested_input, config=config, context=context)
                            evidence = middleware._verification_evidence(result)
                            evidence_holder["evidence"] = evidence
                            coverage = _complete_tool_pages(evidence, required)
                            evidence_holder["coverage"] = coverage
                            summary = "; ".join(
                                f"{Path(path).name}: {item['covered_lines']}/{item['total_lines']} lines"
                                for path, item in coverage.items()
                            )
                            self._coverage("Actual Verifier coverage · " + summary)
                            if not all(item["complete"] for item in coverage.values()):
                                raise AssertionError(
                                    "INCOMPLETE RETRIEVAL: grader intentionally BLOCKED. " + summary
                                )
                            return result

                        async def grader_run(
                            grader: Any, nested_input: dict[str, Any], *, config: dict[str, Any], context: Any
                        ) -> dict[str, Any]:
                            expected = evidence_holder.get("evidence") or []
                            actual_ids = {id(item) for item in nested_input.get("messages") or []}
                            lost = [item for item in expected if id(item) not in actual_ids]
                            if lost:
                                raise AssertionError(f"GRADER INPUT LOST {len(lost)} REAL verifier evidence messages")
                            count = sum(isinstance(item, ToolMessage) for item in expected)
                            if not count:
                                raise AssertionError("Grader input has no real ToolMessages")
                            self._coverage(
                                f"PASS Verifier coverage and forwarding · {count} original ToolMessage(s) in Grader input"
                            )
                            return await grade_base(grader, nested_input, config=config, context=context)

                        middleware._verifier_input = verifier_input
                        middleware._astream_verifier = verifier_run
                        middleware._astream_final_grader = grader_run
                        token = _RUBRIC_RUN.set((RUN_ID, 0, self._event_writer))
                        try:
                            graded = await middleware._ainvoke_grader(state, iteration=0, context=None)
                            self.completed = True
                            self._status(
                                f"PASS probe execution · Grader verdict: {graded.result} · "
                                "Press 1/2 to inspect actual Verifier/Grader events"
                            )
                        finally:
                            _RUBRIC_RUN.reset(token)
                    finally:
                        cleanup = backend.delete("/offloaded")
                        if cleanup.error:
                            self._coverage(f"WARNING: explicit cleanup failed ({cleanup.error}); temp directory still removed")
                self.done = True
            except asyncio.CancelledError:
                self._status("Probe cancelled; original session unchanged")
                raise
            except Exception as exc:
                self.done = True
                self.last_error = f"{type(exc).__name__}: {exc}"
                self._status(f"FAIL: {self.last_error} · original session unchanged · press 1/2 to inspect")
                # A setup failure occurs before any native Rubric events. Show it
                # as a diagnostic error (not a fake grader/tool result).
                if self.live_inspections.get(VERIFIER_ID) is None:
                    self.live_inspections.start(
                        VERIFIER_ID, "Probe setup", "Diagnostic probe initialization", inspection_type="Rubrics"
                    )
                    self.live_inspections.finish(VERIFIER_ID, status="ERROR", error=self.last_error)
                    if self.active_phase == "verifier":
                        self._show_phase("verifier")
                if self.live_inspections.get(GRADER_ID) is None:
                    self._coverage("Grader was not invoked; no verdict is available")

    app = RubricInspectorProbe()
    app.run()
    return 0 if app.completed else 1


if __name__ == "__main__":
    try:
        sys.exit(inspector_main())
    except Exception as exc:
        print(f"Probe could not start: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc(limit=3)
        sys.exit(1)
