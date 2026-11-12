#!/usr/bin/env python
"""Live MIRA rubric evidence-offloading probe (no production source edits).

Run from the MIRA repository with its configured Python/Conda environment:

    python tests/probes/mira_rubric_live_probe_fixed.py
    python tests/probes/mira_rubric_live_probe_fixed.py --session-id SESSION_ID
    python tests/probes/mira_rubric_live_probe_fixed.py --synthetic
    python tests/probes/mira_rubric_live_probe_fixed.py --session-id SESSION_ID --thread-id THREAD_ID
    python tests/probes/mira_rubric_live_probe_fixed.py --dry-run

Alternatively save anywhere and pass --repo D:\\Projects\\mira.

- Sources the real conversation from read-only SQLite BACKUPS of recent
  (or specified) saved sessions, reconstructing LangGraph DeltaChannel state.
  Chooses a session/thread with an actual oversized message by default.
- Uses MIRA's configured real RUBRIC model and its existing verifier and grader.
- Exercises production offloading and its bounded verifier read-coverage guard.
- Uses the production native StateBackend reader; temporary disk copies verify integrity independently.
- Requires the verifier to retrieve EVERY line of EVERY offloaded file, proving
  the source text, not just the header or last-line marker, reached ToolMessages.
- Proves those exact ToolMessages are in the ACTUAL grader invocation.
- Does not call the normal main agent, consume rubric iterations, or update
  the source checkpoint, workspace, config, or sessions.
- Deletes all temp evidence after the grader finishes (also on exceptions).

WARNING: Live mode transmits the selected saved conversation, rubric, and
retrieved evidence to the model endpoint in MIRA's configuration. This may
cost tokens and should be used only with an endpoint trusted for that data.

The offloader, manifest, read guard, verifier, and final grader all use production
code. No original report text is printed to stdout.
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
from contextlib import closing
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
    ap.add_argument("--synthetic", action="store_true", help="Use synthetic evidence only, or append it to an explicit --session-id if needed")
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
    with closing(sqlite3.connect(src.resolve().as_uri() + "?mode=ro", uri=True, timeout=15)) as connection:
        with closing(sqlite3.connect(dest)) as cloned:
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
    from agent.rubric.offload import selected_indices
    return selected_indices(messages)


def _offload(messages: list[Any], backend: Any, max_total_chars: int) -> tuple[list[Any], dict[str, str]]:
    """Inspect production offloading; independently round-trip its source text."""
    from agent.rubric.offload import offload_transcript

    modified, files, manifest = offload_transcript(messages)
    needed = {item["path"]: files[item["path"]]["content"] for item in manifest}
    total = sum(map(len, needed.values()))
    if not needed:
        raise RuntimeError("No oversized messages in the rubric window. Try --synthetic.")
    if total > max_total_chars:
        raise RuntimeError(f"Would offload {total:,} characters, exceeding cap {max_total_chars:,}.")
    for path, content in needed.items():
        result = backend.write(path, content)
        if result.error:
            raise RuntimeError(result.error)
        check = backend.read(path, offset=0, limit=len(content.splitlines()) + 1)
        normalized = content.replace("\r\n", "\n").replace("\r", "\n")
        if check.error or check.file_data is None or check.file_data["content"].rstrip("\n") != normalized.rstrip("\n"):
            raise RuntimeError(f"Evidence round-trip failed: {path}")
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
                if call.get("name") == "read_rubric_evidence" and isinstance(call.get("args"), dict):
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
        source_lines = source.splitlines()
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


async def _live_probe(config: dict[str, Any], messages: list[Any], required: dict[str, str], rubric_text: str) -> None:
    from langchain_core.messages import ToolMessage
    from agent.llm import get_llm, get_rubric_model_name
    from agent.rubric.middleware import MiraRubricMiddleware
    from config.settings import RUBRIC_MODEL

    print(f"Rubric LLM: {get_rubric_model_name(config)}", flush=True)
    model = get_llm(config, role=RUBRIC_MODEL)
    middleware = MiraRubricMiddleware(
        model=model,
        verifier_tools=[],
        verifier_middleware=[],
        max_iterations=1,
    )
    state: dict[str, Any] = {"messages": list(messages), "rubric": rubric_text, "_rubric_criteria": []}
    verifier_input_base = middleware._verifier_input
    verifier_run_base = middleware._astream_verifier
    grader_run_base = middleware._astream_final_grader
    captured: dict[str, Any] = {}

    # Observe production input without changing its prompt, files, or manifest.
    def verifier_input(state_for_verifier: dict[str, Any], iteration: int) -> dict[str, Any]:
        nested_input = verifier_input_base(state_for_verifier, iteration)
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
        missing = [x for x in expected if (
            id(x) not in actual_ids if isinstance(x, ToolMessage) else x not in actual_messages
        )]
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

    graded = await middleware._ainvoke_grader(state, iteration=0, context=None)
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
    candidates = [] if opts.synthetic and not opts.session_id else _session_candidates(config, opts.session_id, opts.scan_sessions)
    print("MIRA LIVE RUBRIC OFFLOAD PROBE", flush=True)
    print(f"Repository: {repo}", flush=True)
    print(f"Workspace:  {workspace}", flush=True)
    print(f"DeepAgents per-message rubric limit: {stock._MAX_TRANSCRIPT_CHARS_PER_MESSAGE:,} chars", flush=True)
    print(f"DeepAgents transcript window: {stock._MAX_TRANSCRIPT_MESSAGES} recent messages + first human", flush=True)
    print("Checkpoint access: SQLite online backups only; DeltaChannel messages reconstructed natively", flush=True)

    with tempfile.TemporaryDirectory(prefix="mira-rubric-offload-probe-") as tmp:
        temp = Path(tmp)
        selected: tuple[str, str, list[Any], dict[str, Any]] | None = (
            ("synthetic", "synthetic", [], {}) if opts.synthetic and not opts.session_id else None
        )
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
            print("Using synthetic evidence in memory only.", flush=True)
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
                print("Live mode: sending selected evidence to configured Rubric model.", flush=True)
                await _live_probe(config, effective, required, rubric)
        finally:
            cleanup = backend.delete("/rubric_offload")
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


if __name__ == "__main__":
    sys.exit(main())
