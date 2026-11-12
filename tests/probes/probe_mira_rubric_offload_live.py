#!/usr/bin/env python
"""Live MIRA rubric evidence-offloading probe (no production source edits).

Run from the MIRA repository with its configured Python/Conda environment:

    python tests/probes/probe_mira_rubric_offload_live.py
    python tests/probes/probe_mira_rubric_offload_live.py --session-id SESSION_ID
    python tests/probes/probe_mira_rubric_offload_live.py --synthetic
    python tests/probes/probe_mira_rubric_offload_live.py --dry-run

Alternatively save anywhere and pass --repo D:\\Projects\\mira.

- Sources the real conversation from a read-only SQLite BACKUP of the latest
  (or specified) persisted session checkpoint.
- Uses MIRA's configured real RUBRIC model and its existing verifier and grader.
- Implements candidate offloading only inside this one probe invocation.
- Uses DeepAgents FilesystemBackend + FilesystemMiddleware.read_file.
- Requires the verifier to retrieve EVERY line of EVERY offloaded file, proving
  the source text, not just the header or last-line marker, reached ToolMessages.
- Proves those exact ToolMessages are in the ACTUAL grader invocation.
- Does not call the normal main agent, consume rubric iterations, or update
  the source checkpoint, workspace, config, or sessions.
- Deletes all temp evidence after the grader finishes (also on exceptions).

WARNING: Live mode transmits the selected saved conversation, rubric, and
retrieved evidence to the model endpoint in MIRA's configuration. This may
cost tokens and should be used only with an endpoint trusted for that data.

This is a CANDIDATE probe; it does not claim to test a feature already
implemented in MIRA. Use it to establish whether the approach works before
changing production code. No original report text is printed to stdout.
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
    ap.add_argument("--session-id", default=None, help="Saved session id (default: most recently updated saved session)")
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


def _load_session_dir(config: dict[str, Any], session_id: str | None) -> tuple[str, Path]:
    from session.store import SessionStore

    store = SessionStore(Path(config["session_dir"]).expanduser().resolve())
    if session_id:
        meta = store.path(session_id)
    else:
        meta = store.latest()
        if meta is None:
            raise RuntimeError(f"No saved sessions in {store.root}; specify --workspace or --session-id")
        session_id = meta.parent.name
    if not meta.is_file():
        raise RuntimeError(f"Session metadata missing: {meta}")
    db = store.database_path(str(session_id))
    if not db.is_file():
        raise RuntimeError(f"Checkpoint database missing: {db}")
    return str(session_id), db


def _copy_db_read_only(src: Path, dest: Path) -> None:
    """Online SQLite backup: never open source read/write, safe while MIRA runs."""
    with sqlite3.connect(src.resolve().as_uri() + "?mode=ro", uri=True, timeout=15) as connection:
        with sqlite3.connect(dest) as cloned:
            connection.backup(cloned, pages=128, sleep=0.05)


async def _checkpoint_messages(db: Path, thread_id: str) -> tuple[list[Any], dict[str, Any]]:
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    async with aiosqlite.connect(str(db)) as connection:
        saver = AsyncSqliteSaver(connection)
        checkpoint_tuple = await saver.aget_tuple({"configurable": {"thread_id": thread_id}})
        if checkpoint_tuple is None:
            raise RuntimeError(f"No LangGraph checkpoint for thread {thread_id!r}")
        values = dict(checkpoint_tuple.checkpoint.get("channel_values") or {})
        messages = list(values.get("messages") or [])
        if not messages:
            raise RuntimeError("Session checkpoint has no messages. Choose a completed Act thread with chat history.")
        return messages, values


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

    config = load_config(workspace)
    session_id, db = _load_session_dir(config, opts.session_id)
    print("MIRA LIVE RUBRIC OFFLOAD PROBE", flush=True)
    print(f"Repository: {repo}", flush=True)
    print(f"Workspace:  {workspace}", flush=True)
    print(f"Session:    {session_id} (source checkpoint READ-ONLY)", flush=True)
    print(f"DeepAgents per-message rubric limit: {stock._MAX_TRANSCRIPT_CHARS_PER_MESSAGE:,} chars", flush=True)
    print(f"DeepAgents transcript window: {stock._MAX_TRANSCRIPT_MESSAGES} recent messages + first human", flush=True)

    with tempfile.TemporaryDirectory(prefix="mira-rubric-offload-probe-") as tmp:
        temp = Path(tmp)
        cloned_db = temp / "checkpoints-backup.db"
        _copy_db_read_only(db, cloned_db)
        original, values = await _checkpoint_messages(cloned_db, session_id)
        effective = _effective_messages(original, values)
        print(f"Checkpoint messages: {len(original)}; effective messages: {len(effective)}", flush=True)
        # Synthetic mode is a fallback for workspaces without a large report;
        # it is added only to the in-memory probe state.
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
        for path, text in required.items():
            print(f"  {path}  ({len(text):,} chars; {len(text.splitlines())} lines)", flush=True)
        try:
            if opts.dry_run:
                print("PASS DRY RUN: all files round-tripped through DeepAgents backend; no LLM calls made.", flush=True)
            else:
                rubric = opts.rubric_file.read_text(encoding="utf-8") if opts.rubric_file else _default_rubric(values, required)
                print("Live mode: sending selected saved transcript to configured Rubric model.", flush=True)
                await _live_probe(config, effective, values, backend, required, surrogates, rubric)
        finally:
            # Delete with the native backend even if verifier or grader fails.
            # The TemporaryDirectory is a second safety net, not the only cleanup.
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


if __name__ == "__main__":
    sys.exit(main())
