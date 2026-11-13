# Transient mode transition verification

Verified on 2026-11-13 from main `a977cf149f209bfa1c704de32ab94aaec7f13cde`.
Installed stack: DeepAgents 0.7.32, LangChain 1.4.7, LangGraph 1.2.18.

## Implementation and changed files

Production diff: **109 added / 1 removed lines**, net **108** (Git line counts,
including comments and blank lines). The standalone middleware is 68 lines.

- `agent/middleware/mode_transition.py`: the sole new middleware class; sync and
  async request-only insertion, stable notice identity, and native compaction handling.
- `agent/middleware/builder.py`: registration for both agent modes after summarization.
- `agent/execution/context.py` and `agent/planning/tool_context.py`: optional trusted
  transition metadata in existing runtime contexts.
- `core/application/app.py`: queue only actual mode changes; latest switch wins.
- `core/application/session.py`: durable pending/active binding and recovery,
  including retained Plan execution and revision transitions.
- `core/application/artifacts.py`: queue Act when implementing from Plan.
- `core/execution/turns.py`: bind the first real user message, supply runtime context,
  and consume the binding on completion without adding a graph input message.
- `tests/test_mode_transition.py`: deterministic effective-model and persistence checks.
- `tests/test_conversation_handoff.py`: account for the transient model notice while
  retaining existing persisted-history expectations.
- `tests/probes/probe_mode_transition.py`: deterministic runner and optional live probe.
- `tests/probes/mode_transition_results.md`: this verification record.
- `tests/manual/prompts.md` and `ARCHITECTURE_DECISIONS.md`: manual checks and rationale.

## Deterministic results

**380 distinct focused tests passed** across these runs (395 executions because
the response-status tests ran twice):

| Run | Tests | Result |
| --- | ---: | --- |
| `python tests/probes/probe_mode_transition.py` | 13 | Pass |
| Compaction, Plan mode, execution context, runner suites | 271 | Pass |
| Conversation handoff, current Plan/Goal, response-status suites | 65 | Pass |
| Inline `/plan` TUI route, response-status, Rubric middleware | 46 | Pass |

All checks used `conda run -n mira`. The recording model observes effective
provider messages through real MIRA orchestration and SQLite checkpoints.
Coverage includes no-op switches, queued/inline prompts, latest-switch precedence,
Plan/Act roundtrips, tool continuations, approval resume, cancellation, incomplete
turn continuation, cold restart, concurrent sessions, prepared messages, retained
Plan resume/revision, and completion before the semantic clear is saved.

Every recorded checkpoint's message history and effective handoff projection is
checked for absent notices. Session JSON and native compaction archives are also
checked. A native summarization test confirms that an active user-message anchor
can be compacted away: the notice then precedes the retained continuation, keeping
the call/result pair intact. No synthetic message is inserted into graph input or
checkpoint message state. Only mode, real message ID, and thread ID are durable.

Working and staged `git diff --check` passed. The full repository suite was not run.

## Live Gemma 4 12B

Command: `conda run -n mira --no-capture-output python tests/probes/probe_mode_transition.py --live-only`

The configured `lmstudio-gemma-4-12b` profile was available at the local endpoint.
The real MIRA session used a disposable workspace with `palindrome.py` and normal
write approval. Model callbacks captured the effective input messages.

| Transition | Model calls | Tool | File changed |
| --- | ---: | --- | --- |
| Act to Plan | 2 | `read_file` | No |
| Plan to Act | 2 | `write_file` | Yes |
| Act to Plan | 2 | `read_file` | No |
| Plan to Act | 2 | `write_file` | Yes |
| Act to Plan | 2 | `read_file` | No |

All ten model invocations contained exactly one current-mode notice immediately
before the real prompt, including the invocations after tools. A subsequent turn
contained no notice. The persistence/handoff checks also passed on this live session.
The current-checkout `python -m cli.main --workspace <disposable> -p ...` smoke
returned exit code 0 and correctly explained the palindrome function.

No mode confusion was observed in this sample. This is one diagnostic sequence,
without a baseline comparison; it does not establish general model reliability
or prove that the notice improves compliance. No unresolved implementation issue
was found in the tested lifecycles.
