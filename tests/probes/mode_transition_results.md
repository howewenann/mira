# Transient mode transition verification

**Supplemental finding:** the completed five-transition A/B pair showed **0
explicit incorrect current-mode claims in either arm**, so it did not measure a
reduction in that failure rate. It did show fewer repeated tool calls with the
notice. The harder exact-prompt regression replay reproduced the wrong-mode
inference without the notice, but the experimental reasoning still contained
historical-mode confusion and repeated failed edits. The notice is useful
guidance; these runs do **not** establish a reliable cure for mode confusion.

Verified on 2026-11-13 from main `a977cf149f209bfa1c704de32ab94aaec7f13cde`.
Installed stack: DeepAgents 0.7.32, LangChain 1.4.7, LangGraph 1.2.18.

## Original implementation and changed files (c24ec70, unchanged)

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

## Original live delivery-only check

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

This original run captured inputs, tools and final replies, but did not inspect
recorded reasoning or include a baseline. It established notice delivery only.
Its successful tool calls and final replies do **not** establish correct mode
awareness. The reasoning-aware verification below supersedes that interpretation.

## Reasoning-aware A/B method

Supplemental verification uses unchanged production commit `c24ec70`. Production
middleware, runner, reasoning stream, recorder, and settings implementation are
unchanged. The original 13 deterministic transition tests are retained.

Run:

```text
conda run -n mira --no-capture-output python tests/probes/probe_mode_transition.py --live-only
conda run -n mira python -m unittest tests.test_mode_transition_probe -q
```

The probe uses local `lmstudio-gemma-4-12b` / `google/gemma-4-12b`, temperature 0,
equivalent disposable workspaces, independent sessions/checkpoints, identical
normalized settings and the same ordered prompts in each pair. MIRA's existing repeated-call
guard is enabled in both arms (3 executions, 2 recovery errors), as the original
regression also contained guard errors. Rubric is at its fresh-workspace default,
disabled; its implementation is not patched. No reference transcript is seeded
into either model history. Two scenarios are retained:

- **Inspection**, the original probe's read/append scenario, seeds the same
  two-line `palindrome.py` in each workspace. Each Plan turn asks:
  `Read /palindrome.py and briefly explain it. This is read-only inspection; do not create a formal Plan.`
  Each Act turn asks to append `# Act transition 1` or `# Act transition 3`.
  Both arms run all five transitions, including both returns to Plan. This is
  the default scenario; select it explicitly with `--scenario inspection`.
- **Regression**, selected with `--scenario regression`, starts from empty
  projects. Its first three prompts are verbatim from the reference, extended
  to the following attempted sequence. A timed-out baseline is retained as a
  failed replay, not counted as a completed five-transition comparison.

| Transition | Requested mode | Prompt |
| --- | --- | --- |
| 1 | Plan | `dont make a plan, just write a palindrome function to the main directory as a python file` |
| 2 | Act | `write the file now` |
| 3, first return to Plan | Plan | `pls edit the file to include more tests inside. dont make a plan` |
| 4 | Act | `edit the file now to include more tests inside. dont make a plan` |
| 5, second return to Plan | Plan | Same prompt as transition 3 |

Baseline patches only `ModeTransitionMiddleware._with_notice` to return its
unmodified request, within the probe's baseline context. Experiment uses normal
injection. Both use the same model factory observation and frontend policy:
approve writes, reject unsolicited formal Plans with Clear (never Implement),
answer clarification requests with the same example-selection policy, and report
the actual retained artifact on display requests. Frontend decisions are saved.
If a display leaves a native tool interrupt pending, the probe supplies the
actual display answer with `Command(resume=...)` through MIRA's normal runner;
it never overwrites graph history, silently resets a session, or invents a model
response. Display continuation is bounded and retained in the evidence.

The capture path is MIRA's `consume_messages` in
`core/execution/streams/messages.py` -> `SessionEventEmitter.reasoning_delta` ->
`SessionRecorder` in `session/recorder.py` -> persisted `reasoning` event. An
observing frontend links each saved event ID and recorder mode to the native
message ID. Model callbacks join that response ID to its effective model input,
including returned tool results. The probe asserts that captured events equal
the saved session's events. It does not synthesize reasoning or treat final
replies as reasoning. An invocation with no recorded reasoning is labelled
`unavailable`; a missing response identity is labelled `unmapped` separately.

Analysis reviews the full recorded reasoning after every transition. An
incorrect current-mode claim means an affirmative assertion inconsistent with
the event's `planning`/`action` metadata. Historical quotations, hypothetical
alternatives and the word "Plan" in an ordinary task outline are not themselves
incorrect claims. Tentative alternatives and self-corrections are discussed
separately. Tool repetition means distinct native call IDs with identical names
and all arguments within one turn; stream replays of one call ID are separated.
Repeated paragraphs are normalized for whitespace and case (minimum 80 characters).
These automated candidates aid inspection; they are not compliance verdicts.

## Original regression reference

The user identified `session(6).json` as
`.mira/_sessions/20261113-085837+0800-5bdca89f/session.json`.
Its SHA-256 is
`bab348b9a4cb9578eaaf154ac313681ef1dd829d9e331fd50ae75bcbd9dcd7e2`.
The source file was read only and remains unchanged. A selected evidence extract,
including all three recorded reasoning events and original repetition counts,
is retained in [reference.json](mode_transition_evidence/reference.json).

- **R2**, metadata `planning`: "I am currently in Plan mode." This matches the
  selected mode, though its long reasoning repeatedly revisits the restriction.
- **R5**, metadata `action`: "I am currently in Plan mode, and I cannot perform
  actions like `write_file`." This is an explicit incorrect current-mode claim,
  despite the successful write at tool events **6/7** and final reply **8**.
- **R10**, metadata `planning`: "I am currently in **Act mode** (since I just
  performed a write action)." This is the opposite mismatch on return to Plan.
  Read **11** already returned the file at **12**, but identical `/palindrome.py`
  reads continued: **35 distinct read requests** in that turn, including **8
  blocked requests**. The same missing-skill validation was requested 9 times,
  plus one different path. Event **101** records interruption, not completion.

The reference export has no invocation-ID linkage, so the absence of additional
reasoning events after **R10** cannot be assigned to particular provider calls.

## Completed five-transition comparison

The full original read/append scenario completed in both arms. Raw recorder
events, assistant messages, tool calls/results, effective-input observations,
event-to-message links, settings and per-transition file snapshots are retained:
[inspection-baseline.json](mode_transition_evidence/inspection-baseline.json) and
[inspection-experiment.json](mode_transition_evidence/inspection-experiment.json).
Their settings, seed file and prompt sequences were checked for equality.
Here **B** and **E** identify those files; numbers are recorder event IDs.

| Transition / recorded mode | Baseline reasoning IDs; available invocations | Experiment reasoning IDs; available invocations | Reasoning finding |
| --- | --- | --- | --- |
| 1 / `planning` | B2; 1/2 | E2; 1/2 | Both explicitly identify Plan as read-only. |
| 2 / `action` | B7; 1/2 | E7; 1/2 | Both deliberate at length over edit versus replace. Neither explicitly claims a current mode; no incorrect claim is visible. |
| 3 / `planning` | B12, B15; 2/2 | E12, E15; 2/2 | Baseline has no explicit mode claim. Experiment explicitly identifies current Plan context. Both recorded continuations discuss the returned file. |
| 4 / `action` | B18, B21, B24, B27, B30, B33; 6/7 | E18, E21; 2/2 | Baseline repeatedly reasons from stale file state and repeats edits. Experiment recognizes that the requested append is already complete. Neither makes an incorrect current-mode claim. |
| 5 / `planning`, second return to Plan | B38, B41; 2/2 | E24, E27; 2/2 | Baseline has no explicit current-mode claim. Experiment explicitly says current Plan is read-only. Both discuss the returned file without repeating the read. |

Relevant short excerpts:

- **B2**, `planning`: "I am in Plan mode, which is read-only."
  **E2**, `planning`: "I am in PLAN mode, which is read-only."
- **B12**, `planning`: "I have already modified this file in the previous turn";
  it then requests a read. This is a correct historical observation, **not** an
  Act-mode claim. **E12**, `planning`: "in the context of being in PLAN mode
  (read-only)."
- **B21**, `action`, after successful edit result **B20**, still describes the
  pre-edit file and says "I will replace `# Act transition 1`", proposing the
  same append again. The identical new call **B22**
  succeeds at **B23**, adding the comment twice. This is direct evidence of poor
  result/state tracking, not evidence of an incorrect mode claim.
- **B27**, `action`, recognizes the duplication: "It seems I ended up with two
  `# Act transition 3` lines." Correction **B28/B29** succeeds. Yet **B30** again
  assumes the duplicated state and repeats the same correction at **B31**,
  receiving "String not found" at **B32**. **B33** finally requests verification.
- **E21**, `action`, after successful write **E19/E20**: "The request was ...
  and I have done that." It ends with "I should confirm completion." The saved
  reasoning, rather than just the final answer, acknowledges the completed work.
- **B38**, second return to `planning`, outlines reading and explaining the
  file but does not identify its current mode. **B41** says "actually, the tool
  output is provided" and describes the current file. **E24**, `planning`, says
  "current \"PLAN mode\" (read-only)"; **E27** says "I have already read the file
  and see its content" and describes both comments. There is no counterpart to
  the reference's explicit wrong Act claim in either arm's fifth-transition
  reasoning. Missing explicit awareness in baseline is **not** scored as correct
  awareness merely because it used a permitted tool.

| Observable measure | Baseline | Experiment |
| --- | ---: | ---: |
| Completed transitions | 5 | 5 |
| Explicit incorrect current-mode claims in available reasoning | 0 | 0 |
| Model invocations | 15 | 10 |
| Invocations with recorded reasoning | 12 | 8 |
| Invocations with unavailable reasoning | 3 | 2 |
| Distinct native tool calls | 10 | 5 |
| Repeated identical call-signature groups within a turn | 3 | 0 |
| Additional calls within those groups | 3 | 0 |
| Exact repeated reasoning-paragraph groups (80+ normalized characters) | 0 | 0 |

Baseline duplicate groups are edits **B19/B22**, reads **B25/B34**, and edits
**B28/B31**. The second read can be justified as verification after intervening
edits; it is not labelled an unnecessary read simply because its arguments match.
The two repeated-edit groups have the concrete stale-result failures above.
Both **B7** and **E7** revisit the same edit-versus-write decision several times,
so the notice did not eliminate repetitive deliberation. Exact paragraph matching
does not detect every semantic repetition.

Baseline invocations **2, 4, 13** and experiment invocations **2, 4** expose no
recorded reasoning. They are explicitly `unavailable`, not counted as mode-aware
or confused from their tool calls/final replies. Assistant event IDs and complete
text remain in the raw evidence (including B5/B10/B16/B36/B42 and
E5/E10/E16/E22/E28); successful final answers do not contribute to the mode score.

## Exact-prompt regression replay and residual failures

The last guarded regression replay is retained separately:
[regression-baseline.json](mode_transition_evidence/regression-baseline.json) and
[regression-experiment.json](mode_transition_evidence/regression-experiment.json).
Here **RB** and **RE** identify those files. Baseline reached transitions 1–3;
its resumed model call timed out before 4–5. Experiment completed all five.
This is **not** a completed five-transition paired rate comparison.

- **RB13**, metadata `planning`, repeats the original failure mechanism:
  "If I just executed `write_file`, it means I must have switched to Act mode."
  It also states that the prior write "implies I am no longer in Plan mode".
  Those are incorrect current-state inferences from historical tool execution.
- Read **RB14/RB15** returns the file. Later **RB19** explicitly acknowledges
  having read it; the failure is not simply absence of a tool result. The model
  then repeats `show_plan` with three distinct call IDs (**RB16, RB20, RB24**),
  despite "No current Plan" results **RB17/RB21/RB25**. Events **RB18/RB22/RB26**
  replay those same call IDs during resume and are **not** counted as new calls.
- **RB23** repeats its intended test additions. Partial recorded reasoning
  **RB27** grows to 32,923 characters; its opening "The user wants to add more
  tests ..." occurs **52 times**. Six exact repeated paragraph groups are found
  across this turn. **RB28** records interruption. Invocation **9** has real
  partial reasoning linked by native run metadata, but no finished response;
  it is not mislabeled as no reasoning. Invocation **6** genuinely has no
  recorded reasoning. This replay reproduces wrong-mode inference and a loop,
  though its repeated tool differs from the reference's read loop.
- **RE13**, metadata `planning`, instead says "I am in **PLAN mode**, which is
  read-only" and "I cannot use `edit_file`". It does not attempt an edit in
  that Plan turn. **RE5/RE8** and **RE16**, recorded `action`, explicitly identify
  Act mode. This is reasoning-level evidence consistent with a beneficial notice
  effect in this harder replay, rather than a conclusion drawn from tool success.
- However, the **second return to Plan, RE42**, still asks: "Wait, if I'm in
  PLAN mode, why did those tools work?" Later: "If I am in a read-only
  environment, `write_file` should not work. But it did. This is confusing."
  It repeatedly speculates that earlier tools worked despite Plan restrictions.
  It ultimately says "I should stick to the instructions now. I am in Plan
  mode." The present-mode assertion is correct, but historical-mode confusion
  clearly remains. Tentative alternatives such as "maybe I am not in Plan mode"
  are retained and discussed, not silently converted into unequivocal claims.
- **RE16**, `action`, correctly says "I am in ACT mode". Yet that Act turn makes
  five identical failed edits (**RE19/23/27/30/32**) with "String not found"
  results (**RE20/24/28/31/33**), three identical reads (**RE17/21/25**), and two
  identical writes (**RE35/38**, both returning success). Available reasoning
  **RE37** acknowledges the edit failures and chooses replacement; nevertheless
  a successful write result **RE36** already preceded its further write **RE38**.
  This is another result-use/repetition problem despite correct Act identification.
  Invocations **7–14 and 16** have unavailable reasoning; no explanation of those
  retries is invented from their outputs.

## Scope, limitations, retained runs and checks

**Conclusion:** this evidence does not establish a reliable or general reduction
in mode confusion. The completed pair has no incorrect-mode baseline to reduce
(0 versus 0). Its drop from three repeated-signature groups to zero is observable
but is not itself a mode-awareness metric. The harder replay shows a promising
change from incorrect Act inference to explicit current Plan recognition, while
still exposing historical-mode confusion, repeated reasoning, and repeated
identical tool calls with the notice. The baseline's failure prevents a matched
comparison of the second return to Plan for that exact-prompt scenario.

This is one completed pair, not a counterbalanced or statistically powered
study. Temperature 0 did not make repeated live runs identical. Generated file
contents diverged after different model actions; later prompts are matched,
not identical resulting transcripts. Earlier replay work overlapped the start
of the inspection baseline on the same local endpoint; no latency comparison
is made. Different model versions, sampling or prompts may change the findings.

Three development pairs are also retained, rather than discarding their failures:

| Evidence prefix | Disposition |
| --- | --- |
| `pilot-` | Guard off. Baseline stopped at transition 3 because the first probe frontend lacked display handling. Experiment reached 5, but its postcheck falsely rejected a notice quoted in genuine model reasoning. |
| `pilot2-` | Guard off. Native display resumes exposed further wrong-mode reasoning and display repetition; baseline hit the probe's three-display limit at transition 3. Experiment reached 5. |
| `pilot3-` | Guard on. Baseline stopped on an unsolicited formal Plan review that the probe did not yet handle. Experiment reached 5 and includes repeated failed-edit evidence. |

Those are development/diagnostic captures, not extra successful A/B trials.
All derived inspections and invocation links were recomputed with the final
analyzer from their preserved raw events, separating replayed call IDs and
linking partial reasoning by native run ID. The raw events remain unchanged.
The final frontend handles actual display, review, clarification and approval
requests with the same declared policy in both arms. A model quoting notice text
in its own recorded reasoning is not a persisted synthetic HumanMessage.

Supplemental changes are confined to this report, the probe, retained JSON
evidence, and `tests/test_mode_transition_probe.py`. Production LOC changed: **0**.
The original 13 tests plus 10 new probe attribution/analysis/frontend checks
pass (**23 tests**). The complete inspection A/B command passes; the exact-prompt
regression command correctly reports failure for the incomplete baseline and
preserves its evidence. The original source reference hash is unchanged.
Working and staged `git diff --check` pass. The full repository suite was not rerun.
