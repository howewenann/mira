# Planning Policy Live Evaluation

Configured-model evaluation for autonomous `ask_user` recognition. Each case
used a fresh planning thread in a disposable workspace. The renderer selected
the recommended or first choice and observed whether the resumed turn reached
`present_plan`.

| Batch | Case | Classification | First terminal action | Prose-choice violation | Resumed outcome | Thread/log |
|---|---:|---|---|---|---|---|
| 1 | 1 | IMPLEMENTATION | `ask_user` | No | `present_plan` | `ask-user-eval:b1:1` |
| 1 | 2 | IMPLEMENTATION | `ask_user` | No | `present_plan` | `ask-user-eval:sequential-probe` |
| 1 | 3 | IMPLEMENTATION | `ask_user` | No | `present_plan` | `ask-user-eval:b1:3` |
| 2 | 4 | IMPLEMENTATION | `ask_user` | Yes, after resume | None | `ask-user-eval:b2:1` |
| 2 | 5 | IMPLEMENTATION | `ask_user` | No | `present_plan` | `ask-user-eval:b2:2` |
| 2 | 6 | IMPLEMENTATION | `ask_user` | Yes, after resume | None | `ask-user-eval:b2:3` |
| 3 | 7 | IMPLEMENTATION | `ask_user` | No | Timeout after selection | `ask-user-eval:b3:1-rerun` |
| 3 | 8 | IMPLEMENTATION | `ask_user` | No | `present_plan` | `ask-user-eval:b3:2` |
| 3 | 9 | IMPLEMENTATION | `ask_user` | No | `present_plan` | `ask-user-eval:b3:3` |
| 4 | 10 | IMPLEMENTATION | `ask_user` | No | `present_plan` | `ask-user-eval:b4:1` |
| 4 | 11 | IMPLEMENTATION | `ask_user` | No | `present_plan` | `ask-user-eval:b4:2` |
| 4 | 12 | IMPLEMENTATION | `ask_user` | No | `present_plan` | `ask-user-eval:b4:3` |

## Batch Findings

- **Batch 1:** 3/3 policy passes. Cases 2 and 3 performed unnecessary research
  before asking, so the prompt was revised to resolve explicit preferences
  before direction-dependent research. Completed turns took 58.0-81.7 seconds.
- **Batch 2:** Initial decision routing passed 3/3 and `ask_user` occurred before
  research. Full resumption passed 1/3; cases 4 and 6 ended in prose after the
  choice. The prompt was revised to make `ask_user` intermediate and preserve
  IMPLEMENTATION intent after resume. Completed turns took 57.5-70.9 seconds.
- **Batch 3:** Initial routing passed 3/3 and full resumption passed 2/3. Case 7
  merged the supplied alternatives and exceeded 120 seconds after selection.
  The prompt was revised to preserve explicit alternatives separately. Cases 8
  and 9 completed in 47.0 and 92.0 seconds.
- **Batch 4 (held out):** 3/3 full passes with no prose-choice violations.
  Completed turns took 57.0-88.4 seconds, so no final prompt revision was made.

The configured endpoint commonly completed the policy turn and printed its
result before the disposable Python process stalled during interpreter
shutdown; command exit 124 in those runs is a shutdown timeout, not a policy
timeout. Case 7 is the exception: it reached `ask_user` but did not reach
`present_plan` within the turn cap.

The motivating failure is
`.mira/_sessions/20260713-091552+0800-9d082712.json`: MIRA offered three
directions and asked the user to choose in assistant prose, then used
`ask_user` only after the follow-up "give me a suggestion."

## Final Broad-Goal Regression

After replacing the software-specific decision wording with the general
facts-versus-preferences policy, the exact test-only prompt
`find a way to make the code base neater` was run once in thread
`ask-user-eval:final-broad-goal`. The configured model did not reach
`ask_user`, `present_plan`, or another terminal marker within the 120-second
turn cap. No further phrase-specific prompt tuning was added; this result is
recorded as a model-compliance/latency limitation.
