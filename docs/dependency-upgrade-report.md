# Dependency upgrade verification

Verified on 2026-11-14 from latest upstream `main`, `4b5f01b`.
This is dependency preparation only: `config/version.py` remains **2.12.0**;
no release, tag or publication was created. The old editable install reported
2.10.0; reinstalling the checkout refreshed that stale metadata to the existing
source version, without changing MIRA's version.

## Dependency changes

PyPI JSON metadata was queried for every direct dependency and the relevant
provider/checkpoint packages. Stable, non-yanked releases were selected and
checked against Python 3.12 and upstream requirements. The table gives installed
versions followed by declared requirements; duplicate extras are collapsed.
The [machine-readable audit](dependency-upgrade-audit.json) retains the query
results and before/after declarations. `environment.yml` still installs `-e .`
and needs no duplicate version list. Existing exact pins, ranges and extras
remain the dependency-management convention; no lockfile was introduced.

| Package | Previous | New | Reason |
|---|---|---|---|
| [agent-client-protocol](https://pypi.org/project/agent-client-protocol/) | 0.12.1; `agent-client-protocol==0.12.1`, `agent-client-protocol[http]==0.12.1` | 0.12.1; `agent-client-protocol==0.12.1`, `agent-client-protocol[http]==0.12.1` | Retain 0.12.1: 1.0.0 removes the helper API and changes schema/transports; see limitation below. |
| [aiosqlite](https://pypi.org/project/aiosqlite/) | 0.22.1; `aiosqlite>=0.20,<1` | 0.22.1; `aiosqlite>=0.22.1,<1` | Already current; raise declared minimum to verified version. |
| [coolname](https://pypi.org/project/coolname/) | 5.0.0; `coolname>=5.0.0` | 5.0.0; `coolname>=5.0.0` | Already the latest stable release; unchanged. |
| [deepagents](https://pypi.org/project/deepagents/) | 0.7.32; `deepagents==0.7.32` | 0.7.38; `deepagents==0.7.38` | Target release; filesystem search, compaction, refusal and delegation fixes. |
| [fastmcp](https://pypi.org/project/fastmcp/) | 4.1.2; `fastmcp>=4.1.2,<5` | 4.2.0; `fastmcp>=4.2.0,<5` | Current stable; keep client, resources, prompts and Skills on the supported API. |
| [hypercorn](https://pypi.org/project/hypercorn/) | 0.18.0; `hypercorn>=0.17` | 0.18.0; `hypercorn>=0.18.0` | Already current; raise optional minimum to verified version. |
| [langchain](https://pypi.org/project/langchain/) | 1.4.7; `langchain[mcp]==1.4.7` | 1.5.0; `langchain[mcp]==1.5.0` | Required by DeepAgents; Python 3.11 minimum matches MIRA. |
| [langchain-anyllm](https://pypi.org/project/langchain-anyllm/) | 0.0.1a2; `langchain-anyllm>=0.0.1a2` | 0.0.1a2; `langchain-anyllm>=0.0.1a2` | No stable release exists; retain the already-required alpha. |
| [langchain-quickjs](https://pypi.org/project/langchain-quickjs/) | 0.3.7; `langchain-quickjs==0.3.7` | 0.3.12; `langchain-quickjs==0.3.12` | Runtime initialization, PTC schema/prompt caching and shutdown fixes. |
| [langgraph](https://pypi.org/project/langgraph/) | 1.2.18; `langgraph>=1.2.12` | 1.2.21; `langgraph>=1.2.21` | Stream namespace fixes, paused-tool handling, per-interrupt IDs. |
| [langgraph-checkpoint-sqlite](https://pypi.org/project/langgraph-checkpoint-sqlite/) | 3.1.2; `langgraph-checkpoint-sqlite>=3.1.2,<4` | 3.1.3; `langgraph-checkpoint-sqlite>=3.1.3,<4` | Current SQLite checkpoint release. |
| [langsmith](https://pypi.org/project/langsmith/) | 0.14.14; `langsmith>=0.14.12,<0.15.0` | 0.14.17; `langsmith>=0.14.17,<0.15.0` | DeepAgents minimum and current stable tracing client. |
| [lmstudio](https://pypi.org/project/lmstudio/) | 1.5.0; `lmstudio>=1.5.0` | 1.5.0; `lmstudio>=1.5.0` | Already the latest stable release; unchanged. |
| [openinference-instrumentation](https://pypi.org/project/openinference-instrumentation/) | 0.1.59; `openinference-instrumentation>=0.1.57,<0.2.0` | 0.1.77; `openinference-instrumentation>=0.1.77,<0.2.0` | Current stable; requires semantic conventions >=0.1.44. |
| [openinference-semantic-conventions](https://pypi.org/project/openinference-semantic-conventions/) | 0.1.33; `openinference-semantic-conventions>=0.1.32,<0.2.0` | 0.1.44; `openinference-semantic-conventions>=0.1.44,<0.2.0` | Current stable; aligned with instrumentation. |
| [opentelemetry-exporter-otlp-proto-http](https://pypi.org/project/opentelemetry-exporter-otlp-proto-http/) | 1.44.0; `opentelemetry-exporter-otlp-proto-http>=1.44.0,<2.0.0` | 1.45.2; `opentelemetry-exporter-otlp-proto-http>=1.45.2,<2.0.0` | Current stable; matches SDK and protocol packages. |
| [opentelemetry-sdk](https://pypi.org/project/opentelemetry-sdk/) | 1.44.0; `opentelemetry-sdk>=1.44.0,<2.0.0` | 1.45.2; `opentelemetry-sdk>=1.45.2,<2.0.0` | Current stable; align all installed OTLP companions. |
| [pyfiglet](https://pypi.org/project/pyfiglet/) | 1.0.4; `pyfiglet>=1.0.4` | 1.0.4; `pyfiglet>=1.0.4` | Already the latest stable release; unchanged. |
| [python-dotenv](https://pypi.org/project/python-dotenv/) | 1.2.3; `python-dotenv>=1.2.2` | 1.2.4; `python-dotenv>=1.2.4` | Current stable; parsing/CLI fixes; no .env edits. |
| [pyyaml](https://pypi.org/project/pyyaml/) | 6.0.3; `PyYAML>=6.0.0` | 6.0.3; `PyYAML>=6.0.3` | Already current; raise declared minimum to verified version. |
| [rich](https://pypi.org/project/rich/) | 15.0.0; `rich>=15.0.0` | 15.0.0; `rich>=15.0.0` | Already the latest stable release; unchanged. |
| [textual](https://pypi.org/project/textual/) | 8.2.7; `textual==8.2.7` | 8.2.8; `textual==8.2.8` | Extended-key parsing and screen-padding click fix. |
| [typer](https://pypi.org/project/typer/) | 0.24.2; `typer>=0.24.2,<0.25.0` | 0.29.1; `typer>=0.29.1,<0.30.0` | Current stable; widen the existing minor-version cap after CLI checks. |

Relevant resolved packages were explicitly advanced where needed, using pip's
`only-if-needed` strategy. Other transitive packages changed only to satisfy
the selected stack or repair its existing shared-environment tracing conflict.

| Related installed package | Previous | New |
|---|---|---|
| any-llm-sdk | 1.26.0 | 1.50.0 |
| langchain-core | 1.7.0 | 1.8.0 |
| langchain-anthropic | 1.7.7 | 1.7.10 |
| langchain-google-genai | 4.4.2 | 4.5.0 |
| langgraph-checkpoint | 4.3.0 | 4.3.1 |
| langgraph-prebuilt | 1.1.0 | 1.1.1 |
| langgraph-sdk | 0.4.10 | 0.4.14 |
| openai | 3.6.0 | 3.51.0 |
| anthropic | 1.2.0 | 1.17.0 |
| google-genai | 2.20.0 | 3.3.0 |
| arize-phoenix | 20.4.0 | 21.0.0 |
| opentelemetry-exporter-otlp | 1.44.0 | 1.45.2 |
| opentelemetry-exporter-otlp-proto-grpc | 1.44.0 | 1.45.2 |
| opentelemetry-instrumentation | 0.65b0 | 0.66b2 |

Phoenix is the separately installed local tracing server documented in README,
not a new mandatory MIRA dependency. Its old 20.4.0 requirement for
`fastmcp-slim<4` already failed baseline `pip check`. Phoenix 21.0.0 accepts
FastMCP 4. Matching OTLP meta/gRPC exporters and the already-installed beta OTel
instrumentation were aligned to SDK 1.45.2 / instrumentation 0.66b2, fixing the
exact-version conflicts exposed by the tracing upgrade. No new prerelease
dependency was opted into. Phoenix's Pydantic-AI resolution was constrained by
`google-genai==3.3.0` so its optional agent cannot downgrade the new Google
integration's SDK. No Phoenix database or running server was migrated.

Intentionally retained or excluded:

- **ACP 0.12.1:** latest stable is 1.0.0, but its removed `acp.helpers`/top-level
  builders are used throughout MIRA's adapter and four teaching clients. Schema
  and HTTP lifecycle changes also need a coordinated migration. This task keeps
  the currently compatible pin; ACP 1.0 is not claimed as verified compatible.
- **langchain-anyllm 0.0.1a2:** PyPI has no stable release. This is the existing
  explicit prerelease exception, not a newly selected alpha.
- **Anthropic SDK 1.17.0:** latest stable overall is 2.8.0; the co-installed
  Phoenix server requires `anthropic<2`. 1.17.0 is the latest stable 1.x release.
- **LM Studio 1.5.0, Rich 15.0.0, coolname 5.0.0, pyfiglet 1.0.4:** already current.
- **langchain-openai:** not installed/used by MIRA's ChatAnyLLM factory; do not
  introduce it just because an OpenAI-compatible endpoint is configured.
- **langchain-mcp-adapters:** installed 0.2.2; latest 0.3.2 was checked, but MIRA
  uses FastMCP integration directly and does not import this leftover package.
- **Hatchling:** unpinned build-isolation dependency, latest stable 1.33.0;
  editable-wheel construction succeeded without adding a runtime requirement.

## Important upstream changes

The exact [DeepAgents 0.7.32...0.7.38 comparison](https://github.com/langchain-ai/deepagents/compare/deepagents%3D%3D0.7.32...deepagents%3D%3D0.7.38)
and changed SDK source files were inspected, not just the installation result.

- **0.7.33:** native model-refusal errors, child refusal notices, task-call IDs
  in checkpoint metadata, notification metadata, failed-compaction error status,
  single-file Python grep fallback and bounded long-line results. Rubric loses
  its beta annotation; MIRA's custom Rubric/offload implementation stays intact.
- **0.7.34:** summaries anchor the cutoff by message ID, tolerating prefix removal.
- **0.7.35:** permitted symlink names are retained in virtual results; forked
  children preserve the parent's prompt-cache prefix; media notifications are marked.
- **0.7.36:** backend binding for middleware requesting it, FIFO download guard,
  and task-result checkpoint namespace artifacts.
- **0.7.37:** optional provider-cache lookup results are cached.
- **0.7.38:** `ls`, `grep`, and `glob` filter symlink targets needing approval
  outside the approved search scope. MIRA's default workspace reads are allowed,
  so this adds no new read prompt; deny rules and backend containment still apply.

`create_deep_agent` call arguments and `ModelRequest.override()` remain compatible.
Main/native child stacks gain ModelRefusalMiddleware; MIRA's precompiled
`create_sub_agent` path does not add it automatically. Registration order and
hook ownership are updated in the [middleware guide](middleware/README.md).
Skills still use native discovery; HITL and task execution stay upstream-owned.
Compaction remains the native engine with MIRA's archive/replay safeguards.
AnyLLM model objects still do not activate native Anthropic caching by brand alone.

Other reviewed changes:

- [LangChain 1.5.0](https://github.com/langchain-ai/langchain/releases/tag/langchain%3D%3D1.5.0)
  / Core 1.8.0 require Python 3.11, matching MIRA. Intermediate versions add
  refusal blocks, non-mutating message merges, tool-cancellation completion,
  and keep middleware-internal model calls off ordinary message streams.
- [LangGraph](https://github.com/langchain-ai/langgraph/releases/tag/1.2.21)
  fixes graph namespace projection, paused-tool errors and namespace lookup;
  each interrupt gets its own ID. Resume remains native `Command(resume=...)`.
- [QuickJS 0.3.12](https://github.com/langchain-ai/deepagents/releases/tag/langchain-quickjs%3D%3D0.3.12)
  includes lazy runtime initialization, PTC prompt caching/nested schema resolution,
  and nonblocking destructor cleanup. Existing MIRA task identity/replay safeguards
  remain; no optional QuickJS filesystem bridge was enabled.
- [FastMCP 4.2.0](https://github.com/PrefectHQ/fastmcp/releases/tag/v4.2.0)
  requires Python 3.11 and replaces the 4.1 skill authoring API. Wire manifests
  are now named MCPSkill; the deprecated Skill name is removed in 4.3.
- [AnyLLM](https://github.com/mozilla-ai/any-llm/compare/1.26.0...1.50.0)
  updates provider support, streamed reasoning/cache usage, tool-argument handling,
  context-overflow classification and model metadata. OpenAI SDK 3.51.0 adds API
  fields; its final provider tests and live Gemma repeat passed. The existing ChatAnyLLM
  factory and Gemma reasoning stream work without a provider shim.
- [Google integration 4.5.0](https://github.com/langchain-ai/langchain-google/releases/tag/libs%2Fgenai%2Fv4.5.0)
  supports Google SDK 3 and drops Python 3.10; Anthropic 1.7.10 supports native
  refusal translation and streamed usage fixes. Remote provider credentials were
  not used for live cloud calls.
- [Textual 8.2.8](https://github.com/Textualize/textual/releases/tag/v8.2.8)
  fixes extended keys and padding clicks and updates backspace aliases.
  [Typer](https://github.com/fastapi/typer/releases/tag/0.29.1) vendors Click from
  0.26 and changes help/metavar formatting; MIRA's CLI smoke passed.
- [OpenInference 0.1.77](https://github.com/Arize-ai/openinference/releases/tag/python-openinference-instrumentation-v0.1.77),
  semantic conventions 0.1.44, LangSmith 0.14.17 and
  [OTel 1.45.2](https://github.com/open-telemetry/opentelemetry-python/releases/tag/v1.45.2)
  were checked with MIRA's tracing tests and exact companion requirements.
- [Phoenix 21 migration](https://github.com/Arize-ai/phoenix/blob/arize-phoenix-v21.0.0/MIGRATION.md):
  loopback keeps unauthenticated local operation unless explicitly configured;
  non-loopback enables authentication by default. `--host` becomes PHOENIX_HOST.
  Existing-server migration and credentials remain operator-owned.

## Compatibility changes

Only three production modules changed:

1. `agent/middleware/compaction.py` passes the current messages into DeepAgents'
   cutoff calculation, compares progress against the resolved prior boundary,
   and persists `cutoff_message_id`. A new regression reproduced a stale cutoff
   of 10 instead of 4 after prefix removal; both new cutoff tests now pass.
2. `core/execution/streams/tools.py` distinguishes a stream closed without a
   payload from an explicit empty-string result. LangGraph now closes a paused
   Eval stream with no error/output. Recording an empty completion consumed the
   call ID before resume and hid its real result. The existing real QuickJS +
   child HITL test failed before this fix and passes afterward. Focused tests
   cover object/dict streams, resumed output and legitimate empty output.
3. `agent/mcp/skills.py` uses FastMCP's MCPSkill wire model under the descriptive
   local name SkillManifest. The MCP fixture uses the supported 4.2 directory
   provider. Existing integrity, permissions and lifecycle assertions remain.

No custom middleware was removed, no reasoning logger added, and no assertions
were weakened. ACP's specification test now expects the upgraded Hypercorn
minimum. Technical docs and manual prompts reflect the verified stack.

## Test results

All Python checks used the shared `mira` Conda environment and current checkout.
Scratch workspaces were disposable; source `.env` and user sessions were unchanged.

| Check | Result |
|---|---|
| Baseline full `unittest discover -s tests -t .` | 1,653 run; 6 failures, 10 errors |
| Baseline mode/handoff/compaction/QuickJS | 54 passed |
| Upgrade before tool-stream fix | Same 54 tests: one new QuickJS HITL failure, reproduced and fixed |
| Final focused compatibility, runner, compaction, MCP Skills | 215 run; 214 passed, 1 skipped |
| Final mode/handoff/Plan/Goal/Rubric/subagent/guard suites | 213 passed |
| Final provider/config/overflow/tracing suites | 64 passed |
| Deterministic mode-transition probe | 13 passed |
| Final full repository suite | 1,661 run in 645.561s; failures=6, errors=10, skipped=1 |
| `pip check` | Passed; no broken requirements |
| CLI help and real `python -m cli.main -p` in a disposable workspace | Passed |
| `git diff --check` | Passed |

The full suite covers automatic summarization, forced compaction, repeated
handoffs, tool pairing, retained Plans/Goals, mode commands/inline prompts,
cancellation/resume/cold recovery, subagents, skills/MCP, Rubric/evidence,
permissions, overflow/status correction, TUI input/paste/disabled prompts,
streaming/rendering/approvals and sessions. Failures are detailed below, so this
coverage is not a claim that every test passed.

### Live Gemma 4 12B

The endpoint was initially unavailable, then returned HTTP 200 with the configured
`google/gemma-4-12b`. Tests used the existing profile and temperature 0.
No cloud provider was substituted.

| Probe | Observed result |
|---|---|
| Existing mode probe, inspection scenario | Both five-transition arms completed; current MIRA model-call counts 2/2/2/3/2; no identical within-turn repeated calls |
| Existing mode probe, exact regression prompts | Current MIRA completed five turns, call counts 1/2/1/3/1; all Plan turns refused writes; Act created/edited the file |
| Comparison with notice bypassed | Stopped at Plan turn 3 after repeated show_plan/show_goal; 22 model calls; paired regression command therefore failed overall |
| Runtime probe | Plan write refusal, Act write/read, Eval 42, forced native compaction, Act→Plan→Act recall and cold-restart recall passed |
| Ordinary recorder/checkpoint inspection | 37 recorded events, 11 reasoning events; no synthetic transition notices persisted |

The [retained ordinary session](../tests/probes/dependency_upgrade_evidence/gemma-runtime-session.json)
contains reasoning IDs 2, 5, 10, 15, 18, 21, 24, 27, 30, 33 and 36. Tool results
include write 9, read 12, Eval 17 (`<result>42</result>`). Gemma first tried an
invalid Eval call in the multi-tool turn, then recovered to write/read; it did
not retry the completed successful results. The separate arithmetic Eval succeeded.
The runtime probe was repeated successfully after the final OpenAI SDK 3.51.0
update; its [ordinary session](../tests/probes/dependency_upgrade_evidence/gemma-runtime-sdk-final-session.json)
and [results](../tests/probes/dependency_upgrade_evidence/gemma-runtime-sdk-final-results.json)
are also retained. The full-suite process began before that final SDK patch;
provider/config/overflow/tracing tests and the live probe were rerun afterward.
Full mode-turn outcomes and reasoning candidates are retained in the
[mode summary](../tests/probes/dependency_upgrade_evidence/gemma-mode-summary.json).

Reasoning was reviewed in the recorder events and existing probe's mode-mention
candidates. Current-MIRA regression reasoning identifies Plan's read-only
authority, and Act follows the current write request. This small sample does
not establish general model reliability or dependency causality. One successful
Act edit duplicated a function definition; the bypassed arm's artifact loop is
also model behavior, not established as a new dependency defect. The same
limitations existed in earlier mode-probe findings. The first runtime harness
attempt read the wrong event field (`text` versus `output`); that harness-only
assertion was corrected and the entire probe rerun successfully.

## Remaining issues

The final 1,661-test run has exactly the same six failing tests and ten errors
as the original 1,653-test baseline, plus one explicit symlink-privilege skip.
No additional deterministic failure remains in the final full run. Its 16 failures/errors
remain outside this dependency task:

- Model-management tests expect a removed footer button (six errors), an older
  `compile_dynamic_subagents(skills=...)` signature, older issue-card behavior
  and a previous save-status message.
- Middleware span construction expects a dynamic child to be compiled;
  settings policy refresh expects `MiraApplication.agent` on its fixture.
- The MCP panel test queries `#mcp-reload` after its screen is dismissed.
- ACP HTTP approval mapping expects `approve` but observes `reject_once`.
- Autocomplete expects a five-item cap; the implementation returns thirteen.
- The inspector CSS test expects an obsolete literal block; Git configuration
  expects focus that the current UI does not assign.

The first upgraded full run had 9 failures / 11 errors, including a stale
Hypercorn-spec assertion and a new test-fixture keyword typo; both were
corrected and their focused checks passed. It also saw narrow-button geometry
and paused-scroll timing failures. Three paired isolated runs used the old
Textual wheel from a scratch import path (without altering installed packages):
8.2.8 passed all three pairs; 8.2.7 passed two, then reproduced the narrow-button
failure on the third. The scroll case failed once in the upgraded isolated run
and passed all paired repeats. These are intermittent UI observations; a new
dependency cause for the scroll timing failure was not established. No UI
production code or assertions were relaxed to hide them.

The [test-result inventory](../tests/probes/dependency_upgrade_evidence/test-results.json)
retains baseline and upgraded failure names. Full logs, PyPI metadata, resolver
reports and original A/B captures are left locally under
`build/dependency-upgrade/` (ignored scratch output).


The actual Windows symlink test is skipped for missing symlink privileges;
the deterministic resolved-target test exercises all three search tools in both
sync/async paths. Interactive OS clipboard behavior and live cloud-provider
calls were not manually exercised. Automated paste/input tests and provider
compatibility tests ran as part of the suite. Live Rubric grading against Gemma
and a running Phoenix collector were not exercised; deterministic Rubric,
offload/verifier and tracing checks ran. A real existing Phoenix database was
not opened or migrated.

**Release preparation:** the dependency changes have passing focused and live validation,
but MIRA is **not release-green** while the full-suite baseline and intermittent
UI failures remain unresolved. Review those failures and the deferred ACP 1.0
migration before treating this as release sign-off.
