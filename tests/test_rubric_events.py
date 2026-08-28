"""Tests for native DeepAgents rubric event projection."""

from __future__ import annotations

import unittest

from runtime.rubric_events import RubricEventRenderer, format_elapsed, rubric_result_text


class RecordingRenderer:
    """Capture rubric callbacks without interpreting their payloads."""

    def __init__(self) -> None:
        self.events: list[tuple[object, ...]] = []

    def rubric_evaluation_started(
        self,
        run_id: str,
        pass_number: int,
        max_iterations: int,
        *,
        grader_model: str = "",
        phase: str = "verifying",
    ) -> None:
        self.events.append(("started", run_id, pass_number, max_iterations, grader_model, phase))

    def rubric_lifecycle_event(self, event: dict[str, object]) -> None:
        self.events.append(("lifecycle", dict(event)))

    def rubric_evaluation_finished(self, evaluation: dict[str, object], max_iterations: int) -> None:
        self.events.append(("finished", evaluation, max_iterations))

    def rubric_evaluations_cancelled(self) -> None:
        self.events.append(("cancelled",))


class RubricEventTests(unittest.TestCase):
    """Rubric progress should enrich, not replace, the native evaluation."""

    def test_fake_clock_adds_grader_and_duration_to_native_evaluation(self) -> None:
        """One matching start/end pair should produce durable identity and timing."""
        ticks = iter((10.0, 144.25))
        renderer = RecordingRenderer()
        rubric = RubricEventRenderer(
            renderer,
            3,
            grader_model="lmstudio:bonsai",
            clock=lambda: next(ticks),
        )

        rubric.handle({"type": "rubric_evaluation_start", "grading_run_id": "grade-1", "iteration": 0})
        rubric.handle(
            {
                "type": "rubric_evaluation_end",
                "grading_run_id": "grade-1",
                "iteration": 0,
                "result": "satisfied",
                "explanation": "All criteria were verified.",
                "criteria": [{"name": "File exists", "passed": True, "gap": ""}],
            }
        )

        self.assertEqual(
            renderer.events[0],
            ("started", "grade-1", 1, 3, "lmstudio:bonsai", "verifying"),
        )
        evaluation = rubric.evaluations[0]
        self.assertEqual(evaluation["grader_model"], "lmstudio:bonsai")
        self.assertEqual(evaluation["duration_ms"], 134250)
        self.assertEqual(evaluation["criteria"][0]["name"], "File exists")

    def test_result_text_lists_every_model_generated_criterion(self) -> None:
        """Passed and failed native criterion names should both remain visible."""
        text = rubric_result_text(
            {
                "grading_run_id": "grade-1",
                "iteration": 0,
                "result": "needs_revision",
                "grader_model": "openai:gpt-5.6-terra",
                "duration_ms": 134000,
                "explanation": "One criterion still needs work.",
                "criteria": [
                    {"name": "File exists", "passed": True, "gap": ""},
                    {"name": "Correct count", "passed": False, "gap": "Found 19 words."},
                ],
            },
            3,
        )

        self.assertIn("Grader: openai:gpt-5.6-terra", text)
        self.assertIn("Completed in 02:14", text)
        self.assertIn("✓ File exists", text)
        self.assertIn("✗ Correct count: Found 19 words.", text)
        self.assertIn("Needs revision: One criterion still needs work.", text)

    def test_unfinished_activity_is_cancelled_without_persisting_ticks(self) -> None:
        """An interrupted grader should stop its transient renderer activity."""
        renderer = RecordingRenderer()
        rubric = RubricEventRenderer(renderer, 2, clock=lambda: 1.0)
        rubric.handle({"type": "rubric_evaluation_start", "grading_run_id": "grade-1", "iteration": 0})

        rubric.cancel()
        rubric.cancel()

        self.assertEqual(renderer.events[-1], ("cancelled",))
        self.assertEqual(renderer.events.count(("cancelled",)), 1)
        self.assertEqual(rubric.evaluations, [])

    def test_nested_tool_lifecycle_is_live_correlated_and_full_fidelity(self) -> None:
        ticks = iter((1.0, 1.0, 2.0, 2.0, 3.0, 4.0))
        renderer = RecordingRenderer()
        rubric = RubricEventRenderer(renderer, 3, clock=lambda: next(ticks))
        start = {"grading_run_id": "grade-live", "iteration": 0}

        rubric.handle({"type": "rubric_evaluation_start", **start})
        rubric.handle({"type": "rubric_verification_start", **start})
        rubric.handle(
            {
                "type": "rubric_tool_call_delta",
                **start,
                "chunk": {
                    "type": "tool_call_chunk",
                    "index": 0,
                    "id": "read-1",
                    "name": "read_file",
                    "args": '{"file_path": "/fo',
                },
            }
        )
        rubric.handle(
            {
                "type": "rubric_tool_call_delta",
                **start,
                "chunk": {
                    "type": "tool_call_chunk",
                    "index": 0,
                    "args": 'o"}',
                },
            }
        )
        rubric.handle(
            {
                "type": "rubric_tool_start",
                **start,
                "tool_call_id": "read-1",
                "tool_name": "read_file",
                "tool_args": {"file_path": "/foo"},
            }
        )
        raw_output = "first line\n" + "x" * 5000
        rubric.handle(
            {
                "type": "rubric_tool_end",
                **start,
                "tool_call_id": "read-1",
                "tool_name": "read_file",
                "output": raw_output,
                "is_error": False,
                "duration_ms": 1250,
            }
        )
        rubric.handle(
            {
                "type": "rubric_tool_start",
                **start,
                "tool_call_id": "read-2",
                "tool_name": "read_file",
                "tool_args": {"file_path": "/missing"},
            }
        )
        rubric.handle(
            {
                "type": "rubric_tool_end",
                **start,
                "tool_call_id": "read-2",
                "tool_name": "read_file",
                "output": "File not found",
                "is_error": True,
                "duration_ms": 10,
            }
        )
        rubric.handle({"type": "rubric_verification_end", **start, "succeeded": True})
        rubric.handle({"type": "rubric_grading_start", **start})
        rubric.handle({"type": "rubric_grading_end", **start, "succeeded": True})
        rubric.handle(
            {
                "type": "rubric_evaluation_end",
                **start,
                "result": "satisfied",
                "explanation": "Verified.",
                "criteria": [{"name": "Current", "passed": True, "gap": ""}],
            }
        )

        lifecycle = [event[1] for event in renderer.events if event[0] == "lifecycle"]
        self.assertEqual(lifecycle[0]["type"], "rubric_verification_start")
        deltas = [event for event in lifecycle if event["type"] == "rubric_tool_call_delta"]
        self.assertEqual(len(deltas), 2)
        self.assertEqual(deltas[-1]["tool_call_id"], "read-1")
        self.assertEqual(deltas[-1]["tool_args"], {"file_path": "/foo"})
        self.assertLess(
            next(i for i, event in enumerate(lifecycle) if event["type"] == "rubric_verification_end"),
            next(i for i, event in enumerate(lifecycle) if event["type"] == "rubric_grading_start"),
        )
        evaluation = rubric.evaluations[0]
        self.assertEqual([tool["call_id"] for tool in evaluation["verifier_tools"]], ["read-1", "read-2"])
        self.assertEqual(evaluation["verifier_tools"][0]["output"], raw_output)
        self.assertTrue(evaluation["verifier_tools"][1]["is_error"])
        self.assertEqual(evaluation["verifier_status"], "complete")
        self.assertEqual(evaluation["grader_status"], "complete")

    def test_verifier_failure_is_distinct_from_a_failed_tool(self) -> None:
        renderer = RecordingRenderer()
        rubric = RubricEventRenderer(renderer, 1, clock=lambda: 1.0)
        identity = {"grading_run_id": "grade-failed", "iteration": 0}
        rubric.handle({"type": "rubric_evaluation_start", **identity})
        rubric.handle({"type": "rubric_verification_start", **identity})
        rubric.handle({"type": "rubric_verification_end", **identity, "succeeded": False})
        rubric.handle(
            {
                "type": "rubric_evaluation_end",
                **identity,
                "result": "grader_error",
                "explanation": "Verifier raised RuntimeError.",
                "criteria": [],
            }
        )

        evaluation = rubric.evaluations[0]
        self.assertEqual(evaluation["verifier_status"], "failed")
        self.assertEqual(evaluation["verifier_tools"], [])
        text = rubric_result_text(evaluation, 1)
        self.assertIn("Verifier · Failed", text)
        self.assertNotIn("Grader ·", text)

    def test_zero_tool_completion_keeps_both_phase_headings_and_result(self) -> None:
        renderer = RecordingRenderer()
        rubric = RubricEventRenderer(renderer, 2, clock=lambda: 1.0)
        identity = {"grading_run_id": "grade-zero", "iteration": 0}
        for event in (
            {"type": "rubric_evaluation_start", **identity},
            {"type": "rubric_verification_start", **identity},
            {"type": "rubric_verification_end", **identity, "succeeded": True},
            {"type": "rubric_grading_start", **identity},
            {"type": "rubric_grading_end", **identity, "succeeded": True},
            {
                "type": "rubric_evaluation_end",
                **identity,
                "result": "satisfied",
                "explanation": "Done.",
                "criteria": [{"name": "Done", "passed": True, "gap": ""}],
            },
        ):
            rubric.handle(event)

        text = rubric_result_text(rubric.evaluations[0], 2)
        self.assertIn("Rubric review · pass 1 of 2", text)
        self.assertIn("Verifier · Complete", text)
        self.assertIn("No tools called.", text)
        self.assertIn("Grader · Complete", text)
        self.assertIn("1 of 1 criteria satisfied", text)
        self.assertIn("Satisfied: Done.", text)

    def test_elapsed_format_is_stable(self) -> None:
        self.assertEqual(format_elapsed(0), "00:00")
        self.assertEqual(format_elapsed(65_999), "01:05")
        self.assertEqual(format_elapsed(3_661_000), "1:01:01")


if __name__ == "__main__":
    unittest.main()
