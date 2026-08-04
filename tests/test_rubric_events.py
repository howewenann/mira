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
    ) -> None:
        self.events.append(("started", run_id, pass_number, max_iterations, grader_model))

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

        self.assertEqual(renderer.events[0], ("started", "grade-1", 1, 3, "lmstudio:bonsai"))
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

    def test_elapsed_format_is_stable(self) -> None:
        self.assertEqual(format_elapsed(0), "00:00")
        self.assertEqual(format_elapsed(65_999), "01:05")
        self.assertEqual(format_elapsed(3_661_000), "1:01:01")


if __name__ == "__main__":
    unittest.main()
