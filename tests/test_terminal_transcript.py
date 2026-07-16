"""Tests for shared plain terminal transcript formatting."""

from __future__ import annotations

import unittest

from ui.terminal_transcript import TerminalTranscript


class TerminalTranscriptTests(unittest.TestCase):
    """Tests for one-shot and trace transcript formatting."""

    def make_transcript(self, *, output_chars: int = 240) -> tuple[TerminalTranscript, list[str]]:
        chunks: list[str] = []
        return TerminalTranscript(chunks.append, tool_output_chars=output_chars), chunks

    def test_reasoning_buffers_and_preserves_line_breaks(self) -> None:
        transcript, chunks = self.make_transcript()

        transcript.reasoning_delta("The user wants:")
        transcript.reasoning_delta("\n")
        transcript.reasoning_delta("1. Check envs")
        transcript.finish_main()

        self.assertEqual("".join(chunks), "\nthinking:\nThe user wants:\n1. Check envs\n")

    def test_whitespace_reasoning_is_skipped(self) -> None:
        transcript, chunks = self.make_transcript()

        transcript.reasoning_delta("   \n")
        transcript.finish_main()

        self.assertEqual("".join(chunks), "")

    def test_assistant_streams_under_one_heading(self) -> None:
        transcript, chunks = self.make_transcript()

        transcript.text_delta("hello")
        transcript.text_delta(" there")
        transcript.finish_main()

        self.assertEqual("".join(chunks), "\nmira:\nhello there\n")

    def test_tool_blocks_match_terminal_spacing(self) -> None:
        transcript, chunks = self.make_transcript(output_chars=12)

        transcript.tool_call("read_file", {"file_path": "README.md"})
        transcript.tool_result("read_file", "x" * 30)

        self.assertEqual(
            "".join(chunks),
            "\nread_file:\nargs: {'file_path' ... truncated ...\nread_file output: xxxxxxxxxxxx ... truncated ...\n",
        )

    def test_completed_tool_result_prints_promptly_at_safe_boundary(self) -> None:
        transcript, chunks = self.make_transcript()

        transcript.tool_call("read_file", {"path": "README.md"})
        transcript.completed_tool_result("read_file", "contents")

        self.assertTrue("".join(chunks).endswith("read_file output: contents\n"))

    def test_completed_tool_result_waits_until_assistant_stream_finishes(self) -> None:
        transcript, chunks = self.make_transcript()

        transcript.text_delta("The answer")
        transcript.completed_tool_result("read_file", "contents")
        transcript.text_delta(" continues.")

        self.assertEqual("".join(chunks), "\nmira:\nThe answer continues.")
        transcript.finish_main()
        self.assertEqual(
            "".join(chunks),
            "\nmira:\nThe answer continues.\nread_file output: contents\n",
        )

    def test_completed_tool_result_waits_until_reasoning_finishes(self) -> None:
        transcript, chunks = self.make_transcript()

        transcript.reasoning_delta("Still thinking")
        transcript.completed_tool_result("read_file", "contents")

        self.assertEqual(chunks, [])
        transcript.finish_main()
        self.assertEqual(
            "".join(chunks),
            "\nthinking:\nStill thinking\nread_file output: contents\n",
        )

    def test_delegation_and_subagents_match_terminal_spacing(self) -> None:
        transcript, chunks = self.make_transcript(output_chars=32)

        transcript.delegation_started([{"args": {"description": "judge all haikus"}}])
        transcript.subagent_started("general-purpose [one]", "judge all haikus")
        transcript.subagent_finished("general-purpose [one]", "winner")

        rendered = "".join(chunks)
        self.assertIn("\ntask:\ndelegating to 1 subagent(s)\nrequest: judge all haikus\n", rendered)
        self.assertIn("\nsubagent - general-purpose [one]:\nrequest: judge all haikus\n", rendered)
        self.assertIn("\nsubagent - general-purpose [one]:\ndone\noutput: winner\n", rendered)

    def test_output_contains_no_ansi(self) -> None:
        transcript, chunks = self.make_transcript()

        transcript.text_delta("hello")
        transcript.finish_main()

        self.assertNotIn("\033[", "".join(chunks))

    def test_rubric_activity_and_result_are_concise(self) -> None:
        transcript, chunks = self.make_transcript()
        transcript.rubric_evaluation_started(1, 3)
        transcript.rubric_evaluation_finished(
            {
                "grading_run_id": "run-1",
                "iteration": 0,
                "result": "needs_revision",
                "explanation": "Verification is missing.",
                "criteria": [
                    {"name": "Files updated", "passed": True, "gap": ""},
                    {"name": "Tests run", "passed": False, "gap": "No test output"},
                ],
            },
            3,
        )

        rendered = "".join(chunks)
        self.assertIn("Reviewing completion criteria · pass 1 of 3", rendered)
        self.assertIn("1 of 2 criteria satisfied", rendered)
        self.assertIn("- Tests run: No test output", rendered)
        self.assertNotIn("grading_run_id", rendered)


if __name__ == "__main__":
    unittest.main()
