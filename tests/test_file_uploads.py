"""Focused tests for backend-native local-file ingestion."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from deepagents.backends import FilesystemBackend

from agent.middleware.file_references import local_file_references
from ui.textual.file_uploads import clear_local_uploads, ingest_local_files, safe_upload_scope
from ui.textual.widgets.autocomplete_input import discover_project_files


class RecordingUploadBackend:
    """Record sequential uploads and optionally reject named source files."""

    def __init__(self, *, rejected_names: set[str] | None = None) -> None:
        self.rejected_names = rejected_names or set()
        self.uploads: list[tuple[str, bytes]] = []

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[object]:
        self.uploads.extend(files)
        path, _content = files[0]
        error = "permission_denied" if Path(path).name in self.rejected_names else None
        return [SimpleNamespace(path=path, error=error)]


class LocalFileUploadTests(unittest.IsolatedAsyncioTestCase):
    async def test_uploads_preserve_order_content_and_collision_safe_names(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            first = root / "first.txt"
            duplicate_one = root / "one" / "report.pdf"
            duplicate_two = root / "two" / "report.pdf"
            duplicate_one.parent.mkdir()
            duplicate_two.parent.mkdir()
            first.write_bytes(b"FIRST-CONTENT")
            duplicate_one.write_bytes(b"REPORT-ONE")
            duplicate_two.write_bytes(b"REPORT-TWO")
            backend = RecordingUploadBackend()

            references, failures = await ingest_local_files(
                [first, duplicate_one, duplicate_two],
                backend,
                "thread-1",
            )

        self.assertEqual(failures, [])
        self.assertEqual(
            [content for _path, content in backend.uploads],
            [b"FIRST-CONTENT", b"REPORT-ONE", b"REPORT-TWO"],
        )
        uploaded_paths = [path for path, _content in backend.uploads]
        self.assertEqual(
            [Path(path).name for path in uploaded_paths],
            ["first.txt", "report.pdf", "report.pdf"],
        )
        self.assertTrue(
            all(path.startswith("/.mira/_uploads/thread-1/") for path in uploaded_paths)
        )
        self.assertEqual(len({Path(path).parent.name for path in uploaded_paths}), 3)
        self.assertTrue(all(len(Path(path).parent.name) == 32 for path in uploaded_paths))
        self.assertEqual(local_file_references(" ".join(references)), uploaded_paths)
        self.assertNotIn("FIRST-CONTENT", " ".join(references))

    async def test_partial_read_and_upload_failures_keep_successful_references(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            good = root / "good.txt"
            rejected = root / "blocked.txt"
            missing = root / "missing.txt"
            good.write_text("good", encoding="utf-8")
            rejected.write_text("blocked", encoding="utf-8")
            backend = RecordingUploadBackend(rejected_names={"blocked.txt"})

            references, failures = await ingest_local_files(
                [good, rejected, missing],
                backend,
                "../unsafe/session",
            )

        self.assertEqual(len(references), 1)
        self.assertIn("good.txt", references[0])
        self.assertEqual(len(failures), 2)
        self.assertIn("blocked.txt: permission_denied", failures)
        self.assertTrue(any(failure.startswith("missing.txt:") for failure in failures))
        self.assertTrue(all(".." not in path for path, _content in backend.uploads))
        self.assertEqual(safe_upload_scope("../unsafe/session"), "_unsafe_session")

    async def test_real_project_backend_upload_is_immediately_discoverable(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            workspace = root / "workspace"
            source_directory = root / "outside"
            workspace.mkdir()
            source_directory.mkdir()
            source = source_directory / "annual report.pdf"
            source.write_bytes(b"pdf-like-bytes")
            backend = FilesystemBackend(root_dir=workspace, virtual_mode=True)

            references, failures = await ingest_local_files(
                [source],
                backend,
                "thread-1",
            )
            discovered = await discover_project_files(backend)

        self.assertEqual(failures, [])
        self.assertEqual(len(references), 1)
        self.assertTrue(references[0].startswith('@"/.mira/_uploads/thread-1/'))
        uploaded_path = local_file_references(references[0])[0].lstrip("/")
        self.assertIn(uploaded_path, discovered)

    async def test_clear_uploads_scopes_current_session_then_all_sessions(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            backend = FilesystemBackend(root_dir=workspace, virtual_mode=True)
            await backend.aupload_files(
                [
                    ("/.mira/_uploads/thread-1/one/report.txt", b"one"),
                    ("/.mira/_uploads/thread-2/two/report.txt", b"two"),
                ]
            )

            self.assertTrue(await clear_local_uploads(backend, "thread-1"))
            self.assertFalse((workspace / ".mira" / "_uploads" / "thread-1").exists())
            self.assertTrue((workspace / ".mira" / "_uploads" / "thread-2").exists())
            self.assertFalse(await clear_local_uploads(backend, "thread-1"))

            self.assertTrue(await clear_local_uploads(backend))
            self.assertFalse((workspace / ".mira" / "_uploads").exists())


if __name__ == "__main__":
    unittest.main()
