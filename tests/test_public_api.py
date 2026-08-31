"""Supported MIRA Python API and consumer-boundary tests."""

from __future__ import annotations

import ast
import runpy
import unittest
from pathlib import Path

import mira
import mira.api as public_api
from core.application.app import MiraApplication as CoreApplication
from core.application.session import MiraSession as CoreSession
from core.interface.events import (
    ArtifactEvent as CoreArtifactEvent,
    CompactionEvent as CoreCompactionEvent,
    FrontendEvent as CoreFrontendEvent,
    InformationEvent as CoreInformationEvent,
    MCPEvent as CoreMCPEvent,
    MessageEvent as CoreMessageEvent,
    RubricEvent as CoreRubricEvent,
    RuntimeEvent as CoreRuntimeEvent,
    SubagentEvent as CoreSubagentEvent,
    ToolEvent as CoreToolEvent,
    UsageEvent as CoreUsageEvent,
)
from core.interface.protocol import Frontend as CoreFrontend
from core.interface.requests import (
    ApprovalRequest as CoreApprovalRequest,
    ArtifactDisplayRequest as CoreArtifactDisplayRequest,
    ArtifactReviewRequest as CoreArtifactReviewRequest,
    AskUserRequest as CoreAskUserRequest,
    ConfirmationRequest as CoreConfirmationRequest,
    FrontendRequest as CoreFrontendRequest,
    MCPApprovalRequest as CoreMCPApprovalRequest,
)
from core.interface.snapshot import SessionSnapshot as CoreSessionSnapshot


ROOT = Path(__file__).resolve().parents[1]

MIRA_EXPORTS = {"MiraApplication", "MiraSession"}
API_EXPORTS = {
    "ApprovalRequest",
    "ArtifactDisplayRequest",
    "ArtifactEvent",
    "ArtifactReviewRequest",
    "AskUserRequest",
    "CompactionEvent",
    "ConfirmationRequest",
    "Frontend",
    "FrontendEvent",
    "FrontendRequest",
    "InformationEvent",
    "MCPApprovalRequest",
    "MCPEvent",
    "MessageEvent",
    "RubricEvent",
    "RuntimeEvent",
    "SessionSnapshot",
    "SubagentEvent",
    "ToolEvent",
    "UsageEvent",
}


class PublicAPITests(unittest.TestCase):
    def test_application_facade_exports_only_supported_classes(self) -> None:
        self.assertEqual(set(mira.__all__), MIRA_EXPORTS)
        self.assertIs(mira.MiraApplication, CoreApplication)
        self.assertIs(mira.MiraSession, CoreSession)
        for helper in (
            "initial_mode",
            "refresh_agent_specs",
            "tool_specs",
            "normalize_tool_specs",
            "available_tools",
            "resources_for",
            "DEFAULT_TOOL_SPECS",
        ):
            self.assertFalse(hasattr(mira, helper), helper)

    def test_frontend_facade_has_exact_intended_surface(self) -> None:
        self.assertEqual(set(public_api.__all__), API_EXPORTS)
        self.assertFalse(hasattr(public_api, "FrontendEmitter"))
        self.assertFalse(hasattr(public_api, "NullFrontend"))
        self.assertFalse(hasattr(public_api, "APPROVAL_CONSEQUENCE"))

    def test_frontend_facade_reexports_identical_interface_types(self) -> None:
        expected = {
            "Frontend": CoreFrontend,
            "FrontendEvent": CoreFrontendEvent,
            "FrontendRequest": CoreFrontendRequest,
            "MessageEvent": CoreMessageEvent,
            "ToolEvent": CoreToolEvent,
            "SubagentEvent": CoreSubagentEvent,
            "RuntimeEvent": CoreRuntimeEvent,
            "UsageEvent": CoreUsageEvent,
            "CompactionEvent": CoreCompactionEvent,
            "ArtifactEvent": CoreArtifactEvent,
            "RubricEvent": CoreRubricEvent,
            "MCPEvent": CoreMCPEvent,
            "InformationEvent": CoreInformationEvent,
            "ApprovalRequest": CoreApprovalRequest,
            "AskUserRequest": CoreAskUserRequest,
            "ArtifactReviewRequest": CoreArtifactReviewRequest,
            "ArtifactDisplayRequest": CoreArtifactDisplayRequest,
            "MCPApprovalRequest": CoreMCPApprovalRequest,
            "ConfirmationRequest": CoreConfirmationRequest,
            "SessionSnapshot": CoreSessionSnapshot,
        }
        for name, core_type in expected.items():
            with self.subTest(name=name):
                self.assertIs(getattr(public_api, name), core_type)

    def test_owned_ui_consumes_public_api_at_the_boundary(self) -> None:
        adapter = (ROOT / "ui" / "shared" / "adapter.py").read_text(encoding="utf-8")
        textual = (ROOT / "ui" / "textual" / "app.py").read_text(encoding="utf-8")
        textual_adapter = (ROOT / "ui" / "textual" / "adapter.py").read_text(
            encoding="utf-8"
        )
        terminal_adapter = (ROOT / "ui" / "terminal" / "adapter.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("from mira.api import", adapter)
        self.assertNotIn("from core.interface", adapter)
        self.assertIn("from mira import MiraApplication, MiraSession", textual)
        for owned_adapter in (textual_adapter, terminal_adapter):
            self.assertIn("from ui.shared.adapter import RendererAdapter", owned_adapter)
            self.assertNotIn("from core.interface", owned_adapter)

    def test_dependency_direction_and_future_consumer_placeholders(self) -> None:
        offenders: list[str] = []
        for path in (ROOT / "core").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                module = node.module if isinstance(node, ast.ImportFrom) else ""
                names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else []
                roots = {(module or "").split(".", 1)[0], *(name.split(".", 1)[0] for name in names)}
                if roots & {"mira", "ui", "protocols"}:
                    offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])
        self.assertFalse((ROOT / "protocols" / "acp").exists())
        self.assertFalse((ROOT / "ui" / "qt").exists())

    def test_example_imports_without_starting_a_live_application(self) -> None:
        namespace = runpy.run_path(str(ROOT / "examples" / "frontend.py"), run_name="frontend_example")
        self.assertIn("ExampleFrontend", namespace)
        self.assertTrue(callable(namespace["main"]))

    def test_wheel_configuration_includes_public_package(self) -> None:
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('"mira"', pyproject)


if __name__ == "__main__":
    unittest.main()
