"""Native LangGraph workflow integration for MIRA."""

from agent.workflows.api import INHERIT, Inherit, MiraWorkflowAPI
from agent.workflows.discovery import WorkflowRegistry, WorkflowSpec, discover_workflows

__all__ = [
    "INHERIT",
    "Inherit",
    "MiraWorkflowAPI",
    "WorkflowRegistry",
    "WorkflowSpec",
    "discover_workflows",
]
