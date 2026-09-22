"""Public facade for building native LangGraph workflows with MIRA."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from agent.execution.context import MiraContext


class Inherit:
    """Type of the public ``INHERIT`` workflow-specialization sentinel."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "INHERIT"


INHERIT = Inherit()


class MiraWorkflowAPI:
    """Application-owned build-time facade for native LangGraph workflows."""

    __slots__ = ("_agent_provider",)

    def __init__(self, agent_provider: Callable[[], Any]) -> None:
        self._agent_provider = agent_provider

    @property
    def context(self) -> MiraContext:
        """Return a context backed by the application's current MIRA runtime."""
        agent = self._current_agent()
        factory = getattr(agent, "mira_context_factory", None)
        if not callable(factory):
            raise RuntimeError("The current MIRA agent has no execution-context factory.")
        return factory()

    def agent(
        self,
        base: str = "general-purpose",
        *,
        name: str | None = None,
        tools: Sequence[str | BaseTool] | None = None,
        system_prompt: str | None | Inherit = INHERIT,
        response_format: Any | None | Inherit = INHERIT,
    ) -> Runnable[Any, Any]:
        """Create a workflow-local specialization of a configured MIRA subagent.

        Parameters
        ----------
        base
            Name of the configured MIRA subagent to specialize. The default,
            ``"general-purpose"``, clones MIRA's effective general-purpose
            definition.
        name
            Optional workflow-local identity. Changing it affects only this
            runnable and never mutates the reusable base definition.
        tools
            Tool surface for the specialization. ``None`` inherits the base
            surface, ``[]`` exposes no tools, and a non-empty sequence is an
            exact replacement allowlist. String names resolve through MIRA's
            existing built-in, project, and MCP tool resolver; concrete
            ``BaseTool`` objects are also accepted.
        system_prompt
            ``INHERIT`` keeps the base prompt. A string replaces it, while
            ``None`` explicitly clears it using the underlying LangChain API.
        response_format
            ``INHERIT`` keeps the base response format. ``None`` explicitly
            disables inherited structured output; a schema, response strategy,
            model type, or supported dictionary replaces it.

        Returns
        -------
        Runnable
            A normal LangChain runnable suitable for direct use as a LangGraph
            node when the graph state follows the agent ``messages`` contract.
            Workflows with domain-shaped state should use an ordinary node
            adapter that maps inputs to messages and maps the returned text or
            ``structured_response`` back into domain state.

        Raises
        ------
        KeyError
            If ``base`` is not an available configured MIRA subagent.
        ValueError
            If an exact tool replacement contains missing, duplicate, disabled,
            or ambiguous names.
        TypeError
            If the base is opaque, remote, or already compiled and the requested
            specialization cannot be applied safely.

        Notes
        -----
        The selected base is copied and is never mutated. Independent calls
        therefore produce independent workflow-local definitions. The model is
        inherited from the selected base MIRA subagent and cannot be overridden
        by ``mira.agent``. To use a different model, define or configure another
        MIRA subagent with that model and select it through ``base``.
        """
        registry = getattr(self._current_agent(), "mira_workflow_agents", None)
        if registry is None:
            raise RuntimeError("The current MIRA agent has no workflow-agent registry.")
        return registry.specialize(
            base,
            name=name,
            tools=tools,
            system_prompt=system_prompt,
            response_format=response_format,
            inherit=INHERIT,
        )

    def _current_agent(self) -> Any:
        agent = self._agent_provider()
        if agent is None:
            raise RuntimeError("MIRA workflows are unavailable because no action agent is active.")
        return agent


__all__ = ["INHERIT", "Inherit", "MiraWorkflowAPI"]
