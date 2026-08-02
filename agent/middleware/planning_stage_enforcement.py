"""Plan/Goal stage-specific model-tool exposure and execution guards."""

from __future__ import annotations

from typing import Any, Literal, NotRequired

from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain_core.messages import ToolMessage

from agent.planning.policy import (
    PLANNING_STAGE_GOAL_FINALIZE,
    PLANNING_STAGE_GOAL_RESEARCH,
    PLANNING_STAGE_PLAN_FINALIZE,
    PLANNING_STAGE_PLAN_RESEARCH,
    PREPARE_GOAL_TOOL,
    PREPARE_PLAN_TOOL,
    PRESENT_GOAL_TOOL,
    PRESENT_PLAN_TOOL,
)
from agent.tools.specs import tool_name as resource_tool_name

CONTROL_TOOL_STAGES = {
    PREPARE_PLAN_TOOL: PLANNING_STAGE_PLAN_RESEARCH,
    PRESENT_PLAN_TOOL: PLANNING_STAGE_PLAN_FINALIZE,
    PREPARE_GOAL_TOOL: PLANNING_STAGE_GOAL_RESEARCH,
    PRESENT_GOAL_TOOL: PLANNING_STAGE_GOAL_FINALIZE,
}


class PlanningStageState(AgentState):
    """Checkpointed stage for Plan/Goal construction tool visibility."""

    planning_stage: NotRequired[
        Literal["plan_research", "plan_finalize", "goal_research", "goal_finalize"]
    ]


class PlanningStageEnforcementMiddleware(AgentMiddleware[PlanningStageState, Any, Any]):
    """Expose and execute only the formal controls valid for the current stage."""

    state_schema = PlanningStageState

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        """Filter and constrain synchronous planning model calls."""
        return handler(self._stage_request(request))

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        """Filter and constrain asynchronous planning model calls."""
        return await handler(self._stage_request(request))

    def wrap_tool_call(self, request: Any, handler: Any) -> Any:
        """Return a native tool error before a wrong-stage control can interrupt."""
        error = self._control_tool_error(request)
        return error if error is not None else handler(request)

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        """Async variant of wrap_tool_call."""
        error = self._control_tool_error(request)
        return error if error is not None else await handler(request)

    def _stage_request(self, request: Any) -> Any:
        stage = str(request.state.get("planning_stage") or PLANNING_STAGE_PLAN_RESEARCH)
        if stage in {PLANNING_STAGE_PLAN_FINALIZE, PLANNING_STAGE_GOAL_FINALIZE}:
            expected_present = (
                PRESENT_GOAL_TOOL if stage == PLANNING_STAGE_GOAL_FINALIZE else PRESENT_PLAN_TOOL
            )
            tools = [tool for tool in request.tools if resource_tool_name(tool) == expected_present]
            if not tools:
                raise RuntimeError(f"formal finalization requires {expected_present}")
            # With one visible tool, ``required`` is deterministic and portable
            # across OpenAI-compatible providers that reject named-tool objects.
            return request.override(tools=tools, tool_choice="required")

        expected_prepare = (
            PREPARE_GOAL_TOOL if stage == PLANNING_STAGE_GOAL_RESEARCH else PREPARE_PLAN_TOOL
        )
        hidden_controls = {
            PREPARE_PLAN_TOOL,
            PREPARE_GOAL_TOOL,
            PRESENT_PLAN_TOOL,
            PRESENT_GOAL_TOOL,
        } - {expected_prepare}
        tools = [
            tool for tool in request.tools if resource_tool_name(tool) not in hidden_controls
        ]
        return request.override(tools=tools, tool_choice=None)

    @staticmethod
    def _control_tool_error(request: Any) -> ToolMessage | None:
        call = request.tool_call
        name = str(call.get("name") or "")
        expected_stage = CONTROL_TOOL_STAGES.get(name)
        if expected_stage is None:
            return None
        current_stage = str(request.state.get("planning_stage") or PLANNING_STAGE_PLAN_RESEARCH)
        if current_stage == expected_stage:
            return None

        return ToolMessage(
            content=planning_control_tool_error(name, current_stage, expected_stage),
            name=name,
            tool_call_id=str(call.get("id") or ""),
            status="error",
        )


def planning_control_tool_error(name: str, current_stage: str, expected_stage: str) -> str:
    """Return actionable feedback for a wrong-stage formal control call."""
    if current_stage == PLANNING_STAGE_PLAN_RESEARCH:
        guidance = (
            "Call plan_show to display the retained Plan, or prepare_plan to construct "
            "a new or revised Plan. present_plan is finalization-only."
        )
    elif current_stage == PLANNING_STAGE_GOAL_RESEARCH:
        guidance = (
            "Call goal_show to display the retained Goal, or prepare_goal to construct "
            "a new or revised Goal. present_goal is finalization-only."
        )
    elif current_stage == PLANNING_STAGE_PLAN_FINALIZE:
        guidance = "Plan finalization requires present_plan."
    elif current_stage == PLANNING_STAGE_GOAL_FINALIZE:
        guidance = "Goal finalization requires present_goal."
    else:
        guidance = "Use only a control tool exposed for the current workflow stage."
    return (
        f"{name} is unavailable during {current_stage}; it requires {expected_stage}. "
        f"{guidance}"
    )


__all__ = [
    "CONTROL_TOOL_STAGES",
    "PlanningStageEnforcementMiddleware",
    "PlanningStageState",
    "planning_control_tool_error",
]
