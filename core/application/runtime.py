"""Build and install workspace runtime changes without frontend participation."""

from __future__ import annotations

from typing import Any


async def reconstruct(application: Any, config: dict[str, Any], *, inspect_model: bool = True, policy_only: bool = False, metadata: Any = None, preserve_model: bool = False) -> None:
    """Prepare every pair before replacing resources used by open sessions."""
    from agent.llm import active_model_issues, get_llm, get_model_name, model_unavailable_message
    from agent.resources import build_resources, configure_subagents
    from agent.subagents.discovery import subagent_model_issues
    from config.metadata import ModelMetadata, infer_model_metadata
    from config.settings import MAIN_MODEL, model_assignment, rubric_enabled, rubric_max_iterations
    from core.diagnostics.issues import unique_issues
    from session.dashboard import ensure_dashboard
    from core.application.app import build_agent_pair, refresh_agent_specs

    resources = build_resources(
        application.workspace, settings=config.get("settings"), config=None,
        mcp_manager=application.mcp_manager,
    )
    if not policy_only and resources.subagent_discovery.complete and config.get("settings_valid", True):
        from config.settings import prune_subagent_settings, save_settings

        settings = config.get("settings") or {}
        pruned = prune_subagent_settings(settings, {item.name for item in resources.subagent_discovery.items})
        if settings != pruned and save_settings(application.workspace, pruned):
            config["settings"] = pruned
    assignment_issues = [] if policy_only or preserve_model else [*active_model_issues(config), *subagent_model_issues(resources.subagent_discovery, config)]
    blocked = (getattr(application, "agent", None) is None) if preserve_model else not config.get("settings_valid", True) or bool(assignment_issues)
    metadata = metadata or getattr(application, "agent_metadata", None) or ModelMetadata(
        context_tokens=int(application.context_limit_tokens or 0) if policy_only else int((config.get("settings") or {}).get("models", {}).get("context_limit_tokens", 32768)),
        context_source=application.context_limit_source if policy_only else "settings.models.context_limit_tokens",
    )
    action_resources = None
    if not blocked:
        if inspect_model:
            metadata = await infer_model_metadata(config, model=get_llm(config, metadata=ModelMetadata()))
        action_resources = configure_subagents(resources, config)
    pairs = []
    owners = [session for session in application._sessions.values() if session.runtime_state != "closed"]
    for session in owners or ([] if policy_only else [None]):
        pair = (None, None) if blocked else build_agent_pair(
            config=config, workspace=application.workspace,
            checkpointer=session.checkpointer if session is not None else application.checkpointer,
            metadata=metadata, mcp_manager=application.mcp_manager, action_resources=action_resources,
        )
        pairs.append((session, pair))
    manager = application.mcp_manager
    issues = list(getattr(application, "issues", [])) if policy_only else unique_issues([
        *(config.get("issues") or []), *(manager.issues if manager is not None else []),
        *(manager.prompt_registry.issues if manager is not None else []),
        *resources.issues, *assignment_issues,
        *(issue for _, (agent, _) in pairs for issue in getattr(agent, "mira_resource_issues", [])),
    ])
    application.config = config
    application.agent_metadata = metadata
    application.context_limit_tokens = metadata.context_tokens
    application.context_limit_source = metadata.context_source
    if not policy_only:
        application.model_name = get_model_name(config)
    application.action_resources = action_resources
    application.resource_metadata = resources.metadata
    application.project_backend = resources.project_backend
    application.tool_failures = resources.tool_failures
    application.issues = issues
    application.agent_unavailable_message = model_unavailable_message(config) if blocked else ""
    for session, (agent, plan_agent) in pairs:
        application.agent, application.plan_agent = agent, plan_agent
        if session is None:
            continue
        if session.recovery is not None:
            if session.recovery["agent"] is session.plan_agent:
                session.recovery["agent"] = plan_agent
            elif session.recovery["agent"] is session.agent:
                session.recovery["agent"] = agent
        session.agent, session.plan_agent = agent, plan_agent
        refresh_agent_specs(session.mode, agent, plan_agent)
        session.mode.update(resources=resources.metadata, rubric_enabled=rubric_enabled(config.get("settings")),
                            rubric_max_iterations=rubric_max_iterations(config.get("settings")))
        ensure_dashboard(session.record, model_name=application.model_name,
                         context_limit_tokens=metadata.context_tokens, context_limit_source=metadata.context_source)
        application.store.save(session.record)
    application.effective_model_alias = str(model_assignment(config, MAIN_MODEL) or "") if getattr(application, "agent", None) is not None else ""
    if not policy_only:
        application.reload_workflows()
    application._runtime_changed()
