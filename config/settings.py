"""Workspace settings stored under .mira/settings.yml."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from config.llm import DEFAULT_CONTEXT_TOKENS
from core.diagnostics.issues import Issue

SETTINGS_FILE = "settings.yml"
TRACING = "tracing"
TRACING_DEFAULT_PROFILE = "phoenix"
MIDDLEWARE_SPANS = "middleware_spans"
MIDDLEWARE_SPAN_MODES = ("hidden", "full")
MIDDLEWARE_SPANS_DEFAULT = "hidden"
EXECUTE_TOOL = "execute"
DELETE_TOOL = "delete"
DYNAMIC_SUBAGENTS = "dynamic_subagents"
DYNAMIC_SUBAGENT_RESPONSE_SCHEMA = "response_schema"
PLANNING_TODOS = "planning_todos"
PLANNING_RESPONSE_STATUS = "planning_response_status"
PLANNING_RESPONSE_STATUS_MAX_RETRIES = "max_retries"
PLANNING_RESPONSE_STATUS_MAX_RETRIES_LIMIT = 20
RUBRIC = "rubric"
RUBRIC_MAX_ITERATIONS = "max_iterations"
RUBRIC_MAX_ITERATIONS_LIMIT = 20
MODELS = "models"
CONTEXT_LIMIT_TOKENS = "context_limit_tokens"
MAIN_MODEL = "main"
RUBRIC_MODEL = "rubric"
SUMMARIZATION_MODEL = "summarization"
SUBAGENT_MODELS = "subagents"
EXECUTE_ENV_MODES = ("system", "conda_name", "conda_prefix", "venv")
READ_ONLY_BUILTIN_TOOLS = ("ls", "read_file", "glob", "grep")
INBUILT_DANGEROUS_TOOLS = ("write_file", "edit_file", DELETE_TOOL, EXECUTE_TOOL, "eval", "task")
INBUILT_TOOLS = (*READ_ONLY_BUILTIN_TOOLS, *INBUILT_DANGEROUS_TOOLS)
PTC_INAPPLICABLE_TOOLS = frozenset({"eval", "task"})


@dataclass(frozen=True, slots=True)
class ToolPolicy:
    """One canonical policy shape shared by Custom and MCP tools."""

    enabled: bool = True
    always_allow: bool = False
    plan_access: bool = False
    ptc: bool = False
    rubric: bool = False


@dataclass(frozen=True, slots=True)
class SettingsLoadResult:
    """Effective settings plus non-fatal source diagnostics."""

    settings: dict[str, Any]
    issues: tuple[Issue, ...] = ()
    valid: bool = True


DEFAULT_SETTINGS: dict[str, Any] = {
    TRACING: {
        "enabled": False,
        "profile": TRACING_DEFAULT_PROFILE,
        MIDDLEWARE_SPANS: MIDDLEWARE_SPANS_DEFAULT,
    },
    MODELS: {
        CONTEXT_LIMIT_TOKENS: DEFAULT_CONTEXT_TOKENS,
        MAIN_MODEL: None,
        RUBRIC_MODEL: None,
        SUMMARIZATION_MODEL: None,
        SUBAGENT_MODELS: {},
    },
    "system": {
        DYNAMIC_SUBAGENTS: {
            "enabled": False,
            DYNAMIC_SUBAGENT_RESPONSE_SCHEMA: True,
        },
        PLANNING_TODOS: {
            "enabled": False,
        },
        PLANNING_RESPONSE_STATUS: {
            PLANNING_RESPONSE_STATUS_MAX_RETRIES: 2,
        },
        RUBRIC: {
            "enabled": False,
            RUBRIC_MAX_ITERATIONS: 3,
        },
    },
    "hitl": {
        "git_protection": {"enabled": True},
        "execute_env": {
            "mode": "system",
            "name": "",
            "prefix": "",
            "path": "",
            "allow": [],
        },
        "tools": {
            "ls": {"enabled": True},
            "read_file": {"enabled": True},
            "glob": {"enabled": True},
            "grep": {"enabled": True},
            "write_file": {"enabled": True, "always_allow": False, "ptc": False},
            "edit_file": {"enabled": True, "always_allow": False, "ptc": False},
            DELETE_TOOL: {"enabled": True, "always_allow": False, "ptc": False},
            "execute": {"enabled": False, "always_allow": False, "ptc": False, "rubric": False},
            "eval": {"enabled": True, "always_allow": False},
            "task": {"enabled": True, "always_allow": False},
        },
    }
}


def settings_path(workspace: Path) -> Path:
    """Return the workspace-local settings path."""
    return workspace.expanduser().resolve() / ".mira" / SETTINGS_FILE


def load_settings_result(workspace: Path) -> SettingsLoadResult:
    """Load settings without preventing independent startup diagnostics."""
    path = settings_path(workspace)
    if not path.exists():
        return SettingsLoadResult(deepcopy(DEFAULT_SETTINGS))
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        issue = Issue(
            "STARTUP",
            "Invalid settings.yml",
            ".mira/settings.yml",
            f"{type(exc).__name__}: {exc}",
            "Fix .mira/settings.yml and run /reload.",
        )
        return SettingsLoadResult(deepcopy(DEFAULT_SETTINGS), (issue,), False)
    issues = tuple(settings_issues(raw))
    normalized = normalize_settings(raw)
    return SettingsLoadResult(normalized, issues, not issues)


def load_settings(workspace: Path) -> dict[str, Any]:
    """Return effective settings, using defaults for omitted current fields."""
    return load_settings_result(workspace).settings


def save_settings(workspace: Path, settings: dict[str, Any]) -> bool:
    """Persist normalized workspace settings."""
    path = settings_path(workspace)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(normalize_settings(settings), sort_keys=False),
            encoding="utf-8",
        )
    except OSError:
        return False
    return True


def normalize_settings(raw: Any) -> dict[str, Any]:
    """Return settings with defaults and only supported HITL shapes."""
    settings = deepcopy(DEFAULT_SETTINGS)
    if not isinstance(raw, dict):
        return settings

    tracing = raw.get(TRACING)
    if isinstance(tracing, dict):
        if isinstance(tracing.get("enabled"), bool):
            settings[TRACING]["enabled"] = tracing["enabled"]
        profile = tracing.get("profile")
        if isinstance(profile, str) and profile.strip():
            settings[TRACING]["profile"] = profile.strip()
        middleware_spans = tracing.get(MIDDLEWARE_SPANS)
        if middleware_spans in MIDDLEWARE_SPAN_MODES:
            settings[TRACING][MIDDLEWARE_SPANS] = middleware_spans

    models = raw.get(MODELS)
    if isinstance(models, dict):
        context_limit = models.get(CONTEXT_LIMIT_TOKENS)
        if valid_context_limit_tokens(context_limit):
            settings[MODELS][CONTEXT_LIMIT_TOKENS] = context_limit
        for key in (MAIN_MODEL, RUBRIC_MODEL, SUMMARIZATION_MODEL):
            value = models.get(key)
            if value is None or isinstance(value, str):
                settings[MODELS][key] = value.strip() if isinstance(value, str) and value.strip() else None
        raw_subagents = models.get(SUBAGENT_MODELS)
        if isinstance(raw_subagents, dict):
            subagents: dict[str, dict[str, Any]] = {}
            for name, spec in raw_subagents.items():
                if not isinstance(name, str) or not name.strip() or not isinstance(spec, dict):
                    continue
                normalized_spec: dict[str, Any] = {}
                if isinstance(spec.get("enabled"), bool):
                    normalized_spec["enabled"] = spec["enabled"]
                model = spec.get("model")
                if model is None or isinstance(model, str):
                    normalized_spec["model"] = model.strip() if isinstance(model, str) and model.strip() else None
                subagents[name] = normalized_spec
            settings[MODELS][SUBAGENT_MODELS] = subagents

    system = raw.get("system")
    if isinstance(system, dict):
        dynamic_subagents = system.get(DYNAMIC_SUBAGENTS)
        if isinstance(dynamic_subagents, dict):
            if isinstance(dynamic_subagents.get("enabled"), bool):
                settings["system"][DYNAMIC_SUBAGENTS]["enabled"] = dynamic_subagents["enabled"]
            if isinstance(dynamic_subagents.get(DYNAMIC_SUBAGENT_RESPONSE_SCHEMA), bool):
                settings["system"][DYNAMIC_SUBAGENTS][DYNAMIC_SUBAGENT_RESPONSE_SCHEMA] = dynamic_subagents[
                    DYNAMIC_SUBAGENT_RESPONSE_SCHEMA
                ]
        planning_todos = system.get(PLANNING_TODOS)
        if isinstance(planning_todos, dict) and isinstance(planning_todos.get("enabled"), bool):
            settings["system"][PLANNING_TODOS]["enabled"] = planning_todos["enabled"]
        planning_response_status = system.get(PLANNING_RESPONSE_STATUS)
        if isinstance(planning_response_status, dict):
            retries = planning_response_status.get(PLANNING_RESPONSE_STATUS_MAX_RETRIES)
            if valid_planning_response_status_max_retries(retries):
                settings["system"][PLANNING_RESPONSE_STATUS][
                    PLANNING_RESPONSE_STATUS_MAX_RETRIES
                ] = retries
        rubric = system.get(RUBRIC)
        if isinstance(rubric, dict):
            if isinstance(rubric.get("enabled"), bool):
                settings["system"][RUBRIC]["enabled"] = rubric["enabled"]
            iterations = rubric.get(RUBRIC_MAX_ITERATIONS)
            if valid_rubric_max_iterations(iterations):
                settings["system"][RUBRIC][RUBRIC_MAX_ITERATIONS] = iterations

    hitl = raw.get("hitl")
    if not isinstance(hitl, dict):
        return settings

    git_protection = hitl.get("git_protection")
    if isinstance(git_protection, dict) and isinstance(git_protection.get("enabled"), bool):
        settings["hitl"]["git_protection"]["enabled"] = git_protection["enabled"]

    execute_env = hitl.get("execute_env")
    if isinstance(execute_env, dict):
        settings["hitl"]["execute_env"] = normalize_execute_env(execute_env)

    tools = hitl.get("tools")
    if isinstance(tools, dict):
        normalized_tools = {name: dict(spec) for name, spec in settings["hitl"]["tools"].items()}
        for name, spec in tools.items():
            if not isinstance(name, str) or not name.strip() or not isinstance(spec, dict):
                continue
            if name in READ_ONLY_BUILTIN_TOOLS:
                if isinstance(spec.get("enabled"), bool):
                    normalized_tools[name]["enabled"] = spec["enabled"]
                continue
            always_allow = spec.get("always_allow")
            enabled = spec.get("enabled")
            current = dict(
                normalized_tools.get(
                    name,
                    {"enabled": True, "always_allow": False, "ptc": False},
                )
            )
            if isinstance(enabled, bool):
                current["enabled"] = enabled
            if isinstance(always_allow, bool):
                current["always_allow"] = always_allow
            plan_access = spec.get("plan_access")
            if name not in INBUILT_DANGEROUS_TOOLS and isinstance(plan_access, bool):
                current["plan_access"] = plan_access
            ptc = spec.get("ptc")
            if name not in PTC_INAPPLICABLE_TOOLS and isinstance(ptc, bool):
                current["ptc"] = ptc
            rubric = spec.get("rubric")
            if name not in INBUILT_DANGEROUS_TOOLS or name == EXECUTE_TOOL:
                if isinstance(rubric, bool):
                    current["rubric"] = rubric
            normalized_tools[name] = current
        settings["hitl"]["tools"] = normalized_tools

    mcp = raw.get("mcp")
    if isinstance(mcp, dict) and isinstance(mcp.get("servers"), dict):
        normalized_servers: dict[str, Any] = {}
        for name, spec in mcp["servers"].items():
            if not isinstance(name, str) or not name.strip() or not isinstance(spec, dict):
                continue
            server: dict[str, Any] = {}
            for key in ("enabled", "always_allow"):
                if isinstance(spec.get(key), bool):
                    server[key] = spec[key]
            fingerprint = spec.get("approved_fingerprint")
            if isinstance(fingerprint, str) and _valid_fingerprint(fingerprint):
                server["approved_fingerprint"] = fingerprint
            if isinstance(spec.get("tools"), dict):
                tool_specs: dict[str, Any] = {}
                for tool_name, tool_spec in spec["tools"].items():
                    if not isinstance(tool_name, str) or not tool_name or not isinstance(tool_spec, dict):
                        continue
                    normalized_tool = {
                        key: tool_spec[key]
                        for key in ("enabled", "always_allow", "plan_access", "ptc", "rubric")
                        if isinstance(tool_spec.get(key), bool)
                    }
                    if normalized_tool:
                        tool_specs[tool_name] = normalized_tool
                if tool_specs:
                    server["tools"] = tool_specs
            normalized_servers[name] = server
        if normalized_servers:
            settings["mcp"] = {"servers": normalized_servers}

    normalized_tools = settings["hitl"]["tools"]
    eval_enabled = bool(normalized_tools.get("eval", {}).get("enabled", True))
    task_enabled = bool(normalized_tools.get("task", {}).get("enabled", True))
    if not eval_enabled or not task_enabled:
        settings["system"][DYNAMIC_SUBAGENTS]["enabled"] = False

    return settings


def settings_issues(raw: Any) -> list[Issue]:
    """Return one Issue per unknown or invalid current-schema setting."""
    issues: list[Issue] = []
    if not isinstance(raw, dict):
        return [_invalid_setting("settings", "top level must be a mapping")]
    _unknown_settings(raw, {TRACING, MODELS, "system", "hitl", "mcp"}, "", issues)

    tracing = raw.get(TRACING)
    if tracing is not None:
        if not isinstance(tracing, dict):
            issues.append(_invalid_setting(TRACING, "must be a mapping"))
        else:
            _unknown_settings(tracing, {"enabled", "profile", MIDDLEWARE_SPANS}, TRACING, issues)
            if "enabled" in tracing and not isinstance(tracing["enabled"], bool):
                issues.append(_invalid_setting(f"{TRACING}.enabled", "must be true or false"))
            if "profile" in tracing and (
                not isinstance(tracing["profile"], str) or not tracing["profile"].strip()
            ):
                issues.append(_invalid_setting(f"{TRACING}.profile", "must be a non-empty string"))
            if MIDDLEWARE_SPANS in tracing and tracing[MIDDLEWARE_SPANS] not in MIDDLEWARE_SPAN_MODES:
                issues.append(
                    _invalid_setting(
                        f"{TRACING}.{MIDDLEWARE_SPANS}",
                        "must be hidden or full",
                    )
                )

    models = raw.get(MODELS)
    if models is not None:
        if not isinstance(models, dict):
            issues.append(_invalid_setting(MODELS, "must be a mapping"))
        else:
            _unknown_settings(
                models,
                {CONTEXT_LIMIT_TOKENS, MAIN_MODEL, RUBRIC_MODEL, SUMMARIZATION_MODEL, SUBAGENT_MODELS},
                MODELS,
                issues,
            )
            if CONTEXT_LIMIT_TOKENS in models and not valid_context_limit_tokens(models[CONTEXT_LIMIT_TOKENS]):
                issues.append(_invalid_setting(f"{MODELS}.{CONTEXT_LIMIT_TOKENS}", "must be a positive integer"))
            for key in (MAIN_MODEL, RUBRIC_MODEL, SUMMARIZATION_MODEL):
                if key in models and models[key] is not None and not isinstance(models[key], str):
                    issues.append(_invalid_setting(f"{MODELS}.{key}", "must be a profile name or null"))
            subagents = models.get(SUBAGENT_MODELS)
            if subagents is not None:
                if not isinstance(subagents, dict):
                    issues.append(_invalid_setting(f"{MODELS}.{SUBAGENT_MODELS}", "must be a mapping"))
                else:
                    for name, spec in subagents.items():
                        path = f"{MODELS}.{SUBAGENT_MODELS}.{name}"
                        if not isinstance(name, str) or not name.strip() or not isinstance(spec, dict):
                            issues.append(_invalid_setting(path, "must be a named mapping"))
                            continue
                        _unknown_settings(spec, {"enabled", "model"}, path, issues)
                        if "enabled" in spec and not isinstance(spec["enabled"], bool):
                            issues.append(_invalid_setting(f"{path}.enabled", "must be true or false"))
                        if "model" in spec and spec["model"] is not None and not isinstance(spec["model"], str):
                            issues.append(_invalid_setting(f"{path}.model", "must be a profile name or null"))

    system = raw.get("system")
    if system is not None:
        if not isinstance(system, dict):
            issues.append(_invalid_setting("system", "must be a mapping"))
        else:
            system_shapes = {
                DYNAMIC_SUBAGENTS: {"enabled", DYNAMIC_SUBAGENT_RESPONSE_SCHEMA},
                PLANNING_TODOS: {"enabled"},
                PLANNING_RESPONSE_STATUS: {PLANNING_RESPONSE_STATUS_MAX_RETRIES},
                RUBRIC: {"enabled", RUBRIC_MAX_ITERATIONS},
            }
            _unknown_settings(system, set(system_shapes), "system", issues)
            for name, allowed in system_shapes.items():
                spec = system.get(name)
                if spec is None:
                    continue
                path = f"system.{name}"
                if not isinstance(spec, dict):
                    issues.append(_invalid_setting(path, "must be a mapping"))
                    continue
                _unknown_settings(spec, allowed, path, issues)
            _validate_boolean(system, DYNAMIC_SUBAGENTS, "enabled", issues)
            _validate_boolean(system, DYNAMIC_SUBAGENTS, DYNAMIC_SUBAGENT_RESPONSE_SCHEMA, issues)
            _validate_boolean(system, PLANNING_TODOS, "enabled", issues)
            _validate_boolean(system, RUBRIC, "enabled", issues)
            retries = (system.get(PLANNING_RESPONSE_STATUS) or {}).get(PLANNING_RESPONSE_STATUS_MAX_RETRIES) if isinstance(system.get(PLANNING_RESPONSE_STATUS), dict) else None
            if retries is not None and not valid_planning_response_status_max_retries(retries):
                issues.append(_invalid_setting(f"system.{PLANNING_RESPONSE_STATUS}.{PLANNING_RESPONSE_STATUS_MAX_RETRIES}", "is out of range"))
            iterations = (system.get(RUBRIC) or {}).get(RUBRIC_MAX_ITERATIONS) if isinstance(system.get(RUBRIC), dict) else None
            if iterations is not None and not valid_rubric_max_iterations(iterations):
                issues.append(_invalid_setting(f"system.{RUBRIC}.{RUBRIC_MAX_ITERATIONS}", "is out of range"))

    hitl = raw.get("hitl")
    if hitl is not None:
        if not isinstance(hitl, dict):
            issues.append(_invalid_setting("hitl", "must be a mapping"))
        else:
            _unknown_settings(hitl, {"git_protection", "execute_env", "tools"}, "hitl", issues)
            git = hitl.get("git_protection")
            if git is not None:
                if not isinstance(git, dict):
                    issues.append(_invalid_setting("hitl.git_protection", "must be a mapping"))
                else:
                    _unknown_settings(git, {"enabled"}, "hitl.git_protection", issues)
                    if "enabled" in git and not isinstance(git["enabled"], bool):
                        issues.append(_invalid_setting("hitl.git_protection.enabled", "must be true or false"))
            execute_env = hitl.get("execute_env")
            if execute_env is not None:
                if not isinstance(execute_env, dict):
                    issues.append(_invalid_setting("hitl.execute_env", "must be a mapping"))
                else:
                    _unknown_settings(execute_env, {"mode", "name", "prefix", "path", "allow"}, "hitl.execute_env", issues)
                    mode = execute_env.get("mode")
                    if mode is not None and (not isinstance(mode, str) or mode not in EXECUTE_ENV_MODES):
                        issues.append(_invalid_setting("hitl.execute_env.mode", "must name a supported environment mode"))
                    for key in ("name", "prefix", "path"):
                        if key in execute_env and not isinstance(execute_env[key], str):
                            issues.append(_invalid_setting(f"hitl.execute_env.{key}", "must be a string"))
                    allow = execute_env.get("allow")
                    if allow is not None and not (
                        isinstance(allow, list) and all(isinstance(item, str) for item in allow)
                    ):
                        issues.append(_invalid_setting("hitl.execute_env.allow", "must be a list of names"))
            tools = hitl.get("tools")
            if tools is not None:
                _validate_dynamic_policy_mapping(
                    tools,
                    "hitl.tools",
                    issues,
                    include_plan=True,
                    fixed_read_only_builtins=True,
                )

    mcp = raw.get("mcp")
    if mcp is not None:
        if not isinstance(mcp, dict):
            issues.append(_invalid_setting("mcp", "must be a mapping"))
        else:
            _unknown_settings(mcp, {"servers"}, "mcp", issues)
            servers = mcp.get("servers")
            if servers is not None:
                if not isinstance(servers, dict):
                    issues.append(_invalid_setting("mcp.servers", "must be a mapping"))
                else:
                    for name, spec in servers.items():
                        path = f"mcp.servers.{name}"
                        if not isinstance(name, str) or not name.strip() or not isinstance(spec, dict):
                            issues.append(_invalid_setting(path, "must be a named mapping"))
                            continue
                        _unknown_settings(spec, {"enabled", "always_allow", "approved_fingerprint", "tools"}, path, issues)
                        for key in ("enabled", "always_allow"):
                            if key in spec and not isinstance(spec[key], bool):
                                issues.append(_invalid_setting(f"{path}.{key}", "must be true or false"))
                        if "approved_fingerprint" in spec and not (
                            isinstance(spec["approved_fingerprint"], str) and _valid_fingerprint(spec["approved_fingerprint"])
                        ):
                            issues.append(_invalid_setting(f"{path}.approved_fingerprint", "must be a SHA-256 fingerprint"))
                        if "tools" in spec:
                            _validate_dynamic_policy_mapping(spec["tools"], f"{path}.tools", issues, include_plan=True)
    return issues


def _unknown_settings(raw: dict[Any, Any], allowed: set[str], prefix: str, issues: list[Issue]) -> None:
    for key in raw:
        if key in allowed:
            continue
        path = f"{prefix}.{key}" if prefix else str(key)
        issues.append(
            Issue(
                "STARTUP",
                f"Unsupported setting: {path}",
                ".mira/settings.yml",
                f"{path} is not part of the current settings schema.",
                "Remove the unsupported key and run /reload.",
            )
        )


def _invalid_setting(path: str, detail: str) -> Issue:
    return Issue(
        "STARTUP",
        f"Invalid setting: {path}",
        ".mira/settings.yml",
        f"{path} {detail}.",
        "Correct the setting and run /reload.",
    )


def _validate_boolean(system: dict[str, Any], section: str, key: str, issues: list[Issue]) -> None:
    spec = system.get(section)
    if isinstance(spec, dict) and key in spec and not isinstance(spec[key], bool):
        issues.append(_invalid_setting(f"system.{section}.{key}", "must be true or false"))


def _validate_dynamic_policy_mapping(
    raw: Any,
    path: str,
    issues: list[Issue],
    *,
    include_plan: bool,
    fixed_read_only_builtins: bool = False,
) -> None:
    if not isinstance(raw, dict):
        issues.append(_invalid_setting(path, "must be a mapping"))
        return
    allowed = {"enabled", "always_allow", "ptc", "rubric"}
    if include_plan:
        allowed.add("plan_access")
    for name, spec in raw.items():
        item_path = f"{path}.{name}"
        if not isinstance(name, str) or not name.strip() or not isinstance(spec, dict):
            issues.append(_invalid_setting(item_path, "must be a named mapping"))
            continue
        item_allowed = (
            {"enabled"}
            if fixed_read_only_builtins and name in READ_ONLY_BUILTIN_TOOLS
            else allowed
        )
        _unknown_settings(spec, item_allowed, item_path, issues)
        for key in item_allowed:
            if key in spec and not isinstance(spec[key], bool):
                issues.append(_invalid_setting(f"{item_path}.{key}", "must be true or false"))


def valid_context_limit_tokens(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def tracing_settings(config_or_settings: dict[str, Any] | None) -> dict[str, Any]:
    """Return the normalized generic OTLP tracing configuration."""
    settings = _settings_object(config_or_settings)
    return dict(normalize_settings(settings)[TRACING])


def tracing_enabled(config_or_settings: dict[str, Any] | None) -> bool:
    return bool(tracing_settings(config_or_settings)["enabled"])


def middleware_span_mode(config_or_settings: dict[str, Any] | None) -> str:
    """Return the normalized AgentMiddleware tracing mode."""
    return str(tracing_settings(config_or_settings)[MIDDLEWARE_SPANS])


def set_tracing_enabled(settings: dict[str, Any], enabled: bool) -> dict[str, Any]:
    current = tracing_settings(settings)
    current["enabled"] = bool(enabled)
    updated = normalize_settings(settings)
    updated[TRACING] = current
    return updated


def set_tracing_profile(settings: dict[str, Any], profile: str) -> dict[str, Any]:
    """Return settings with one selected tracing profile."""
    if not isinstance(profile, str) or not profile.strip():
        raise ValueError("tracing profile is required")
    current = tracing_settings(settings)
    current["profile"] = profile.strip()
    updated = normalize_settings(settings)
    updated[TRACING] = current
    return updated


def set_middleware_span_mode(settings: dict[str, Any], mode: str) -> dict[str, Any]:
    """Return settings with the selected AgentMiddleware tracing mode."""
    if mode not in MIDDLEWARE_SPAN_MODES:
        raise ValueError("middleware span mode must be hidden or full")
    current = tracing_settings(settings)
    current[MIDDLEWARE_SPANS] = mode
    updated = normalize_settings(settings)
    updated[TRACING] = current
    return updated


def model_settings(config_or_settings: dict[str, Any] | None) -> dict[str, Any]:
    settings = _settings_object(config_or_settings)
    normalized = normalize_settings(settings)
    return deepcopy(normalized[MODELS])


def context_limit_tokens(config_or_settings: dict[str, Any] | None) -> int:
    return int(model_settings(config_or_settings)[CONTEXT_LIMIT_TOKENS])


def model_assignment(config_or_settings: dict[str, Any] | None, role: str) -> str | None:
    if role not in {MAIN_MODEL, RUBRIC_MODEL, SUMMARIZATION_MODEL}:
        return None
    value = model_settings(config_or_settings).get(role)
    return str(value) if isinstance(value, str) and value else None


def set_context_limit_tokens(settings: dict[str, Any], value: Any) -> dict[str, Any]:
    updated = normalize_settings(settings)
    if valid_context_limit_tokens(value):
        updated[MODELS][CONTEXT_LIMIT_TOKENS] = value
    return updated


def set_model_assignment(settings: dict[str, Any], role: str, profile: str | None) -> dict[str, Any]:
    updated = normalize_settings(settings)
    if role in {MAIN_MODEL, RUBRIC_MODEL, SUMMARIZATION_MODEL}:
        updated[MODELS][role] = str(profile).strip() if profile is not None and str(profile).strip() else None
    return updated


def subagent_model_settings(config_or_settings: dict[str, Any] | None, name: str) -> dict[str, Any]:
    value = model_settings(config_or_settings).get(SUBAGENT_MODELS, {}).get(name, {})
    return dict(value) if isinstance(value, dict) else {}


def subagent_enabled(config_or_settings: dict[str, Any] | None, name: str) -> bool:
    if name == "general-purpose":
        return True
    return bool(subagent_model_settings(config_or_settings, name).get("enabled", False))


def subagent_model_assignment(config_or_settings: dict[str, Any] | None, name: str) -> str | None:
    value = subagent_model_settings(config_or_settings, name).get("model")
    return str(value) if isinstance(value, str) and value else None


def set_subagent_enabled(settings: dict[str, Any], name: str, enabled: bool) -> dict[str, Any]:
    updated = normalize_settings(settings)
    spec = dict(updated[MODELS][SUBAGENT_MODELS].get(name, {}))
    spec["enabled"] = True if name == "general-purpose" else bool(enabled)
    spec.setdefault("model", None)
    updated[MODELS][SUBAGENT_MODELS][name] = spec
    return updated


def set_subagent_model_assignment(settings: dict[str, Any], name: str, profile: str | None) -> dict[str, Any]:
    updated = normalize_settings(settings)
    spec = dict(updated[MODELS][SUBAGENT_MODELS].get(name, {}))
    spec.setdefault("enabled", name == "general-purpose")
    spec["model"] = str(profile).strip() if profile is not None and str(profile).strip() else None
    updated[MODELS][SUBAGENT_MODELS][name] = spec
    return updated


def prune_subagent_settings(settings: dict[str, Any], names: set[str]) -> dict[str, Any]:
    updated = normalize_settings(settings)
    current = updated[MODELS][SUBAGENT_MODELS]
    updated[MODELS][SUBAGENT_MODELS] = {name: spec for name, spec in current.items() if name in names}
    return updated


def normalize_execute_env(raw: Any) -> dict[str, Any]:
    """Return normalized project execute environment settings."""
    current = deepcopy(DEFAULT_SETTINGS["hitl"]["execute_env"])
    if not isinstance(raw, dict):
        return current

    mode = str(raw.get("mode") or "").strip()
    if mode in EXECUTE_ENV_MODES:
        current["mode"] = mode
    for key in ("name", "prefix", "path"):
        value = raw.get(key)
        if isinstance(value, str):
            current[key] = value.strip()

    allow = raw.get("allow")
    if isinstance(allow, str):
        values = allow.split(",")
    elif isinstance(allow, list):
        values = allow
    else:
        values = []
    current["allow"] = normalize_env_names(values)
    return current


def normalize_env_names(values: list[Any]) -> list[str]:
    """Return unique environment variable names, dropping values and wildcards."""
    names: list[str] = []
    for value in values:
        name = str(value or "").strip()
        if not name or "=" in name or name == "*":
            continue
        if not all(char.isalnum() or char == "_" for char in name):
            continue
        name = name.upper()
        if name not in names:
            names.append(name)
    return names


def hitl_settings(config_or_settings: dict[str, Any] | None) -> dict[str, Any]:
    """Extract the HITL section from a runtime config or settings object."""
    if not isinstance(config_or_settings, dict):
        return deepcopy(DEFAULT_SETTINGS["hitl"])
    settings = config_or_settings.get("settings", config_or_settings)
    return normalize_settings(settings).get("hitl", deepcopy(DEFAULT_SETTINGS["hitl"]))


def execute_env_settings(config_or_settings: dict[str, Any] | None) -> dict[str, Any]:
    """Return normalized execute environment settings."""
    hitl = hitl_settings(config_or_settings)
    return normalize_execute_env(hitl.get("execute_env"))


def git_protection_enabled(config_or_settings: dict[str, Any] | None) -> bool:
    """Return whether startup Git protection is enabled."""
    hitl = hitl_settings(config_or_settings)
    return bool(hitl.get("git_protection", {}).get("enabled", True))


def dynamic_subagents_enabled(config_or_settings: dict[str, Any] | None) -> bool:
    """Return whether eval may spawn dynamic subagents."""
    if not isinstance(config_or_settings, dict):
        return False
    settings = config_or_settings.get("settings", config_or_settings)
    normalized = normalize_settings(settings)
    configured = bool(normalized.get("system", {}).get(DYNAMIC_SUBAGENTS, {}).get("enabled", False))
    tools = normalized["hitl"]["tools"]
    return (
        configured
        and bool(tools.get("eval", {}).get("enabled", True))
        and bool(tools.get("task", {}).get("enabled", True))
    )


def set_dynamic_subagents(settings: dict[str, Any], enabled: bool) -> dict[str, Any]:
    """Return settings with dynamic eval subagents enabled or disabled."""
    updated = normalize_settings(settings)
    tools = updated["hitl"]["tools"]
    updated["system"][DYNAMIC_SUBAGENTS]["enabled"] = (
        bool(enabled)
        and bool(tools.get("eval", {}).get("enabled", True))
        and bool(tools.get("task", {}).get("enabled", True))
    )
    return updated


def dynamic_subagent_response_schema_enabled(config_or_settings: dict[str, Any] | None) -> bool:
    """Return whether eval may request dynamic subagent response schemas."""
    if not isinstance(config_or_settings, dict):
        return True
    settings = config_or_settings.get("settings", config_or_settings)
    normalized = normalize_settings(settings)
    return bool(
        normalized.get("system", {})
        .get(DYNAMIC_SUBAGENTS, {})
        .get(DYNAMIC_SUBAGENT_RESPONSE_SCHEMA, True)
    )


def set_dynamic_subagent_response_schema(settings: dict[str, Any], enabled: bool) -> dict[str, Any]:
    """Return settings with dynamic subagent response schemas configured."""
    updated = normalize_settings(settings)
    updated["system"][DYNAMIC_SUBAGENTS][DYNAMIC_SUBAGENT_RESPONSE_SCHEMA] = bool(enabled)
    return updated


def planning_todos_enabled(config_or_settings: dict[str, Any] | None) -> bool:
    """Return whether MIRA adds DeepAgents planning todos."""
    if not isinstance(config_or_settings, dict):
        return False
    settings = config_or_settings.get("settings", config_or_settings)
    normalized = normalize_settings(settings)
    return bool(normalized.get("system", {}).get(PLANNING_TODOS, {}).get("enabled", False))


def set_planning_todos(settings: dict[str, Any], enabled: bool) -> dict[str, Any]:
    """Return settings with planning todos enabled or disabled."""
    updated = normalize_settings(settings)
    updated["system"][PLANNING_TODOS]["enabled"] = bool(enabled)
    return updated


def planning_response_status_max_retries(config_or_settings: dict[str, Any] | None) -> int:
    """Return the configured Plan/Goal response-status retry cap."""
    if not isinstance(config_or_settings, dict):
        return 2
    settings = config_or_settings.get("settings", config_or_settings)
    normalized = normalize_settings(settings)
    return int(
        normalized.get("system", {})
        .get(PLANNING_RESPONSE_STATUS, {})
        .get(PLANNING_RESPONSE_STATUS_MAX_RETRIES, 2)
    )


def set_planning_response_status_max_retries(
    settings: dict[str, Any],
    value: Any,
) -> dict[str, Any]:
    """Return settings with a valid Plan/Goal response-status retry cap."""
    updated = normalize_settings(settings)
    if valid_planning_response_status_max_retries(value):
        updated["system"][PLANNING_RESPONSE_STATUS][PLANNING_RESPONSE_STATUS_MAX_RETRIES] = value
    return updated


def valid_planning_response_status_max_retries(value: Any) -> bool:
    """Return whether a Plan/Goal response-status retry cap is supported."""
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 1 <= value <= PLANNING_RESPONSE_STATUS_MAX_RETRIES_LIMIT
    )


def rubric_enabled(config_or_settings: dict[str, Any] | None) -> bool:
    """Return whether goal-driven rubric grading is enabled."""
    if not isinstance(config_or_settings, dict):
        return False
    settings = config_or_settings.get("settings", config_or_settings)
    normalized = normalize_settings(settings)
    return bool(normalized.get("system", {}).get(RUBRIC, {}).get("enabled", False))


def rubric_max_iterations(config_or_settings: dict[str, Any] | None) -> int:
    """Return the configured rubric grading iteration cap."""
    if not isinstance(config_or_settings, dict):
        return 3
    settings = config_or_settings.get("settings", config_or_settings)
    normalized = normalize_settings(settings)
    return int(normalized.get("system", {}).get(RUBRIC, {}).get(RUBRIC_MAX_ITERATIONS, 3))


def set_rubric_enabled(settings: dict[str, Any], enabled: bool) -> dict[str, Any]:
    """Return settings with rubric grading enabled or disabled."""
    updated = normalize_settings(settings)
    updated["system"][RUBRIC]["enabled"] = bool(enabled)
    return updated


def set_rubric_max_iterations(settings: dict[str, Any], value: Any) -> dict[str, Any]:
    """Return settings with a valid rubric iteration cap, preserving invalid input."""
    updated = normalize_settings(settings)
    if valid_rubric_max_iterations(value):
        updated["system"][RUBRIC][RUBRIC_MAX_ITERATIONS] = value
    return updated


def valid_rubric_max_iterations(value: Any) -> bool:
    """Return whether a value is supported by the minimum DeepAgents version."""
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 1 <= value <= RUBRIC_MAX_ITERATIONS_LIMIT
    )


def tool_always_allow(config_or_settings: dict[str, Any] | None, tool_name: str) -> bool:
    """Return whether a tool intrinsically or explicitly skips HITL approval."""
    if tool_name in READ_ONLY_BUILTIN_TOOLS:
        return True
    hitl = hitl_settings(config_or_settings)
    tools = hitl.get("tools", {})
    spec = tools.get(tool_name) if isinstance(tools, dict) else None
    if isinstance(spec, dict) and isinstance(spec.get("always_allow"), bool):
        return bool(spec["always_allow"])
    return False


def tool_policy(config_or_settings: dict[str, Any] | None, tool_name: str) -> ToolPolicy:
    """Return the shared policy projection for one configurable local tool."""
    return ToolPolicy(
        enabled=tool_enabled(config_or_settings, tool_name),
        always_allow=tool_always_allow(config_or_settings, tool_name),
        plan_access=tool_plan_access(config_or_settings, tool_name),
        ptc=tool_ptc(config_or_settings, tool_name),
        rubric=tool_rubric_access(config_or_settings, tool_name),
    )


def tool_plan_access(config_or_settings: dict[str, Any] | None, tool_name: str) -> bool:
    if tool_name in READ_ONLY_BUILTIN_TOOLS:
        return tool_enabled(config_or_settings, tool_name)
    return _tool_access_value(config_or_settings, tool_name, "plan_access")


def tool_ptc(config_or_settings: dict[str, Any] | None, tool_name: str) -> bool:
    """Return whether an applicable local tool may be called from eval."""
    if tool_name in READ_ONLY_BUILTIN_TOOLS:
        return tool_enabled(config_or_settings, tool_name)
    if tool_name in PTC_INAPPLICABLE_TOOLS:
        return False
    return _tool_access_value(config_or_settings, tool_name, "ptc")


def tool_rubric_access(config_or_settings: dict[str, Any] | None, tool_name: str) -> bool:
    """Return whether an applicable local tool is available to the rubric grader."""
    if tool_name in READ_ONLY_BUILTIN_TOOLS:
        return tool_enabled(config_or_settings, tool_name)
    if tool_name in INBUILT_DANGEROUS_TOOLS and tool_name != EXECUTE_TOOL:
        return False
    return _tool_access_value(config_or_settings, tool_name, "rubric")


def _tool_access_value(config_or_settings: dict[str, Any] | None, tool_name: str, key: str) -> bool:
    tools = hitl_settings(config_or_settings).get("tools", {})
    spec = tools.get(tool_name) if isinstance(tools, dict) else None
    return bool(spec.get(key, False)) if isinstance(spec, dict) else False


def tool_enabled(config_or_settings: dict[str, Any] | None, tool_name: str) -> bool:
    """Return whether a configurable user tool is enabled."""
    hitl = hitl_settings(config_or_settings)
    tools = hitl.get("tools", {})
    spec = tools.get(tool_name) if isinstance(tools, dict) else None
    if isinstance(spec, dict) and isinstance(spec.get("enabled"), bool):
        return bool(spec["enabled"])
    return True


def set_git_protection(settings: dict[str, Any], enabled: bool) -> dict[str, Any]:
    """Return settings with the Git protection toggle updated."""
    updated = normalize_settings(settings)
    updated["hitl"]["git_protection"]["enabled"] = bool(enabled)
    return updated


def set_tool_always_allow(settings: dict[str, Any], tool_name: str, always_allow: bool) -> dict[str, Any]:
    """Return settings with one tool approval toggle updated."""
    if tool_name in READ_ONLY_BUILTIN_TOOLS:
        return normalize_settings(settings)
    updated = normalize_settings(settings)
    current = dict(updated["hitl"].setdefault("tools", {}).get(tool_name, {"enabled": True}))
    current["always_allow"] = bool(always_allow)
    updated["hitl"]["tools"][tool_name] = current
    return updated


def set_tool_enabled(settings: dict[str, Any], tool_name: str, enabled: bool) -> dict[str, Any]:
    """Return settings with one configurable tool enabled or disabled."""
    updated = normalize_settings(settings)
    current = dict(updated["hitl"].setdefault("tools", {}).get(tool_name, {"always_allow": False}))
    current["enabled"] = bool(enabled)
    current.setdefault("always_allow", False)
    updated["hitl"]["tools"][tool_name] = current
    if tool_name in {"eval", "task"} and not enabled:
        updated["system"][DYNAMIC_SUBAGENTS]["enabled"] = False
    return updated


def set_tool_plan_access(settings: dict[str, Any], tool_name: str, plan_access: bool) -> dict[str, Any]:
    """Return settings with explicit Custom Tool Plan trust."""
    if tool_name in READ_ONLY_BUILTIN_TOOLS:
        return normalize_settings(settings)
    return _set_tool_access_value(settings, tool_name, "plan_access", plan_access)


def set_tool_ptc(settings: dict[str, Any], tool_name: str, ptc: bool) -> dict[str, Any]:
    """Return settings with explicit local-tool PTC access."""
    if tool_name in READ_ONLY_BUILTIN_TOOLS:
        return normalize_settings(settings)
    if tool_name in PTC_INAPPLICABLE_TOOLS:
        return normalize_settings(settings)
    return _set_tool_access_value(settings, tool_name, "ptc", ptc)


def set_tool_rubric_access(settings: dict[str, Any], tool_name: str, value: bool) -> dict[str, Any]:
    """Return settings with explicit local-tool Rubric access."""
    if tool_name in READ_ONLY_BUILTIN_TOOLS:
        return normalize_settings(settings)
    if tool_name in INBUILT_DANGEROUS_TOOLS and tool_name != EXECUTE_TOOL:
        return normalize_settings(settings)
    return _set_tool_access_value(settings, tool_name, "rubric", value)


def _set_tool_access_value(settings: dict[str, Any], tool_name: str, key: str, value: bool) -> dict[str, Any]:
    updated = normalize_settings(settings)
    current = dict(updated["hitl"].setdefault("tools", {}).get(tool_name, {}))
    current.setdefault("enabled", True)
    current.setdefault("always_allow", False)
    current[key] = bool(value)
    updated["hitl"]["tools"][tool_name] = current
    return updated


def mcp_server_settings(config_or_settings: dict[str, Any] | None, server: str) -> dict[str, Any]:
    settings = _settings_object(config_or_settings)
    mcp = settings.get("mcp", {}) if isinstance(settings, dict) else {}
    servers = mcp.get("servers", {}) if isinstance(mcp, dict) else {}
    spec = servers.get(server, {}) if isinstance(servers, dict) else {}
    return dict(spec) if isinstance(spec, dict) else {}


def mcp_server_enabled(config_or_settings: dict[str, Any] | None, server: str) -> bool:
    return bool(mcp_server_settings(config_or_settings, server).get("enabled", True))


def mcp_server_always_allow(config_or_settings: dict[str, Any] | None, server: str) -> bool:
    return bool(mcp_server_settings(config_or_settings, server).get("always_allow", False))


def mcp_server_approved_fingerprint(config_or_settings: dict[str, Any] | None, server: str) -> str:
    value = mcp_server_settings(config_or_settings, server).get("approved_fingerprint", "")
    return value if isinstance(value, str) and _valid_fingerprint(value) else ""


def mcp_tool_policy(
    config_or_settings: dict[str, Any] | None,
    server: str,
    tool_name: str,
) -> ToolPolicy:
    spec = mcp_server_settings(config_or_settings, server).get("tools", {})
    value = spec.get(tool_name, {}) if isinstance(spec, dict) else {}
    value = value if isinstance(value, dict) else {}
    return ToolPolicy(
        enabled=bool(value.get("enabled", True)),
        always_allow=bool(value.get("always_allow", False)),
        plan_access=bool(value.get("plan_access", False)),
        ptc=bool(value.get("ptc", False)),
        rubric=bool(value.get("rubric", False)),
    )


def set_mcp_server_enabled(settings: dict[str, Any], server: str, enabled: bool) -> dict[str, Any]:
    return _set_mcp_server_value(settings, server, "enabled", bool(enabled))


def set_mcp_server_always_allow(settings: dict[str, Any], server: str, value: bool) -> dict[str, Any]:
    return _set_mcp_server_value(settings, server, "always_allow", bool(value))


def set_mcp_server_approved_fingerprint(settings: dict[str, Any], server: str, value: str) -> dict[str, Any]:
    return _set_mcp_server_value(settings, server, "approved_fingerprint", value if _valid_fingerprint(value) else "")


def set_mcp_tool_policy_value(
    settings: dict[str, Any],
    server: str,
    tool_name: str,
    key: str,
    value: bool,
) -> dict[str, Any]:
    if key not in {"enabled", "always_allow", "plan_access", "ptc", "rubric"}:
        return normalize_settings(settings)
    updated = normalize_settings(settings)
    servers = updated.setdefault("mcp", {}).setdefault("servers", {})
    server_spec = dict(servers.get(server, {}))
    tools = dict(server_spec.get("tools", {}))
    tool_spec = dict(tools.get(tool_name, {}))
    tool_spec[key] = bool(value)
    tools[tool_name] = tool_spec
    server_spec["tools"] = tools
    servers[server] = server_spec
    return updated


def _set_mcp_server_value(settings: dict[str, Any], server: str, key: str, value: Any) -> dict[str, Any]:
    updated = normalize_settings(settings)
    servers = updated.setdefault("mcp", {}).setdefault("servers", {})
    spec = dict(servers.get(server, {}))
    if value == "":
        spec.pop(key, None)
    else:
        spec[key] = value
    servers[server] = spec
    return updated


def _settings_object(config_or_settings: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(config_or_settings, dict):
        return {}
    value = config_or_settings.get("settings", config_or_settings)
    return value if isinstance(value, dict) else {}


def _valid_fingerprint(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def set_execute_env_mode(settings: dict[str, Any], mode: str) -> dict[str, Any]:
    """Return settings with the execute environment mode updated."""
    updated = normalize_settings(settings)
    current = execute_env_settings(updated)
    if mode in EXECUTE_ENV_MODES:
        current["mode"] = mode
    updated["hitl"]["execute_env"] = current
    return updated


def set_execute_env_value(settings: dict[str, Any], key: str, value: str) -> dict[str, Any]:
    """Return settings with one execute environment selector value updated."""
    updated = normalize_settings(settings)
    current = execute_env_settings(updated)
    if key in {"name", "prefix", "path"}:
        current[key] = str(value or "").strip()
    updated["hitl"]["execute_env"] = current
    return updated


def set_execute_env_allow(settings: dict[str, Any], value: str | list[Any]) -> dict[str, Any]:
    """Return settings with additional execute environment variable names updated."""
    updated = normalize_settings(settings)
    current = execute_env_settings(updated)
    values = value.split(",") if isinstance(value, str) else value
    current["allow"] = normalize_env_names(list(values))
    updated["hitl"]["execute_env"] = current
    return updated
