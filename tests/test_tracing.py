"""Focused checks for registry-backed LangSmith/OpenInference OTLP tracing."""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import inspect
import os
import subprocess
import sys
import tempfile
import unittest
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from config import settings
from config.tracing import (
    TRACING_REGISTRY_TEMPLATE,
    TracingProfile,
    TracingRegistry,
    bootstrap_tracing_registry,
    load_tracing_registry,
    tracing_path,
)
from tracing import bootstrap


_CURRENT_TURN_TRACE: ContextVar[str | None] = ContextVar(
    "test_current_turn_trace",
    default=None,
)


class _Resource:
    instances: list["_Resource"] = []

    def __init__(self, attributes: dict[str, str]) -> None:
        self.attributes = attributes
        self.instances.append(self)

    @classmethod
    def create(cls, attributes: dict[str, str]) -> "_Resource":
        return cls(attributes)


class _ResourceAttributes:
    PROJECT_NAME = "openinference.project.name"


class _Provider:
    instances: list["_Provider"] = []

    def __init__(self, *, resource: _Resource) -> None:
        self.resource = resource
        self.processors: list[object] = []
        self.force_flush_calls: list[int] = []
        self.shutdown_called = False
        self.instances.append(self)

    def add_span_processor(self, processor: object) -> None:
        self.processors.append(processor)

    def force_flush(self, timeout_millis: int) -> bool:
        self.force_flush_calls.append(timeout_millis)
        return True

    def shutdown(self) -> None:
        self.shutdown_called = True


class _Exporter:
    instances: list["_Exporter"] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.instances.append(self)


class _FailingExporter:
    def __init__(self, **kwargs: object) -> None:
        raise RuntimeError("exporter failed")


class _Processor:
    def __init__(self, exporter: object) -> None:
        self.exporter = exporter


class _SemanticProcessor:
    instances: list["_SemanticProcessor"] = []

    def __init__(self, span_attributes: dict[str, object]) -> None:
        self.span_attributes = span_attributes
        self.instances.append(self)


class _Client:
    instances: list["_Client"] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.close_calls: list[float | None] = []
        self.instances.append(self)

    def close(self, timeout: float | None = None) -> None:
        self.close_calls.append(timeout)


class _Langsmith:
    Client = _Client
    configure_calls: list[dict[str, object]] = []
    fail_configure = False

    @classmethod
    def configure(cls, **kwargs: object) -> None:
        cls.configure_calls.append(kwargs)
        if cls.fail_configure:
            raise RuntimeError("configuration failed")


class _TurnTrace:
    def __init__(self, owner: "_TurnLangsmith", trace_id: str, inputs: dict[str, object]) -> None:
        self.owner = owner
        self.trace_id = trace_id
        self.inputs = inputs
        self.set_calls: list[dict[str, object]] = []
        self.end_calls = 0
        self.token = None

    async def __aenter__(self) -> "_TurnTrace":
        self.token = _CURRENT_TURN_TRACE.set(self.trace_id)
        self.owner.entered.append(self.trace_id)
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        self.owner.exited.append(self.trace_id)
        assert self.token is not None
        _CURRENT_TURN_TRACE.reset(self.token)

    def set(self, **kwargs: object) -> None:
        self.set_calls.append(kwargs)

    def end(self) -> None:
        self.end_calls += 1


class _TurnLangsmith:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, object]] = []
        self.traces: list[_TurnTrace] = []
        self.entered: list[str] = []
        self.exited: list[str] = []

    def trace(
        self,
        name: str,
        *,
        run_type: str,
        inputs: dict[str, object],
        client: object,
    ) -> _TurnTrace:
        trace_id = f"turn-{len(self.calls) + 1}"
        self.calls.append((name, run_type, client))
        trace = _TurnTrace(self, trace_id, inputs)
        self.traces.append(trace)
        return trace


def _runtime() -> tuple[object, ...]:
    return (
        _Langsmith,
        _Resource,
        _ResourceAttributes,
        _Provider,
        _SemanticProcessor,
        _Processor,
        _Exporter,
    )


def _registry(
    *,
    name: str = "phoenix",
    endpoint: str = "http://127.0.0.1:6006/v1/traces",
    headers: dict[str, str] | None = None,
    span_attributes: dict[str, object] | None = None,
) -> TracingRegistry:
    profile = TracingProfile(
        name,
        endpoint,
        dict(headers or {}),
        dict(span_attributes or {}),
    )
    return TracingRegistry({name: profile})


def _tracing_settings(profile: str = "phoenix") -> dict[str, object]:
    return settings.set_tracing_profile(
        settings.set_tracing_enabled(settings.DEFAULT_SETTINGS, True),
        profile,
    )


def _real_tracing_available() -> bool:
    try:
        return all(
            importlib.util.find_spec(name) is not None
            for name in (
                "langsmith",
                "openinference.instrumentation",
                "opentelemetry.sdk.trace",
            )
        )
    except ModuleNotFoundError:
        return False


class TracingTests(unittest.TestCase):
    def setUp(self) -> None:
        bootstrap._active_langsmith = None
        bootstrap._active_client = None
        bootstrap._active_provider = None
        bootstrap._active_environment = None
        bootstrap._saved_environment = None
        _Resource.instances.clear()
        _Provider.instances.clear()
        _Exporter.instances.clear()
        _SemanticProcessor.instances.clear()
        _Client.instances.clear()
        _Langsmith.configure_calls.clear()
        _Langsmith.fail_configure = False

    def tearDown(self) -> None:
        bootstrap.shutdown_tracing()
        bootstrap._active_langsmith = None
        bootstrap._active_client = None
        bootstrap._active_provider = None
        bootstrap._active_environment = None
        bootstrap._saved_environment = None

    def test_registry_bootstraps_exact_profiles_when_missing(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            path = bootstrap_tracing_registry(workspace)

            self.assertEqual(path, tracing_path(workspace))
            bootstrapped = path.read_text(encoding="utf-8")
            self.assertEqual(bootstrapped, TRACING_REGISTRY_TEMPLATE)
            for field in ("endpoint", "headers", "span_attributes"):
                self.assertIn(f"#   {field}:", bootstrapped)
            self.assertEqual(
                list(load_tracing_registry(workspace).profiles),
                ["phoenix", "mlflow", "langsmith"],
            )
            self.assertEqual(
                yaml.safe_load(path.read_text(encoding="utf-8")),
                {
                    "profiles": {
                        "phoenix": {
                            "endpoint": "http://127.0.0.1:6006/v1/traces",
                            "headers": {},
                        },
                        "mlflow": {
                            "endpoint": "http://127.0.0.1:5000/v1/traces",
                            "headers": {"x-mlflow-experiment-id": "0"},
                            "span_attributes": {"mlflow.message.format": "langchain"},
                        },
                        "langsmith": {
                            "endpoint": "https://api.smith.langchain.com/otel/v1/traces",
                            "headers": {
                                "x-api-key": "${LANGSMITH_API_KEY}",
                                "X-Tenant-Id": "${LANGSMITH_WORKSPACE_ID}",
                                "Langsmith-Project": "MIRA",
                            },
                        },
                    }
                },
            )

    def test_existing_registry_is_never_overwritten_or_injected(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            path = tracing_path(workspace)
            path.parent.mkdir()
            custom = (
                "profiles:\n"
                "  corporate:\n"
                "    endpoint: https://otel.example/v1/traces\n"
                "    headers: {}\n"
            )
            path.write_text(custom, encoding="utf-8")

            bootstrap_tracing_registry(workspace)

            self.assertEqual(path.read_text(encoding="utf-8"), custom)
            self.assertEqual(list(load_tracing_registry(workspace).profiles), ["corporate"])

    def test_user_profile_is_discovered_without_environment_resolution(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            path = tracing_path(workspace)
            path.parent.mkdir()
            path.write_text(
                "profiles:\n"
                "  corporate:\n"
                "    endpoint: https://otel.example/v1/traces\n"
                "    headers:\n"
                '      Authorization: "Bearer ${CORP_OTEL_TOKEN}"\n',
                encoding="utf-8",
            )

            registry = load_tracing_registry(workspace)

            self.assertEqual(list(registry.profiles), ["corporate"])
            self.assertEqual(
                registry.profiles["corporate"].headers["Authorization"],
                "Bearer ${CORP_OTEL_TOKEN}",
            )
            self.assertEqual(registry.profiles["corporate"].span_attributes, {})

    def test_span_attributes_parse_with_otel_compatible_values(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            path = tracing_path(workspace)
            path.parent.mkdir()
            path.write_text(
                "profiles:\n"
                "  corporate:\n"
                "    endpoint: https://otel.example/v1/traces\n"
                "    headers: {}\n"
                "    span_attributes:\n"
                "      deployment.environment: test\n"
                "      mira.retry_count: 2\n"
                "      mira.sampled: true\n"
                "      mira.ratios: [0.25, 0.5]\n",
                encoding="utf-8",
            )

            profile = load_tracing_registry(workspace).profiles["corporate"]

            self.assertEqual(
                profile.span_attributes,
                {
                    "deployment.environment": "test",
                    "mira.retry_count": 2,
                    "mira.sampled": True,
                    "mira.ratios": [0.25, 0.5],
                },
            )

    def test_invalid_headers_are_a_friendly_nonfatal_issue(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            path = tracing_path(workspace)
            path.parent.mkdir()
            path.write_text(
                "profiles:\n"
                "  broken:\n"
                "    endpoint: https://otel.example/v1/traces\n"
                "    headers: []\n",
                encoding="utf-8",
            )

            registry = load_tracing_registry(workspace)

            self.assertEqual(registry.invalid_names, ("broken",))
            self.assertFalse(registry.profiles)
            self.assertIn("Invalid tracing profile: broken", registry.issues[0].summary)
            self.assertIn("headers must be a mapping", registry.issues[0].details)

    def test_disabled_tracing_never_loads_optional_runtime(self) -> None:
        with patch.object(bootstrap, "_load_runtime") as load_runtime:
            issues = bootstrap.configure_tracing(settings.DEFAULT_SETTINGS, environ={})

        self.assertEqual(issues, [])
        load_runtime.assert_not_called()

    def test_safe_revision_lookup_detaches_git_from_stdin(self) -> None:
        with patch.object(
            bootstrap.subprocess,
            "check_output",
            return_value="v1.2.3-4-gabcdef\n",
        ) as check_output:
            revision = bootstrap._safe_revision_id()

        self.assertEqual(revision, "v1.2.3-4-gabcdef")
        check_output.assert_called_once_with(
            ["git", "describe", "--tags", "--always", "--dirty"],
            stdin=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            encoding="utf-8",
        )

    def test_missing_selected_profile_is_a_friendly_nonfatal_issue(self) -> None:
        issues = bootstrap.configure_tracing(
            _tracing_settings("corporate"),
            _registry(),
            environ={},
        )

        self.assertEqual(len(issues), 1)
        self.assertEqual(
            issues[0].summary,
            "Tracing profile 'corporate' was not found in .mira/tracing.yml.",
        )
        self.assertIn("add it", issues[0].details)

    def test_missing_extra_is_a_friendly_nonfatal_issue(self) -> None:
        with patch.object(
            bootstrap,
            "_load_runtime",
            side_effect=ModuleNotFoundError("openinference.instrumentation"),
        ):
            issues = bootstrap.configure_tracing(_tracing_settings(), _registry(), environ={})

        self.assertEqual(len(issues), 1)
        self.assertIn("Tracing dependencies", issues[0].summary)
        self.assertIn('pip install "mira[tracing]"', issues[0].guidance)
        self.assertNotIn("langsmith[otel]", issues[0].guidance)

    def test_missing_environment_reference_does_not_expose_a_value(self) -> None:
        registry = _registry(
            name="corporate",
            endpoint="https://example.com/v1/traces",
            headers={"Authorization": "Bearer ${TRACE_TOKEN}"},
        )

        issues = bootstrap.configure_tracing(
            _tracing_settings("corporate"),
            registry,
            environ={},
        )

        self.assertEqual(len(issues), 1)
        self.assertIn("TRACE_TOKEN", issues[0].details)
        self.assertIn("process environment", issues[0].details)
        self.assertNotIn("resolved-secret", issues[0].details)

    def test_settings_and_preview_keep_environment_references_unresolved(self) -> None:
        configured = _tracing_settings("corporate")
        profile = _registry(
            name="corporate",
            endpoint="https://example.com/otel/v1/traces",
            headers={"Authorization": "Bearer ${TRACE_TOKEN}"},
        ).profiles["corporate"]
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            self.assertTrue(settings.save_settings(workspace, configured))
            persisted = settings.settings_path(workspace).read_text(encoding="utf-8")

        preview = bootstrap.tracing_yaml_fragment(
            enabled=True,
            profile="corporate",
            middleware_spans="hidden",
            endpoint=profile.endpoint,
            headers=profile.headers,
            span_attributes=profile.span_attributes,
        )
        self.assertIn("profile: corporate", persisted)
        self.assertIn("middleware_spans: hidden", persisted)
        self.assertNotIn("endpoint:", persisted)
        self.assertNotIn("headers:", persisted)
        self.assertIn("${TRACE_TOKEN}", preview)
        self.assertIn("span_attributes: {}", preview)
        self.assertIn("middleware_spans: hidden", preview)
        self.assertNotIn("resolved-secret", preview)

    def test_client_processors_exporter_and_resource_receive_resolved_profile(self) -> None:
        registry = TracingRegistry(
            {
                "corporate": TracingProfile(
                    "corporate",
                    "https://example.com/otel/v1/traces",
                    {
                        "Authorization": "Bearer ${TRACE_TOKEN}",
                        "tenant": "my team",
                    },
                    {
                        "deployment.environment": "${DEPLOYMENT_ENV}",
                        "mira.retry_count": 2,
                    },
                ),
                "unused": TracingProfile(
                    "unused",
                    "https://unused.example/v1/traces",
                    {"Authorization": "Bearer ${UNRESOLVED_UNUSED_TOKEN}"},
                ),
            }
        )
        environ = {"TRACE_TOKEN": "resolved-secret", "DEPLOYMENT_ENV": "staging"}

        with patch.object(bootstrap, "_load_runtime", return_value=_runtime()):
            issues = bootstrap.configure_tracing(
                _tracing_settings("corporate"),
                registry,
                environ=environ,
            )

        self.assertEqual(issues, [])
        self.assertEqual(
            _Exporter.instances[0].kwargs,
            {
                "endpoint": "https://example.com/otel/v1/traces",
                "headers": {
                    "Authorization": "Bearer resolved-secret",
                    "tenant": "my team",
                },
            },
        )
        provider = _Provider.instances[0]
        client = _Client.instances[0]
        self.assertEqual(
            client.kwargs,
            {"tracing_mode": "otel", "otel_tracer_provider": provider},
        )
        self.assertEqual(_Langsmith.configure_calls, [{"client": client, "enabled": True}])
        self.assertIsInstance(provider.processors[0], _SemanticProcessor)
        self.assertEqual(
            provider.processors[0].span_attributes,
            {"deployment.environment": "staging", "mira.retry_count": 2},
        )
        self.assertIsInstance(provider.processors[1], _Processor)
        self.assertIs(provider.processors[1].exporter, _Exporter.instances[0])
        self.assertEqual(
            provider.resource.attributes,
            {
                "service.name": "MIRA",
                "openinference.project.name": "MIRA",
            },
        )
        self.assertEqual(environ["LANGSMITH_TRACING"], "true")
        self.assertEqual(environ["LANGSMITH_TRACING_MODE"], "otel")

    def test_reconfiguration_tears_down_old_runtime_and_initializes_new(self) -> None:
        with patch.object(bootstrap, "_load_runtime", return_value=_runtime()):
            self.assertEqual(
                bootstrap.configure_tracing(_tracing_settings(), _registry(), environ={}),
                [],
            )
            first_client = _Client.instances[0]
            first_provider = _Provider.instances[0]
            corporate = _registry(
                name="corporate",
                endpoint="https://changed.example/v1/traces",
            )
            self.assertEqual(
                bootstrap.configure_tracing(
                    _tracing_settings("corporate"),
                    corporate,
                    environ={},
                ),
                [],
            )

        self.assertEqual(first_client.close_calls, [1.0])
        self.assertEqual(first_provider.force_flush_calls, [bootstrap._FLUSH_TIMEOUT_MILLIS])
        self.assertTrue(first_provider.shutdown_called)
        self.assertEqual(len(_Client.instances), 2)
        self.assertEqual(len(_SemanticProcessor.instances), 2)
        self.assertTrue(all(len(provider.processors) == 2 for provider in _Provider.instances))
        self.assertEqual(
            _Exporter.instances[1].kwargs["endpoint"],
            "https://changed.example/v1/traces",
        )

    def test_disable_detaches_active_runtime_without_loading_optional_packages(self) -> None:
        with patch.object(bootstrap, "_load_runtime", return_value=_runtime()):
            self.assertEqual(
                bootstrap.configure_tracing(_tracing_settings(), _registry(), environ={}),
                [],
            )
        client = _Client.instances[0]
        provider = _Provider.instances[0]

        with patch.object(bootstrap, "_load_runtime") as load_runtime:
            self.assertEqual(
                bootstrap.configure_tracing(settings.DEFAULT_SETTINGS, environ={}),
                [],
            )
            bootstrap.shutdown_tracing()

        load_runtime.assert_not_called()
        self.assertEqual(client.close_calls, [1.0])
        self.assertEqual(provider.force_flush_calls, [bootstrap._FLUSH_TIMEOUT_MILLIS])
        self.assertTrue(provider.shutdown_called)

    def test_initialization_failure_cleans_partial_runtime_and_returns_issue(self) -> None:
        _Langsmith.fail_configure = True
        with patch.object(bootstrap, "_load_runtime", return_value=_runtime()):
            issues = bootstrap.configure_tracing(_tracing_settings(), _registry(), environ={})

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].summary, "Tracing could not be initialized")
        self.assertIn("RuntimeError", issues[0].details)
        self.assertIsNone(bootstrap._active_client)
        self.assertIsNone(bootstrap._active_provider)
        self.assertEqual(_Client.instances[0].close_calls, [1.0])
        self.assertEqual(
            _Provider.instances[0].force_flush_calls,
            [bootstrap._FLUSH_TIMEOUT_MILLIS],
        )
        self.assertTrue(_Provider.instances[0].shutdown_called)

    def test_exporter_failure_shuts_down_provider_without_attaching_client(self) -> None:
        runtime = (*_runtime()[:-1], _FailingExporter)
        with patch.object(bootstrap, "_load_runtime", return_value=runtime):
            issues = bootstrap.configure_tracing(_tracing_settings(), _registry(), environ={})

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].summary, "Tracing could not be initialized")
        self.assertIn("MIRA remains usable", issues[0].guidance)
        self.assertEqual(_Client.instances, [])
        self.assertEqual(
            _Provider.instances[0].force_flush_calls,
            [bootstrap._FLUSH_TIMEOUT_MILLIS],
        )
        self.assertTrue(_Provider.instances[0].shutdown_called)

    def test_user_langsmith_environment_is_restored_after_disable(self) -> None:
        environ = {
            "LANGSMITH_TRACING": "false",
            "LANGSMITH_TRACING_MODE": "langsmith",
            "LANGCHAIN_REVISION_ID": "ci-revision",
            "LANGCHAIN_TRACING_V2": "1",
            "LANGSMITH_API_KEY": "secret-value",
        }
        original = dict(environ)

        with patch.object(bootstrap, "_load_runtime", return_value=_runtime()):
            issues = bootstrap.configure_tracing(
                _tracing_settings(),
                _registry(),
                environ=environ,
            )

        self.assertEqual(issues, [])
        self.assertEqual(environ["LANGSMITH_TRACING"], "true")
        self.assertEqual(environ["LANGSMITH_TRACING_MODE"], "otel")
        self.assertEqual(environ["LANGCHAIN_REVISION_ID"], "ci-revision")
        self.assertEqual(environ["LANGCHAIN_TRACING_V2"], "1")
        self.assertIsNotNone(bootstrap._active_provider)

        first_client = _Client.instances[0]
        with patch.object(bootstrap, "_load_runtime", return_value=_runtime()):
            self.assertEqual(
                bootstrap.configure_tracing(
                    _tracing_settings(),
                    _registry(),
                    environ=environ,
                ),
                [],
            )
        self.assertEqual(first_client.close_calls, [1.0])
        self.assertEqual(environ["LANGSMITH_TRACING"], "true")
        self.assertEqual(environ["LANGSMITH_TRACING_MODE"], "otel")

        bootstrap.shutdown_tracing()

        self.assertEqual(environ, original)

    def test_failed_revision_lookup_uses_nonempty_fallback_and_reload_restores_it(self) -> None:
        environ: dict[str, str] = {}
        failure = subprocess.CalledProcessError(128, ["git", "describe"])

        with (
            patch.object(bootstrap, "_load_runtime", return_value=_runtime()),
            patch.object(
                bootstrap.subprocess,
                "check_output",
                side_effect=[failure, "next-revision\n"],
            ),
        ):
            self.assertEqual(
                bootstrap.configure_tracing(
                    _tracing_settings(),
                    _registry(),
                    environ=environ,
                ),
                [],
            )
            self.assertEqual(environ["LANGCHAIN_REVISION_ID"], "unknown")

            self.assertEqual(
                bootstrap.configure_tracing(
                    _tracing_settings(),
                    _registry(),
                    environ=environ,
                ),
                [],
            )
            self.assertEqual(environ["LANGCHAIN_REVISION_ID"], "next-revision")

        bootstrap.shutdown_tracing()
        self.assertNotIn("LANGCHAIN_REVISION_ID", environ)

    def test_generic_profile_runtime_has_no_vendor_branching(self) -> None:
        corporate = _registry(
            name="corporate",
            endpoint="https://observability.example.com/v1/traces",
        )
        with patch.object(bootstrap, "_load_runtime", return_value=_runtime()):
            self.assertEqual(
                bootstrap.configure_tracing(
                    _tracing_settings("corporate"),
                    corporate,
                    environ={},
                ),
                [],
            )

        runtime_tree = ast.parse(inspect.getsource(bootstrap.configure_tracing))
        branch_conditions = "\n".join(
            ast.unparse(node.test).lower()
            for node in ast.walk(runtime_tree)
            if isinstance(node, ast.If)
        )
        self.assertNotIn("phoenix", branch_conditions)
        self.assertNotIn("mlflow", branch_conditions)
        self.assertNotIn("langsmith", branch_conditions)


class ModelCompatibilityTests(unittest.TestCase):
    def _import_value(self, explicit: str | None) -> str:
        environment = dict(os.environ)
        if explicit is None:
            environment.pop("DEFER_PYDANTIC_BUILD", None)
        else:
            environment["DEFER_PYDANTIC_BUILD"] = explicit
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import os; import agent.llm; print(os.environ['DEFER_PYDANTIC_BUILD'])",
            ],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout.strip()

    def test_deferred_pydantic_build_defaults_before_anyllm_import(self) -> None:
        source = Path("agent/__init__.py").read_text(encoding="utf-8")
        self.assertIn('os.environ.setdefault("DEFER_PYDANTIC_BUILD", "false")', source)
        self.assertEqual(self._import_value(None), "false")

    def test_explicit_deferred_pydantic_build_value_is_preserved(self) -> None:
        self.assertEqual(self._import_value("true"), "true")

    def test_compatibility_fix_does_not_subclass_or_patch_chatanyllm(self) -> None:
        tree = ast.parse(Path("agent/llm.py").read_text(encoding="utf-8"))
        self.assertFalse(
            any(
                isinstance(node, ast.ClassDef)
                and any(ast.unparse(base) == "ChatAnyLLM" for base in node.bases)
                for node in ast.walk(tree)
            )
        )
        self.assertNotIn("MockValSer", Path("agent/llm.py").read_text(encoding="utf-8"))


class TurnTracingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        bootstrap._active_langsmith = None
        bootstrap._active_client = None
        bootstrap._active_provider = None
        bootstrap._active_environment = None

    def tearDown(self) -> None:
        bootstrap._active_langsmith = None
        bootstrap._active_client = None
        bootstrap._active_provider = None
        bootstrap._active_environment = None

    async def _run_turn(
        self,
        child_parents: list[str | None],
        *,
        text: str = "hello",
        display_text: str | None = None,
    ) -> None:
        from core.execution.runner import TurnResult
        from tests.support.turns import run_user_turn

        async def fake_run_turn(**kwargs: object) -> TurnResult:
            child_parents.append(_CURRENT_TURN_TRACE.get())
            await asyncio.sleep(0)
            child_parents.append(_CURRENT_TURN_TRACE.get())
            return TurnResult(
                final_text="done",
                input_tokens=8,
                output_tokens=3,
                total_tokens=11,
            )

        class Store:
            def save(self, session: dict[str, object]) -> None:
                pass

        session: dict[str, object] = {
            "id": "thread-1",
            "events": [],
            "turns": 0,
            "dashboard": {},
        }
        with patch("tests.support.turns.run_turn", side_effect=fake_run_turn):
            await run_user_turn(
                agent=object(),
                plan_agent=object(),
                renderer=SimpleNamespace(),
                store=Store(),
                session=session,
                mode={"planning": False},
                text=text,
                display_text=display_text,
            )

    async def test_run_user_turn_is_a_noop_when_tracing_is_inactive(self) -> None:
        child_parents: list[str | None] = []

        await self._run_turn(child_parents)

        self.assertEqual(child_parents, [None, None])

    async def test_one_parent_contains_all_work_and_consecutive_turns_are_separate(self) -> None:
        langsmith = _TurnLangsmith()
        client = object()
        bootstrap._active_langsmith = langsmith
        bootstrap._active_client = client
        first_children: list[str | None] = []
        second_children: list[str | None] = []

        await self._run_turn(first_children)
        await self._run_turn(second_children)

        self.assertEqual(
            langsmith.calls,
            [
                ("MIRA Turn", "chain", client),
                ("MIRA Turn", "chain", client),
            ],
        )
        self.assertEqual(langsmith.entered, ["turn-1", "turn-2"])
        self.assertEqual(langsmith.exited, ["turn-1", "turn-2"])
        self.assertEqual(first_children, ["turn-1", "turn-1"])
        self.assertEqual(second_children, ["turn-2", "turn-2"])
        self.assertEqual([trace.inputs for trace in langsmith.traces], [{"input": "hello"}] * 2)

    async def test_root_records_visible_input_final_output_and_aggregate_usage(self) -> None:
        langsmith = _TurnLangsmith()
        bootstrap._active_langsmith = langsmith
        bootstrap._active_client = object()

        await self._run_turn([], text="internal expanded prompt", display_text="visible text")

        trace = langsmith.traces[0]
        self.assertEqual(trace.inputs, {"input": "visible text"})
        self.assertEqual(
            trace.set_calls,
            [
                {
                    "outputs": {"output": "done"},
                    "usage_metadata": {
                        "input_tokens": 8,
                        "output_tokens": 3,
                        "total_tokens": 11,
                    },
                }
            ],
        )
        self.assertEqual(trace.end_calls, 1)

    async def test_multi_phase_continuation_stays_under_one_root(self) -> None:
        from core.execution.runner import TurnResult

        langsmith = _TurnLangsmith()
        bootstrap._active_langsmith = langsmith
        bootstrap._active_client = object()
        child_parents: list[str | None] = []

        @bootstrap.trace_user_turn
        async def immediate_implement(**kwargs: object) -> TurnResult:
            child_parents.append(_CURRENT_TURN_TRACE.get())
            await asyncio.sleep(0)
            child_parents.append(_CURRENT_TURN_TRACE.get())
            return TurnResult(final_text="final Act response", total_tokens=9)

        await immediate_implement(text="internal goal prompt", display_text="make it")

        self.assertEqual(child_parents, ["turn-1", "turn-1"])
        self.assertEqual(len(langsmith.traces), 1)
        self.assertEqual(
            langsmith.traces[0].set_calls[0]["outputs"],
            {"output": "final Act response"},
        )


@unittest.skipUnless(_real_tracing_available(), "requires the mira[tracing] extra")
class SemanticProcessorTests(unittest.TestCase):
    def _finished_spans(
        self,
        definitions: list[tuple[str, dict[str, object]]],
        *,
        span_attributes: dict[str, object] | None = None,
    ):
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
        from tracing.semantic_processor import LangSmithOpenInferenceProcessor

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(LangSmithOpenInferenceProcessor(span_attributes))
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("mira.semantic-tests")
        for name, attributes in definitions:
            with tracer.start_as_current_span(name, attributes=attributes):
                pass
        provider.shutdown()
        return list(exporter.get_finished_spans())

    def test_profile_attributes_reach_every_span_without_discarding_semantics(self) -> None:
        spans = self._finished_spans(
            [
                (
                    "chain",
                    {
                        "langsmith.span.kind": "chain",
                        "existing.attribute": "preserved",
                    },
                ),
                ("plain-otel", {"plain.attribute": 7}),
            ],
            span_attributes={
                "deployment.environment": "test",
                "mira.sampled": True,
                "mira.owners": ["runtime", "observability"],
            },
        )

        by_name = {span.name: span.attributes for span in spans}
        for attributes in by_name.values():
            self.assertEqual(attributes["deployment.environment"], "test")
            self.assertIs(attributes["mira.sampled"], True)
            self.assertEqual(attributes["mira.owners"], ["runtime", "observability"])
        self.assertEqual(by_name["chain"]["existing.attribute"], "preserved")
        self.assertEqual(by_name["chain"]["openinference.span.kind"], "CHAIN")
        self.assertEqual(by_name["plain-otel"]["plain.attribute"], 7)

    def test_span_kinds_input_output_and_tool_semantics(self) -> None:
        import json

        common = {
            "gen_ai.prompt": json.dumps({"value": 1}),
            "gen_ai.completion": json.dumps({"value": 2}),
        }
        spans = self._finished_spans(
            [
                ("chain", {**common, "langsmith.span.kind": "chain"}),
                ("model", {**common, "langsmith.span.kind": "llm"}),
                (
                    "weather",
                    {
                        "langsmith.span.kind": "tool",
                        "langsmith.trace.name": "weather",
                        "gen_ai.prompt": json.dumps({"city": "Paris"}),
                        "gen_ai.completion": json.dumps(
                            {"output": {"temperature": 21, "condition": "sunny"}}
                        ),
                    },
                ),
                ("lookup", {**common, "langsmith.span.kind": "retriever"}),
                ("vectors", {**common, "langsmith.span.kind": "embedding"}),
                ("template", {**common, "langsmith.span.kind": "prompt"}),
                ("parser", {**common, "langsmith.span.kind": "parser"}),
                ("mystery", {**common, "langsmith.span.kind": "unknown"}),
                (
                    "general-purpose",
                    {
                        **common,
                        "langsmith.span.kind": "chain",
                        "langsmith.metadata.ls_agent_type": "subagent",
                        "langsmith.metadata.lc_agent_name": "general-purpose",
                    },
                ),
                (
                    "subagent-model",
                    {
                        **common,
                        "langsmith.span.kind": "llm",
                        "langsmith.metadata.ls_agent_type": "subagent",
                        "langsmith.metadata.lc_agent_name": "general-purpose",
                    },
                ),
                (
                    "subagent-tool",
                    {
                        **common,
                        "langsmith.span.kind": "tool",
                        "langsmith.metadata.ls_agent_type": "subagent",
                    },
                ),
                (
                    "SubAgentMiddleware.awrap_model_call",
                    {**common, "langsmith.span.kind": "chain", "gen_ai.system": "langchain"},
                ),
                (
                    "SkillsMiddleware.before_agent",
                    {**common, "langsmith.span.kind": "chain", "gen_ai.system": "langchain"},
                ),
                ("rubric_grader", {**common, "langsmith.span.kind": "chain"}),
            ]
        )
        by_name = {span.name: span for span in spans}

        self.assertEqual(by_name["chain"].attributes["openinference.span.kind"], "CHAIN")
        self.assertEqual(by_name["model"].attributes["openinference.span.kind"], "LLM")
        self.assertEqual(by_name["weather"].attributes["openinference.span.kind"], "TOOL")
        self.assertEqual(by_name["lookup"].attributes["openinference.span.kind"], "RETRIEVER")
        self.assertEqual(by_name["vectors"].attributes["openinference.span.kind"], "EMBEDDING")
        self.assertEqual(by_name["template"].attributes["openinference.span.kind"], "PROMPT")
        self.assertEqual(by_name["parser"].attributes["openinference.span.kind"], "CHAIN")
        self.assertEqual(by_name["mystery"].attributes["openinference.span.kind"], "CHAIN")
        self.assertEqual(
            by_name["general-purpose"].attributes["openinference.span.kind"],
            "AGENT",
        )
        self.assertEqual(by_name["general-purpose"].attributes["agent.name"], "general-purpose")
        self.assertEqual(by_name["subagent-model"].attributes["openinference.span.kind"], "LLM")
        self.assertNotIn("agent.name", by_name["subagent-model"].attributes)
        self.assertEqual(by_name["subagent-tool"].attributes["openinference.span.kind"], "TOOL")
        self.assertEqual(
            by_name["SubAgentMiddleware.awrap_model_call"].attributes[
                "openinference.span.kind"
            ],
            "CHAIN",
        )
        self.assertEqual(
            by_name["SkillsMiddleware.before_agent"].attributes["openinference.span.kind"],
            "CHAIN",
        )
        self.assertEqual(by_name["rubric_grader"].attributes["openinference.span.kind"], "CHAIN")
        self.assertNotIn("gen_ai.system", by_name["SkillsMiddleware.before_agent"].attributes)
        self.assertFalse(
            any(
                key.startswith("llm.")
                for key in by_name["SubAgentMiddleware.awrap_model_call"].attributes
            )
        )
        self.assertEqual(by_name["weather"].attributes["tool.name"], "weather")
        self.assertEqual(
            json.loads(by_name["weather"].attributes["input.value"]),
            {"city": "Paris"},
        )
        self.assertEqual(
            json.loads(by_name["weather"].attributes["output.value"]),
            {"temperature": 21, "condition": "sunny"},
        )

    def test_messages_model_and_token_counts_use_structured_attributes(self) -> None:
        import json

        def message(message_type: str, **kwargs: object) -> dict[str, object]:
            return {
                "lc": 1,
                "type": "constructor",
                "id": ["langchain", "schema", "messages", f"{message_type.title()}Message"],
                "kwargs": {"type": message_type, **kwargs},
            }

        prompt = {
            "messages": [
                [
                    message("system", content="Follow the rules."),
                    message("human", content="Check Paris."),
                    message(
                        "ai",
                        content="",
                        tool_calls=[
                            {
                                "name": "weather",
                                "args": {"city": "Paris"},
                                "id": "call-weather",
                            }
                        ],
                    ),
                    message(
                        "tool",
                        content='{"temperature":21}',
                        tool_call_id="call-weather",
                    ),
                ]
            ]
        }
        completion = {
            "generations": [
                [
                    {
                        "message": message(
                            "ai",
                            content="It is 21 degrees.",
                            tool_calls=[
                                {
                                    "name": "weather",
                                    "args": {"city": "Paris"},
                                    "id": "call-weather-output",
                                }
                            ],
                        )
                    }
                ]
            ]
        }
        span = self._finished_spans(
            [
                (
                    "ChatModel",
                    {
                        "langsmith.span.kind": "llm",
                        "langsmith.metadata.ls_provider": "anyllm",
                        "langsmith.metadata.provider": "openai",
                        "langsmith.metadata.invocation_params": json.dumps(
                            {
                                "model": "request-model",
                                "temperature": 0.2,
                                "tools": [
                                    {
                                        "type": "function",
                                        "function": {
                                            "name": "weather",
                                            "parameters": {
                                                "type": "object",
                                                "properties": {"city": {"type": "string"}},
                                            },
                                        },
                                    }
                                ],
                            }
                        ),
                        "gen_ai.request.model": "model-1",
                        "gen_ai.response.model": "model-2",
                        "gen_ai.request.max_tokens": 250,
                        "gen_ai.response.finish_reasons": "tool_calls, stop",
                        "gen_ai.usage.input_tokens": 11,
                        "gen_ai.usage.output_tokens": 7,
                        "gen_ai.usage.total_tokens": 18,
                        "gen_ai.usage.input_token_details": "{'cache_read': 4, 'cache_creation': 2}",
                        "gen_ai.usage.output_token_details": "{'reasoning': 3}",
                        "langsmith.span.tags": "seq:step:1, smoke",
                        "gen_ai.prompt": json.dumps(prompt),
                        "gen_ai.completion": json.dumps(completion),
                    },
                )
            ]
        )[0]
        attributes = span.attributes

        self.assertEqual(attributes["llm.input_messages.0.message.role"], "system")
        self.assertEqual(attributes["llm.input_messages.1.message.role"], "user")
        self.assertEqual(
            attributes[
                "llm.input_messages.2.message.tool_calls.0.tool_call.function.name"
            ],
            "weather",
        )
        self.assertEqual(
            json.loads(
                attributes[
                    "llm.input_messages.2.message.tool_calls.0.tool_call.function.arguments"
                ]
            ),
            {"city": "Paris"},
        )
        self.assertEqual(
            attributes["llm.input_messages.3.message.tool_call_id"],
            "call-weather",
        )
        self.assertEqual(
            attributes["llm.input_messages.3.message.content"],
            '{"temperature":21}',
        )
        self.assertEqual(attributes["llm.output_messages.0.message.role"], "assistant")
        self.assertEqual(
            attributes["llm.output_messages.0.message.content"],
            "It is 21 degrees.",
        )
        self.assertEqual(
            attributes[
                "llm.output_messages.0.message.tool_calls.0.tool_call.function.name"
            ],
            "weather",
        )
        self.assertEqual(
            attributes["llm.output_messages.0.message.tool_calls.0.tool_call.id"],
            "call-weather-output",
        )
        self.assertEqual(attributes["llm.model_name"], "model-2")
        self.assertEqual(attributes["llm.provider"], "openai")
        self.assertNotIn("llm.system", attributes)
        self.assertEqual(
            json.loads(attributes["llm.invocation_parameters"]),
            {"temperature": 0.2, "max_tokens": 250},
        )
        self.assertEqual(attributes["llm.finish_reason"], "tool_calls, stop")
        self.assertEqual(attributes["llm.token_count.prompt"], 11)
        self.assertEqual(attributes["llm.token_count.completion"], 7)
        self.assertEqual(attributes["llm.token_count.total"], 18)
        self.assertEqual(attributes["llm.token_count.prompt_details.cache_read"], 4)
        self.assertEqual(attributes["llm.token_count.prompt_details.cache_write"], 2)
        self.assertEqual(attributes["llm.token_count.completion_details.reasoning"], 3)
        self.assertEqual(attributes["tag.tags"], ["seq:step:1", "smoke"])
        tool_schema = json.loads(attributes["llm.tools.0.tool.json_schema"])
        self.assertEqual(tool_schema["function"]["name"], "weather")

    def test_reasoning_contents_preserve_order_fallback_and_tool_calls(self) -> None:
        import json

        def completion(content: object, **kwargs: object) -> str:
            message = {
                "lc": 1,
                "type": "constructor",
                "id": ["langchain", "schema", "messages", "AIMessage"],
                "kwargs": {"type": "ai", "content": content, **kwargs},
            }
            return json.dumps({"generations": [[{"message": message}]]})

        spans = self._finished_spans(
            [
                (
                    "structured",
                    {
                        "langsmith.span.kind": "llm",
                        "gen_ai.completion": completion(
                            [
                                {"type": "reasoning", "reasoning": "think", "index": 0},
                                {"type": "text", "text": "answer", "index": 1},
                                {
                                    "type": "tool_call",
                                    "name": "weather",
                                    "args": {"city": "Paris"},
                                    "id": "call-1",
                                    "index": 2,
                                },
                            ],
                            additional_kwargs={"reasoning_content": "think"},
                            tool_calls=[],
                        ),
                    },
                ),
                (
                    "plain",
                    {
                        "langsmith.span.kind": "llm",
                        "gen_ai.completion": completion("plain answer"),
                    },
                ),
                (
                    "fallback",
                    {
                        "langsmith.span.kind": "llm",
                        "gen_ai.completion": completion(
                            "visible answer",
                            additional_kwargs={"reasoning_content": "fallback thought"},
                        ),
                    },
                ),
            ]
        )
        by_name = {span.name: span.attributes for span in spans}
        structured = by_name["structured"]
        prefix = "llm.output_messages.0.message.contents"
        self.assertEqual(structured[f"{prefix}.0.message_content.type"], "reasoning")
        self.assertEqual(structured[f"{prefix}.0.message_content.text"], "think")
        self.assertEqual(structured[f"{prefix}.1.message_content.type"], "text")
        self.assertEqual(structured[f"{prefix}.1.message_content.text"], "answer")
        self.assertNotIn("llm.output_messages.0.message.content", structured)
        self.assertEqual(
            structured[
                "llm.output_messages.0.message.tool_calls.0.tool_call.function.name"
            ],
            "weather",
        )
        self.assertEqual(by_name["plain"]["llm.output_messages.0.message.content"], "plain answer")
        self.assertFalse(any(key.startswith("llm.tools.") for key in by_name["plain"]))
        fallback = by_name["fallback"]
        self.assertEqual(fallback[f"{prefix}.0.message_content.type"], "reasoning")
        self.assertEqual(fallback[f"{prefix}.0.message_content.text"], "fallback thought")
        self.assertEqual(fallback[f"{prefix}.1.message_content.type"], "text")
        self.assertEqual(fallback[f"{prefix}.1.message_content.text"], "visible answer")

    def test_processor_preserves_topology_and_exports_each_source_once(self) -> None:
        from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
        from tracing.semantic_processor import LangSmithOpenInferenceProcessor

        before: dict[str, tuple[int, int, int | None]] = {}
        before_attributes: dict[str, dict[str, object]] = {}

        class IdentityRecorder(SpanProcessor):
            def on_end(self, span) -> None:  # type: ignore[no-untyped-def]
                before[span.name] = (
                    span.context.trace_id,
                    span.context.span_id,
                    span.parent.span_id if span.parent else None,
                )
                before_attributes[span.name] = dict(span.attributes or {})

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(IdentityRecorder())
        provider.add_span_processor(LangSmithOpenInferenceProcessor())
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("mira.topology-tests")
        with tracer.start_as_current_span(
            "eval",
            attributes={"langsmith.span.kind": "tool", "gen_ai.prompt": "{}"},
        ):
            with tracer.start_as_current_span(
                "task",
                attributes={"langsmith.span.kind": "tool", "gen_ai.prompt": "{}"},
            ):
                with tracer.start_as_current_span(
                    "subagent",
                    attributes={"langsmith.span.kind": "chain", "gen_ai.prompt": "{}"},
                ):
                    pass
        provider.shutdown()

        spans = list(exporter.get_finished_spans())
        self.assertEqual(len(spans), 3)
        self.assertEqual(len({span.context.trace_id for span in spans}), 1)
        after = {
            span.name: (
                span.context.trace_id,
                span.context.span_id,
                span.parent.span_id if span.parent else None,
            )
            for span in spans
        }
        self.assertEqual(after, before)
        for span in spans:
            for key, value in before_attributes[span.name].items():
                self.assertEqual(span.attributes[key], value)
        by_name = {span.name: span for span in spans}
        self.assertEqual(by_name["task"].parent.span_id, by_name["eval"].context.span_id)
        self.assertEqual(by_name["subagent"].parent.span_id, by_name["task"].context.span_id)

    def test_real_turn_parent_preserves_langsmith_tree_and_enrichment(self) -> None:
        import json
        import os
        from collections.abc import Callable, Sequence
        from typing import Any

        import langsmith
        from langchain.agents import create_agent
        from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
        from langchain_core.messages import AIMessage, HumanMessage
        from langchain_core.runnables import Runnable
        from langchain_core.tools import BaseTool, tool
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
        from core.execution.runner import TurnResult
        from tracing.semantic_processor import LangSmithOpenInferenceProcessor

        class ToolCallingModel(FakeMessagesListChatModel):
            def bind_tools(
                self,
                tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
                *,
                tool_choice: str | None = None,
                **kwargs: Any,
            ) -> Runnable[Any, AIMessage]:
                return self

        @tool
        def double(value: int) -> int:
            """Double one integer."""
            return value * 2

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(LangSmithOpenInferenceProcessor())
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        request_patch = patch.object(langsmith.Client, "request_with_retries", autospec=True)
        rest_request = request_patch.start()
        client = None
        try:
            client = langsmith.Client(
                tracing_mode="otel",
                otel_tracer_provider=provider,
            )
            self.assertEqual(client.tracing_mode, "otel")
            with patch.dict(
                os.environ,
                {"LANGSMITH_TRACING": "true", "LANGSMITH_TRACING_MODE": "otel"},
            ):
                langsmith.configure(client=client, enabled=True)
                model = ToolCallingModel(
                    responses=[
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "double",
                                    "args": {"value": 2},
                                    "id": "call-double",
                                    "type": "tool_call",
                                }
                            ],
                            usage_metadata={
                                "input_tokens": 3,
                                "output_tokens": 2,
                                "total_tokens": 5,
                            },
                        ),
                        AIMessage(
                            content="The answer is 4.",
                            usage_metadata={
                                "input_tokens": 6,
                                "output_tokens": 4,
                                "total_tokens": 10,
                            },
                        ),
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "double",
                                    "args": {"value": 3},
                                    "id": "call-double-again",
                                    "type": "tool_call",
                                }
                            ],
                            usage_metadata={
                                "input_tokens": 3,
                                "output_tokens": 2,
                                "total_tokens": 5,
                            },
                        ),
                        AIMessage(
                            content="The answer is 6.",
                            usage_metadata={
                                "input_tokens": 6,
                                "output_tokens": 4,
                                "total_tokens": 10,
                            },
                        ),
                    ]
                )
                agent = create_agent(model, tools=[double], name="langsmith-smoke")

                captured: dict[str, dict[str, Any]] = {}

                @bootstrap.trace_user_turn
                async def invoke_twice(**kwargs: object) -> TurnResult:
                    first = agent.invoke({"messages": [HumanMessage("Double 2.")]})
                    second = agent.invoke({"messages": [HumanMessage("Double 3.")]})
                    captured["first"] = first
                    captured["second"] = second
                    return TurnResult(
                        final_text="The final visible answer is 6.",
                        input_tokens=18,
                        output_tokens=12,
                        total_tokens=30,
                    )

                bootstrap._active_langsmith = langsmith
                bootstrap._active_client = client
                result = asyncio.run(
                    invoke_twice(
                        text="internal expanded text",
                        display_text="Double twice visibly.",
                    )
                )
                self.assertEqual(captured["first"]["messages"][-1].content, "The answer is 4.")
                self.assertEqual(captured["second"]["messages"][-1].content, "The answer is 6.")
                self.assertEqual(result.total_tokens, 30)
        finally:
            bootstrap._active_langsmith = None
            bootstrap._active_client = None
            langsmith.configure(client=None, enabled=None)
            if client is not None:
                client.close(timeout=1.0)
            request_patch.stop()
            provider.force_flush(timeout_millis=1_000)
            provider.shutdown()

        rest_request.assert_not_called()
        spans = list(exporter.get_finished_spans())
        self.assertEqual(sum(span.name == "MIRA Turn" for span in spans), 1)
        self.assertGreaterEqual(len(spans), 9)
        self.assertEqual(len({span.context.trace_id for span in spans}), 1)
        self.assertEqual({span.instrumentation_scope.name for span in spans}, {"langsmith"})
        self.assertEqual(sum(span.name == "double" for span in spans), 2)
        kinds = {span.attributes.get("openinference.span.kind") for span in spans}
        self.assertTrue({"CHAIN", "LLM", "TOOL"}.issubset(kinds))
        parent = next(span for span in spans if span.name == "MIRA Turn")
        self.assertIsNone(parent.parent)
        self.assertEqual(parent.attributes.get("openinference.span.kind"), "CHAIN")
        self.assertEqual(
            json.loads(parent.attributes["input.value"]),
            {"input": "Double twice visibly."},
        )
        self.assertEqual(
            json.loads(parent.attributes["output.value"]),
            {"output": "The final visible answer is 6."},
        )
        self.assertEqual(
            json.loads(parent.attributes["langsmith.metadata.usage_metadata"]),
            {"input_tokens": 18, "output_tokens": 12, "total_tokens": 30},
        )
        self.assertFalse(any(key.startswith("llm.") for key in parent.attributes))
        self.assertNotIn("gen_ai.usage.input_tokens", parent.attributes)
        self.assertEqual(
            sum(
                span.attributes.get("llm.token_count.total", 0)
                for span in spans
            ),
            30,
        )
        self.assertEqual(
            sum(
                span.parent is not None
                and span.parent.span_id == parent.context.span_id
                for span in spans
            ),
            2,
        )
        span_ids = {span.context.span_id for span in spans}
        self.assertTrue(
            all(
                span.parent is None or span.parent.span_id in span_ids
                for span in spans
            )
        )


if __name__ == "__main__":
    unittest.main()
