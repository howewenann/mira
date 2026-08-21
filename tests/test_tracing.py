"""Focused checks for registry-backed OpenInference OTLP tracing."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import tempfile
import unittest
from pathlib import Path
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


class _Instrumentor:
    instances: list["_Instrumentor"] = []
    fail_instrument = False

    def __init__(self) -> None:
        self.instrument_calls: list[dict[str, object]] = []
        self.uninstrument_calls = 0
        self.instances.append(self)

    def instrument(self, **kwargs: object) -> None:
        self.instrument_calls.append(kwargs)
        if self.fail_instrument:
            raise RuntimeError("instrumentation failed")

    def uninstrument(self) -> None:
        self.uninstrument_calls += 1


def _runtime() -> tuple[object, ...]:
    return (
        _Instrumentor,
        _Resource,
        _ResourceAttributes,
        _Provider,
        _Processor,
        _Exporter,
    )


def _registry(
    *,
    name: str = "phoenix",
    endpoint: str = "http://127.0.0.1:6006/v1/traces",
    headers: dict[str, str] | None = None,
) -> TracingRegistry:
    profile = TracingProfile(name, endpoint, dict(headers or {}))
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
                "openinference.instrumentation.langchain",
                "opentelemetry.sdk.trace",
            )
        )
    except ModuleNotFoundError:
        return False


class TracingTests(unittest.TestCase):
    def setUp(self) -> None:
        bootstrap._active_instrumentor = None
        bootstrap._active_provider = None
        _Resource.instances.clear()
        _Provider.instances.clear()
        _Exporter.instances.clear()
        _Instrumentor.instances.clear()
        _Instrumentor.fail_instrument = False

    def tearDown(self) -> None:
        bootstrap._active_instrumentor = None
        bootstrap._active_provider = None

    def test_registry_bootstraps_exact_profiles_when_missing(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            path = bootstrap_tracing_registry(workspace)

            self.assertEqual(path, tracing_path(workspace))
            self.assertEqual(path.read_text(encoding="utf-8"), TRACING_REGISTRY_TEMPLATE)
            self.assertEqual(
                yaml.safe_load(path.read_text(encoding="utf-8")),
                {
                    "profiles": {
                        "phoenix": {
                            "endpoint": "http://127.0.0.1:6006/v1/traces",
                            "headers": {},
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
            custom = "profiles:\n  corporate:\n    endpoint: https://otel.example/v1/traces\n    headers: {}\n"
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

    def test_invalid_headers_are_a_friendly_nonfatal_issue(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            path = tracing_path(workspace)
            path.parent.mkdir()
            path.write_text(
                "profiles:\n  broken:\n    endpoint: https://otel.example/v1/traces\n    headers: []\n",
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
            side_effect=ModuleNotFoundError("openinference.instrumentation.langchain"),
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
            endpoint=profile.endpoint,
            headers=profile.headers,
        )
        self.assertIn("profile: corporate", persisted)
        self.assertNotIn("endpoint:", persisted)
        self.assertNotIn("headers:", persisted)
        self.assertIn("${TRACE_TOKEN}", preview)
        self.assertNotIn("resolved-secret", preview)

    def test_instrumentor_exporter_and_resource_receive_resolved_profile(self) -> None:
        registry = TracingRegistry(
            {
                "corporate": TracingProfile(
                    "corporate",
                    "https://example.com/otel/v1/traces",
                    {
                        "Authorization": "Bearer ${TRACE_TOKEN}",
                        "tenant": "my team",
                    },
                ),
                "unused": TracingProfile(
                    "unused",
                    "https://unused.example/v1/traces",
                    {"Authorization": "Bearer ${UNRESOLVED_UNUSED_TOKEN}"},
                ),
            }
        )
        environ = {"TRACE_TOKEN": "resolved-secret"}

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
        instrumentor = _Instrumentor.instances[0]
        self.assertEqual(instrumentor.instrument_calls, [{"tracer_provider": provider}])
        self.assertEqual(
            provider.resource.attributes,
            {
                "service.name": "MIRA",
                "openinference.project.name": "MIRA",
            },
        )
        self.assertEqual(environ, {"TRACE_TOKEN": "resolved-secret"})

    def test_reconfiguration_tears_down_old_runtime_and_initializes_new(self) -> None:
        with patch.object(bootstrap, "_load_runtime", return_value=_runtime()):
            self.assertEqual(
                bootstrap.configure_tracing(_tracing_settings(), _registry(), environ={}),
                [],
            )
            first_instrumentor = _Instrumentor.instances[0]
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

        self.assertEqual(first_instrumentor.uninstrument_calls, 1)
        self.assertEqual(first_provider.force_flush_calls, [bootstrap._FLUSH_TIMEOUT_MILLIS])
        self.assertTrue(first_provider.shutdown_called)
        self.assertEqual(len(_Instrumentor.instances), 2)
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
        instrumentor = _Instrumentor.instances[0]
        provider = _Provider.instances[0]

        with patch.object(bootstrap, "_load_runtime") as load_runtime:
            self.assertEqual(
                bootstrap.configure_tracing(settings.DEFAULT_SETTINGS, environ={}),
                [],
            )
            bootstrap.shutdown_tracing()

        load_runtime.assert_not_called()
        self.assertEqual(instrumentor.uninstrument_calls, 1)
        self.assertEqual(provider.force_flush_calls, [bootstrap._FLUSH_TIMEOUT_MILLIS])
        self.assertTrue(provider.shutdown_called)

    def test_initialization_failure_cleans_partial_runtime_and_returns_issue(self) -> None:
        _Instrumentor.fail_instrument = True
        with patch.object(bootstrap, "_load_runtime", return_value=_runtime()):
            issues = bootstrap.configure_tracing(_tracing_settings(), _registry(), environ={})

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].summary, "Tracing could not be initialized")
        self.assertIn("RuntimeError", issues[0].details)
        self.assertIsNone(bootstrap._active_instrumentor)
        self.assertIsNone(bootstrap._active_provider)
        self.assertEqual(_Instrumentor.instances[0].uninstrument_calls, 1)
        self.assertEqual(
            _Provider.instances[0].force_flush_calls,
            [bootstrap._FLUSH_TIMEOUT_MILLIS],
        )
        self.assertTrue(_Provider.instances[0].shutdown_called)

    def test_exporter_failure_shuts_down_provider_without_attaching_instrumentor(self) -> None:
        runtime = (*_runtime()[:-1], _FailingExporter)
        with patch.object(bootstrap, "_load_runtime", return_value=runtime):
            issues = bootstrap.configure_tracing(_tracing_settings(), _registry(), environ={})

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].summary, "Tracing could not be initialized")
        self.assertIn("MIRA remains usable", issues[0].guidance)
        self.assertEqual(_Instrumentor.instances, [])
        self.assertEqual(
            _Provider.instances[0].force_flush_calls,
            [bootstrap._FLUSH_TIMEOUT_MILLIS],
        )
        self.assertTrue(_Provider.instances[0].shutdown_called)

    def test_external_langsmith_tracing_warns_without_mutating_environment(self) -> None:
        environ = {
            "LANGSMITH_TRACING": "true",
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

        self.assertEqual(environ, original)
        self.assertEqual(len(issues), 1)
        self.assertIn("External LangSmith tracing", issues[0].summary)
        self.assertIn("duplicate traces", issues[0].details)
        self.assertNotIn("secret-value", repr(issues[0]))
        self.assertIsNotNone(bootstrap._active_provider)

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
        self.assertNotIn("langsmith", branch_conditions)


@unittest.skipUnless(_real_tracing_available(), "requires the mira[tracing] extra")
class OpenInferenceIntegrationTests(unittest.TestCase):
    def test_real_langchain_agent_model_and_tool_emit_openinference_spans(self) -> None:
        from collections.abc import Callable, Sequence
        from typing import Any

        from langchain.agents import create_agent
        from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
        from langchain_core.messages import AIMessage, HumanMessage
        from langchain_core.runnables import Runnable
        from langchain_core.tools import BaseTool, tool
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

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
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        instrumentor = LangChainInstrumentor()
        try:
            instrumentor.instrument(tracer_provider=provider)
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
                ]
            )
            agent = create_agent(model, tools=[double], name="openinference-smoke")
            result = agent.invoke({"messages": [HumanMessage("Double 2.")]})
            self.assertEqual(result["messages"][-1].content, "The answer is 4.")
        finally:
            instrumentor.uninstrument()
            provider.force_flush(timeout_millis=1_000)
            provider.shutdown()

        spans = list(exporter.get_finished_spans())
        self.assertGreaterEqual(len(spans), 4)
        kinds = {span.attributes.get("openinference.span.kind") for span in spans}
        self.assertIn("LLM", kinds)
        self.assertIn("TOOL", kinds)
        self.assertIn("CHAIN", kinds)

        semantic_spans = [
            span for span in spans if span.attributes.get("openinference.span.kind")
        ]
        self.assertTrue(all("input.value" in span.attributes for span in semantic_spans))
        self.assertTrue(all("output.value" in span.attributes for span in semantic_spans))

        model_spans = [
            span
            for span in spans
            if span.attributes.get("openinference.span.kind") == "LLM"
        ]
        self.assertTrue(
            any(
                "llm.input_messages.0.message.role" in span.attributes
                for span in model_spans
            )
        )
        self.assertTrue(
            any("llm.token_count.total" in span.attributes for span in model_spans)
        )

        tool_spans = [
            span
            for span in spans
            if span.attributes.get("openinference.span.kind") == "TOOL"
        ]
        self.assertTrue(any(span.attributes.get("tool.name") == "double" for span in tool_spans))

        span_ids = {span.context.span_id for span in spans}
        self.assertTrue(
            all(
                span.parent is None or span.parent.span_id in span_ids
                for span in spans
            )
        )


if __name__ == "__main__":
    unittest.main()
