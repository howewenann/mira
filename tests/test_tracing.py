"""Focused checks for registry-backed LangSmith/OpenInference OTLP tracing."""

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


class _SemanticProcessor:
    instances: list["_SemanticProcessor"] = []

    def __init__(self) -> None:
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
            endpoint=profile.endpoint,
            headers=profile.headers,
        )
        self.assertIn("profile: corporate", persisted)
        self.assertNotIn("endpoint:", persisted)
        self.assertNotIn("headers:", persisted)
        self.assertIn("${TRACE_TOKEN}", preview)
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
        client = _Client.instances[0]
        self.assertEqual(
            client.kwargs,
            {"tracing_mode": "otel", "otel_tracer_provider": provider},
        )
        self.assertEqual(_Langsmith.configure_calls, [{"client": client, "enabled": True}])
        self.assertIsInstance(provider.processors[0], _SemanticProcessor)
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
class SemanticProcessorTests(unittest.TestCase):
    def _finished_spans(self, definitions: list[tuple[str, dict[str, object]]]):
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
        from tracing.semantic_processor import LangSmithOpenInferenceProcessor

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(LangSmithOpenInferenceProcessor())
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("mira.semantic-tests")
        for name, attributes in definitions:
            with tracer.start_as_current_span(name, attributes=attributes):
                pass
        provider.shutdown()
        return list(exporter.get_finished_spans())

    def test_span_kinds_input_output_and_tool_semantics(self) -> None:
        import json

        common = {
            "gen_ai.prompt": json.dumps({"value": 1}),
            "gen_ai.completion": json.dumps({"value": 2}),
        }
        spans = self._finished_spans(
            [
                ("chain", {**common, "langsmith.span.kind": "chain"}),
                ("research-agent", {**common, "langsmith.span.kind": "chain"}),
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
            ]
        )
        by_name = {span.name: span for span in spans}

        self.assertEqual(by_name["chain"].attributes["openinference.span.kind"], "CHAIN")
        self.assertEqual(
            by_name["research-agent"].attributes["openinference.span.kind"],
            "AGENT",
        )
        self.assertEqual(by_name["model"].attributes["openinference.span.kind"], "LLM")
        self.assertEqual(by_name["weather"].attributes["openinference.span.kind"], "TOOL")
        self.assertEqual(by_name["lookup"].attributes["openinference.span.kind"], "RETRIEVER")
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
                        "gen_ai.request.model": "model-1",
                        "gen_ai.usage.input_tokens": 11,
                        "gen_ai.usage.output_tokens": 7,
                        "gen_ai.usage.total_tokens": 18,
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
        self.assertEqual(attributes["llm.model_name"], "model-1")
        self.assertEqual(attributes["llm.token_count.prompt"], 11)
        self.assertEqual(attributes["llm.token_count.completion"], 7)
        self.assertEqual(attributes["llm.token_count.total"], 18)

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

    def test_real_langsmith_langchain_tree_is_enriched_without_duplicate_export(self) -> None:
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
                    ]
                )
                agent = create_agent(model, tools=[double], name="langsmith-smoke")
                result = agent.invoke({"messages": [HumanMessage("Double 2.")]})
                self.assertEqual(result["messages"][-1].content, "The answer is 4.")
        finally:
            langsmith.configure(client=None, enabled=None)
            if client is not None:
                client.close(timeout=1.0)
            request_patch.stop()
            provider.force_flush(timeout_millis=1_000)
            provider.shutdown()

        rest_request.assert_not_called()
        spans = list(exporter.get_finished_spans())
        self.assertGreaterEqual(len(spans), 4)
        self.assertEqual(len({span.context.trace_id for span in spans}), 1)
        self.assertEqual({span.instrumentation_scope.name for span in spans}, {"langsmith"})
        self.assertEqual(sum(span.name == "double" for span in spans), 1)
        kinds = {span.attributes.get("openinference.span.kind") for span in spans}
        self.assertTrue({"CHAIN", "LLM", "TOOL"}.issubset(kinds))
        span_ids = {span.context.span_id for span in spans}
        self.assertTrue(
            all(
                span.parent is None or span.parent.span_id in span_ids
                for span in spans
            )
        )


if __name__ == "__main__":
    unittest.main()
