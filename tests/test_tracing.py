"""Focused checks for optional generic OTLP tracing."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from config import settings
from tracing import bootstrap


class _Provider:
    instances: list["_Provider"] = []

    def __init__(self) -> None:
        self.processors: list[object] = []
        self.shutdown_called = False
        self.instances.append(self)

    def add_span_processor(self, processor: object) -> None:
        self.processors.append(processor)

    def shutdown(self) -> None:
        self.shutdown_called = True


class _Exporter:
    instances: list["_Exporter"] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.instances.append(self)


class _Processor:
    def __init__(self, exporter: object) -> None:
        self.exporter = exporter


class _Client:
    instances: list["_Client"] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.closed = False
        self.instances.append(self)

    def close(self, timeout: float | None = None) -> None:
        self.closed = True


class TracingTests(unittest.TestCase):
    def setUp(self) -> None:
        bootstrap._active_client = None
        bootstrap._active_provider = None
        bootstrap._saved_environment = None
        _Provider.instances.clear()
        _Exporter.instances.clear()
        _Client.instances.clear()

    def tearDown(self) -> None:
        bootstrap._active_client = None
        bootstrap._active_provider = None
        bootstrap._saved_environment = None

    def test_disabled_tracing_never_loads_optional_runtime(self) -> None:
        with patch.object(bootstrap, "_load_runtime") as load_runtime:
            issues = bootstrap.configure_tracing(settings.DEFAULT_SETTINGS, environ={})

        self.assertEqual(issues, [])
        load_runtime.assert_not_called()

    def test_missing_extra_is_a_friendly_nonfatal_issue(self) -> None:
        configured = settings.set_tracing_enabled(settings.DEFAULT_SETTINGS, True)
        with patch.object(bootstrap, "_load_runtime", side_effect=ModuleNotFoundError("opentelemetry.sdk")):
            issues = bootstrap.configure_tracing(configured, environ={})

        self.assertEqual(len(issues), 1)
        self.assertIn("Tracing dependencies", issues[0].summary)
        self.assertIn('pip install "mira[tracing]"', issues[0].guidance)
        self.assertNotIn("langsmith[otel]", issues[0].guidance)

    def test_missing_environment_reference_does_not_suggest_dotenv(self) -> None:
        configured = settings.set_tracing_config(
            settings.set_tracing_enabled(settings.DEFAULT_SETTINGS, True),
            endpoint="https://example.com/v1/traces",
            headers={"Authorization": "Bearer ${TRACE_TOKEN}"},
        )

        issues = bootstrap.configure_tracing(configured, environ={})

        self.assertEqual(len(issues), 1)
        self.assertIn("TRACE_TOKEN", issues[0].details)
        self.assertIn("process environment", issues[0].details)
        self.assertNotIn(".env", issues[0].details)

    def test_headers_parse_blank_mapping_and_reject_invalid_yaml(self) -> None:
        self.assertEqual(bootstrap.parse_headers_yaml(""), {})
        self.assertEqual(
            bootstrap.parse_headers_yaml('Authorization: "Bearer ${TRACE_TOKEN}"\ntenant: my-team'),
            {"Authorization": "Bearer ${TRACE_TOKEN}", "tenant": "my-team"},
        )
        for value in ("- one\n- two", "Authorization: [", "Authorization: 4"):
            with self.assertRaises(ValueError):
                bootstrap.parse_headers_yaml(value)

        issues = settings.settings_issues({"tracing": {"headers": []}})
        self.assertEqual(len(issues), 1)
        self.assertIn("tracing.headers", issues[0].summary)

    def test_settings_and_preview_keep_environment_references_unresolved(self) -> None:
        configured = settings.set_tracing_config(
            settings.set_tracing_enabled(settings.DEFAULT_SETTINGS, True),
            endpoint="https://example.com/otel/v1/traces",
            headers={"Authorization": "Bearer ${TRACE_TOKEN}"},
        )
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            self.assertTrue(settings.save_settings(workspace, configured))
            persisted = settings.settings_path(workspace).read_text(encoding="utf-8")

        preview = bootstrap.tracing_yaml_fragment(**settings.tracing_settings(configured))
        self.assertIn("${TRACE_TOKEN}", persisted)
        self.assertIn("${TRACE_TOKEN}", preview)
        self.assertNotIn("resolved-secret", persisted)
        self.assertNotIn("resolved-secret", preview)

    def test_runtime_receives_resolved_values_and_required_process_configuration(self) -> None:
        configured = settings.set_tracing_config(
            settings.set_tracing_enabled(settings.DEFAULT_SETTINGS, True),
            endpoint="https://example.com/otel/v1/traces",
            headers={"Authorization": "Bearer ${TRACE_TOKEN}", "tenant": "my team"},
        )
        environ = {"TRACE_TOKEN": "resolved-secret"}
        fake_langsmith = Mock(Client=_Client)
        fake_langsmith.configure = Mock()

        with patch.object(
            bootstrap,
            "_load_runtime",
            return_value=(fake_langsmith, _Provider, _Processor, _Exporter),
        ):
            issues = bootstrap.configure_tracing(configured, environ=environ)

        self.assertEqual(issues, [])
        self.assertEqual(environ["LANGSMITH_TRACING"], "true")
        self.assertEqual(environ["LANGSMITH_TRACING_MODE"], "otel")
        self.assertEqual(environ["OTEL_EXPORTER_OTLP_ENDPOINT"], "https://example.com/otel/v1/traces")
        self.assertEqual(
            environ["OTEL_EXPORTER_OTLP_HEADERS"],
            "Authorization=Bearer%20resolved-secret,tenant=my%20team",
        )
        exporter_kwargs = _Exporter.instances[0].kwargs
        self.assertEqual(exporter_kwargs["headers"]["Authorization"], "Bearer resolved-secret")
        fake_langsmith.configure.assert_called_once_with(client=_Client.instances[0], enabled=True)

    def test_reconfiguration_closes_old_client_and_recreates_exporter(self) -> None:
        configured = settings.set_tracing_enabled(settings.DEFAULT_SETTINGS, True)
        environ: dict[str, str] = {}
        fake_langsmith = Mock(Client=_Client)
        fake_langsmith.configure = Mock()
        runtime = (fake_langsmith, _Provider, _Processor, _Exporter)
        with patch.object(bootstrap, "_load_runtime", return_value=runtime):
            self.assertEqual(bootstrap.configure_tracing(configured, environ=environ), [])
            first_client = _Client.instances[0]
            first_provider = _Provider.instances[0]
            changed = settings.set_tracing_config(
                configured,
                endpoint="https://changed.example/v1/traces",
                headers={},
            )
            self.assertEqual(bootstrap.configure_tracing(changed, environ=environ), [])

        self.assertTrue(first_client.closed)
        self.assertTrue(first_provider.shutdown_called)
        self.assertEqual(len(_Client.instances), 2)
        self.assertEqual(environ["OTEL_EXPORTER_OTLP_ENDPOINT"], "https://changed.example/v1/traces")


if __name__ == "__main__":
    unittest.main()
