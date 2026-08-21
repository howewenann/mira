"""Focused checks for registry-backed generic OTLP tracing."""

from __future__ import annotations

import ast
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

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


def _registry(
    *,
    name: str = "phoenix",
    endpoint: str = "http://127.0.0.1:6006/v1/traces",
    headers: dict[str, str] | None = None,
) -> TracingRegistry:
    profile = TracingProfile(name, endpoint, dict(headers or {}))
    return TracingRegistry({name: profile})


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
        configured = settings.set_tracing_profile(
            settings.set_tracing_enabled(settings.DEFAULT_SETTINGS, True),
            "corporate",
        )

        issues = bootstrap.configure_tracing(configured, _registry(), environ={})

        self.assertEqual(len(issues), 1)
        self.assertEqual(
            issues[0].summary,
            "Tracing profile 'corporate' was not found in .mira/tracing.yml.",
        )
        self.assertIn("add it", issues[0].details)

    def test_missing_extra_is_a_friendly_nonfatal_issue(self) -> None:
        configured = settings.set_tracing_enabled(settings.DEFAULT_SETTINGS, True)
        with patch.object(bootstrap, "_load_runtime", side_effect=ModuleNotFoundError("opentelemetry.sdk")):
            issues = bootstrap.configure_tracing(configured, _registry(), environ={})

        self.assertEqual(len(issues), 1)
        self.assertIn("Tracing dependencies", issues[0].summary)
        self.assertIn('pip install "mira[tracing]"', issues[0].guidance)
        self.assertNotIn("langsmith[otel]", issues[0].guidance)

    def test_missing_environment_reference_does_not_expose_a_value(self) -> None:
        configured = settings.set_tracing_profile(
            settings.set_tracing_enabled(settings.DEFAULT_SETTINGS, True),
            "corporate",
        )
        registry = _registry(
            name="corporate",
            endpoint="https://example.com/v1/traces",
            headers={"Authorization": "Bearer ${TRACE_TOKEN}"},
        )

        issues = bootstrap.configure_tracing(configured, registry, environ={})

        self.assertEqual(len(issues), 1)
        self.assertIn("TRACE_TOKEN", issues[0].details)
        self.assertIn("process environment", issues[0].details)
        self.assertNotIn("resolved-secret", issues[0].details)

    def test_settings_and_preview_keep_environment_references_unresolved(self) -> None:
        configured = settings.set_tracing_profile(
            settings.set_tracing_enabled(settings.DEFAULT_SETTINGS, True),
            "corporate",
        )
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

    def test_runtime_receives_only_resolved_profile_values(self) -> None:
        configured = settings.set_tracing_profile(
            settings.set_tracing_enabled(settings.DEFAULT_SETTINGS, True),
            "corporate",
        )
        registry = _registry(
            name="corporate",
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
            issues = bootstrap.configure_tracing(configured, registry, environ=environ)

        self.assertEqual(issues, [])
        self.assertEqual(environ["OTEL_EXPORTER_OTLP_ENDPOINT"], "https://example.com/otel/v1/traces")
        self.assertEqual(
            environ["OTEL_EXPORTER_OTLP_HEADERS"],
            "Authorization=Bearer%20resolved-secret,tenant=my%20team",
        )
        self.assertEqual(
            _Exporter.instances[0].kwargs["headers"]["Authorization"],
            "Bearer resolved-secret",
        )

    def test_reconfiguration_uses_generic_profile_data_without_vendor_branches(self) -> None:
        configured = settings.set_tracing_enabled(settings.DEFAULT_SETTINGS, True)
        environ: dict[str, str] = {}
        fake_langsmith = Mock(Client=_Client)
        fake_langsmith.configure = Mock()
        runtime = (fake_langsmith, _Provider, _Processor, _Exporter)
        with patch.object(bootstrap, "_load_runtime", return_value=runtime):
            self.assertEqual(bootstrap.configure_tracing(configured, _registry(), environ=environ), [])
            first_client = _Client.instances[0]
            first_provider = _Provider.instances[0]
            changed = settings.set_tracing_profile(configured, "corporate")
            corporate = _registry(
                name="corporate",
                endpoint="https://changed.example/v1/traces",
            )
            self.assertEqual(bootstrap.configure_tracing(changed, corporate, environ=environ), [])

        self.assertTrue(first_client.closed)
        self.assertTrue(first_provider.shutdown_called)
        self.assertEqual(environ["OTEL_EXPORTER_OTLP_ENDPOINT"], "https://changed.example/v1/traces")
        runtime_tree = ast.parse(inspect.getsource(bootstrap.configure_tracing))
        branch_conditions = "\n".join(
            ast.unparse(node.test).lower()
            for node in ast.walk(runtime_tree)
            if isinstance(node, ast.If)
        )
        self.assertNotIn("phoenix", branch_conditions)
        self.assertNotIn("langsmith", branch_conditions)


if __name__ == "__main__":
    unittest.main()
