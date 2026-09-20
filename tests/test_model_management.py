"""Focused coverage for registry-backed model and subagent management."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from deepagents.backends import StateBackend

from agent.llm import get_llm, get_model_name, get_profile_model
from agent.mcp.prompts import PromptRegistry
from agent.resources.project_setup import ensure_project_examples
from agent.subagents.discovery import discover_subagents, effective_subagent_specs
from agent.subagents.compilation import compile_dynamic_subagents
from config.interpolation import EnvironmentInterpolationError, resolve_environment
from config.llm import load_model_registry
from config.metadata import ModelMetadata
from config.settings import (
    load_settings,
    set_model_assignment,
    set_subagent_enabled,
    set_subagent_model_assignment,
)
from ui.textual.widgets.autocomplete_input import _completion_row, attachment_items


class ModelManagementTests(unittest.TestCase):
    def test_bootstrap_creates_inert_registry_and_prompts_directory(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            ensure_project_examples(workspace)

            registry_text = (workspace / ".mira" / "models.yml").read_text(encoding="utf-8")
            self.assertIn("models:", registry_text)
            self.assertIn("example-cloud", registry_text)
            self.assertIn("example-endpoint", registry_text)
            self.assertEqual(registry_text.count("#   image_inputs: false"), 2)
            self.assertEqual(load_model_registry(workspace).profiles, {})
            self.assertTrue((workspace / ".mira" / "prompts").is_dir())

    def test_registry_keeps_valid_order_and_excludes_invalid_profiles(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            mira = workspace / ".mira"
            mira.mkdir()
            (mira / "models.yml").write_text(
                """models:
  first:
    provider: openai
    model: a
    api_key: ${TEST_MODEL_KEY}
  bad:
    provider: openai
    model: b
    unknown: value
  second:
    provider: anthropic
    model: c
""",
                encoding="utf-8",
            )
            registry = load_model_registry(workspace, environ={"TEST_MODEL_KEY": "secret"})
            self.assertEqual(list(registry.profiles), ["first", "second"])
            self.assertEqual(registry.invalid_names, ("bad",))
            self.assertEqual(registry.profiles["first"].values["api_key"], "secret")
            self.assertNotIn("secret", registry.issues[0].details)

    def test_registry_rejects_duplicate_yaml_and_reserved_model_kwargs(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            mira = workspace / ".mira"
            mira.mkdir()
            path = mira / "models.yml"
            path.write_text("models:\n  same: {provider: openai, model: a}\n  same: {provider: openai, model: b}\n", encoding="utf-8")
            self.assertFalse(load_model_registry(workspace).profiles)
            path.write_text(
                "models:\n  bad:\n    provider: openai\n    model: a\n    model_kwargs:\n      stream_options: {}\n",
                encoding="utf-8",
            )
            registry = load_model_registry(workspace)
            self.assertFalse(registry.profiles)
            self.assertIn("runtime-owned", registry.issues[0].details)

    def test_registry_validates_image_inputs_as_a_boolean(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            mira = workspace / ".mira"
            mira.mkdir()
            (mira / "models.yml").write_text(
                """models:
  omitted:
    provider: openai
    model: text-model
  disabled:
    provider: openai
    model: text-model
    image_inputs: false
  enabled:
    provider: openai
    model: vision-model
    image_inputs: true
  string-value:
    provider: openai
    model: invalid
    image_inputs: "true"
  integer-value:
    provider: openai
    model: invalid
    image_inputs: 1
  null-value:
    provider: openai
    model: invalid
    image_inputs: null
  nested-value:
    provider: openai
    model: invalid
    model_kwargs:
      image_inputs: true
""",
                encoding="utf-8",
            )

            registry = load_model_registry(workspace)

            self.assertEqual(list(registry.profiles), ["omitted", "disabled", "enabled"])
            self.assertNotIn("image_inputs", registry.profiles["omitted"].values)
            self.assertIs(registry.profiles["disabled"].values["image_inputs"], False)
            self.assertIs(registry.profiles["enabled"].values["image_inputs"], True)
            self.assertEqual(
                registry.invalid_names,
                ("string-value", "integer-value", "null-value", "nested-value"),
            )

    def test_interpolation_is_recursive_and_rejects_legacy_forms(self) -> None:
        value = {"a": ["Bearer ${TOKEN}", {"b": "${TOKEN}"}]}
        self.assertEqual(
            resolve_environment(value, environ={"TOKEN": "abc"}),
            {"a": ["Bearer abc", {"b": "abc"}]},
        )
        for invalid in ("${env:TOKEN}", "$${TOKEN}", "${TOKEN"):
            with self.subTest(invalid=invalid), self.assertRaises(EnvironmentInterpolationError):
                resolve_environment(invalid, environ={"TOKEN": "abc"})

    def test_registry_profiles_use_direct_chat_anyllm_arguments(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            mira = workspace / ".mira"
            mira.mkdir()
            (mira / "models.yml").write_text(
                "models:\n  local:\n    provider: lmstudio\n    model: demo\n    api_base: http://localhost:1234/v1\n",
                encoding="utf-8",
            )
            registry = load_model_registry(workspace)
            settings = set_model_assignment(load_settings(workspace), "main", "local")
            config = {"settings": settings, "model_registry": registry}
            model = Mock(profile=None)
            with patch("agent.llm.ChatAnyLLM", return_value=model) as constructor:
                self.assertIs(get_llm(config), model)
            constructor.assert_called_once_with(
                provider="lmstudio",
                model="demo",
                api_base="http://localhost:1234/v1",
                stream_options={"include_usage": True},
            )
            self.assertIs(model.profile["image_inputs"], False)
            self.assertEqual(model.profile["max_input_tokens"], 32768)
            self.assertEqual(get_model_name(config), "[local] lmstudio:demo")

    def test_image_capability_is_profile_metadata_not_constructor_input(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            mira = workspace / ".mira"
            mira.mkdir()
            (mira / "models.yml").write_text(
                """models:
  disabled:
    provider: anthropic
    model: text-model
    image_inputs: false
  enabled:
    provider: openai
    model: vision-model
    image_inputs: true
""",
                encoding="utf-8",
            )
            registry = load_model_registry(workspace)
            config = {"model_registry": registry}

            disabled = Mock(profile={"existing": "kept"})
            enabled = Mock(profile={"existing": "kept"})
            with patch("agent.llm.ChatAnyLLM", side_effect=[disabled, enabled]) as constructor:
                self.assertIs(
                    get_profile_model(
                        config,
                        "disabled",
                        metadata=ModelMetadata(context_tokens=4096),
                    ),
                    disabled,
                )
                self.assertIs(
                    get_profile_model(
                        config,
                        "enabled",
                        metadata=ModelMetadata(context_tokens=8192),
                    ),
                    enabled,
                )

            self.assertEqual(
                constructor.call_args_list[0].kwargs,
                {"provider": "anthropic", "model": "text-model"},
            )
            self.assertEqual(
                constructor.call_args_list[1].kwargs,
                {
                    "provider": "openai",
                    "model": "vision-model",
                    "stream_options": {"include_usage": True},
                },
            )
            self.assertEqual(
                disabled.profile,
                {"existing": "kept", "max_input_tokens": 4096, "image_inputs": False},
            )
            self.assertEqual(
                enabled.profile,
                {"existing": "kept", "max_input_tokens": 8192, "image_inputs": True},
            )

    def test_openai_profiles_request_stream_usage(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            mira = workspace / ".mira"
            mira.mkdir()
            (mira / "models.yml").write_text(
                "models:\n  cloud:\n    provider: openai\n    model: demo\n",
                encoding="utf-8",
            )
            registry = load_model_registry(workspace)
            settings = set_model_assignment(load_settings(workspace), "main", "cloud")
            config = {"settings": settings, "model_registry": registry}
            model = Mock(profile=None)
            with patch("agent.llm.ChatAnyLLM", return_value=model) as constructor:
                self.assertIs(get_llm(config), model)
            constructor.assert_called_once_with(
                provider="openai",
                model="demo",
                stream_options={"include_usage": True},
            )

    def test_subagents_default_disabled_and_overrides_copy_raw_specs(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            root = workspace / ".mira" / "subagents"
            root.mkdir(parents=True)
            source = root / "helpers.py"
            source.write_text(
                'SUBAGENTS = [{"name": "helper", "description": "Does focused work", '
                '"system_prompt": "Help", "mode": "fork"}]\n',
                encoding="utf-8",
            )
            discovery = discover_subagents(workspace)
            settings = load_settings(workspace)
            config = {"settings": settings}
            self.assertEqual([item["name"] for item in effective_subagent_specs(discovery, config)], ["general-purpose"])

            settings = set_subagent_enabled(settings, "helper", True)
            settings = set_subagent_model_assignment(settings, "helper", "fast")
            configured = {"settings": settings}
            replacement = object()
            with patch("agent.llm.get_profile_model", return_value=replacement):
                effective = effective_subagent_specs(discovery, configured)
            helper = next(item for item in effective if item["name"] == "helper")
            original = next(item.spec for item in discovery.items if item.name == "helper")
            self.assertIs(helper["model"], replacement)
            self.assertEqual(helper["mode"], "fork")
            self.assertEqual(original["mode"], "fork")
            self.assertNotIn("model", original)
            self.assertNotIn("model", source.read_text(encoding="utf-8"))

    def test_subagents_inherit_main_or_use_their_assigned_image_capability(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            mira = workspace / ".mira"
            subagents_root = mira / "subagents"
            subagents_root.mkdir(parents=True)
            (mira / "models.yml").write_text(
                """models:
  main-text:
    provider: openai
    model: text-model
    api_key: test-key
    image_inputs: false
  helper-vision:
    provider: openai
    model: vision-model
    api_key: test-key
    image_inputs: true
""",
                encoding="utf-8",
            )
            (subagents_root / "helper.py").write_text(
                'SUBAGENTS = [{"name": "helper", "description": "Looks at images", '
                '"system_prompt": "Inspect carefully."}]\n',
                encoding="utf-8",
            )
            settings = set_model_assignment(load_settings(workspace), "main", "main-text")
            settings = set_subagent_enabled(settings, "helper", True)
            settings = set_subagent_model_assignment(settings, "helper", "helper-vision")
            config = {
                "settings": settings,
                "model_registry": load_model_registry(workspace),
            }
            main_model = get_llm(config)
            effective = effective_subagent_specs(discover_subagents(workspace), config)

            with (
                patch(
                    "agent.subagents.compilation.create_summarization_middleware",
                    return_value="summary",
                ),
                patch(
                    "agent.subagents.compilation.create_sub_agent",
                    side_effect=lambda spec: spec,
                ),
            ):
                compiled = compile_dynamic_subagents(
                    effective,
                    model=main_model,
                    tools=[],
                    backend=StateBackend(),
                    skills=[],
                    permissions=[],
                    interrupt_on=None,
                )

            general = next(item["runnable"] for item in compiled if item["name"] == "general-purpose")
            helper = next(item["runnable"] for item in compiled if item["name"] == "helper")
            self.assertIs(general["model"], main_model)
            self.assertIs(general["model"].profile["image_inputs"], False)
            self.assertIs(helper["model"].profile["image_inputs"], True)

    def test_recursive_prompt_collisions_are_excluded(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            workspace = Path(directory)
            root = workspace / ".mira" / "prompts"
            (root / "review").mkdir(parents=True)
            (root / "review" / "python.md").write_text("Review {{path}}", encoding="utf-8")
            (root / "review__python.txt").write_text("Other", encoding="utf-8")
            registry = PromptRegistry(workspace)
            self.assertNotIn("/prompt__review__python", registry.specs)
            self.assertEqual(len(registry.issues), 1)
            self.assertIn("review/python.md", registry.issues[0].details)
            self.assertIn("review__python.txt", registry.issues[0].details)

    def test_suba_completion_matches_and_inserts_plain_guidance(self) -> None:
        items = attachment_items(
            [],
            [],
            "GENERAL-",
            subagents=[{"name": "general-purpose", "description": "Investigates broad tasks"}],
        )
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].kind, "subagent")
        self.assertEqual(items[0].insertion, "general-purpose subagent")
        row = _completion_row(items[0])
        self.assertEqual(row.plain, "SUBA  general-purpose subagent  Investigates broad tasks")
        self.assertIn("#93a4c3", str(row.spans[0].style).lower())
        self.assertTrue(row.no_wrap)
        self.assertEqual(row.overflow, "ellipsis")


if __name__ == "__main__":
    unittest.main()
