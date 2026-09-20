"""
Multimodal boundary probe for MIRA / DeepAgents / ChatAnyLLM.

Run from MIRA repo root:

    python tests/probes/multimodal_probe.py

What this proves:

1. DeepAgents read_file emits canonical LangChain image blocks.
2. DeepAgents uses model.profile["image_inputs"] to gate images.
3. DeepAgents filtering is transient -- graph/state message is not rewritten.
4. image_tool_message is independently respected.
5. ChatAnyLLM currently drops canonical {"type": "image"} blocks.
6. LangChain's convert_to_openai_image_block() produces image_url.
7. ChatAnyLLM preserves that image_url representation.
8. Existing image_url and plain text are unaffected.
9. MIRA currently rejects image_inputs in models.yml.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import json
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from deepagents.backends import FilesystemBackend
from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import ToolMessage, convert_to_openai_image_block
from langchain_anyllm import ChatAnyLLM

from config.llm import load_model_registry


PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "YAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)

FAILURES: list[str] = []


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "<not installed>"


def check(name: str, condition: bool, detail: Any = None) -> None:
    marker = "PASS" if condition else "FAIL"
    print(f"\n[{marker}] {name}")
    if detail is not None:
        print(pretty(detail))
    if not condition:
        FAILURES.append(name)


def pretty(value: Any) -> str:
    """Pretty-print without dumping whole base64 payloads."""

    def scrub(obj: Any) -> Any:
        if isinstance(obj, dict):
            result = {}
            for key, item in obj.items():
                if key == "base64" and isinstance(item, str):
                    result[key] = f"<{len(item)} base64 chars>"
                elif key == "url" and isinstance(item, str) and item.startswith("data:"):
                    prefix, _, payload = item.partition(",")
                    result[key] = f"{prefix},<{len(payload)} base64 chars>"
                else:
                    result[key] = scrub(item)
            return result
        if isinstance(obj, list):
            return [scrub(item) for item in obj]
        return obj

    try:
        return json.dumps(scrub(value), indent=2, default=str)
    except TypeError:
        return repr(value)


def first_block(message: ToolMessage) -> dict[str, Any]:
    blocks = message.content_blocks
    assert blocks, "message contains no content blocks"
    block = blocks[0]
    assert isinstance(block, dict)
    return block


async def actual_deepagents_read_file(
    root: Path,
) -> tuple[FilesystemMiddleware, ToolMessage]:
    """Call the actual DeepAgents read_file tool."""
    png_path = root / "probe.png"
    png_path.write_bytes(base64.b64decode(PNG_BASE64))

    backend = FilesystemBackend(root_dir=root, virtual_mode=True)
    middleware = FilesystemMiddleware(backend=backend, tools=["read_file"])

    read_file = next(tool for tool in middleware.tools if tool.name == "read_file")
    assert read_file.coroutine is not None

    runtime = SimpleNamespace(tool_call_id="probe-read-image")
    result = await read_file.coroutine(
        file_path="/probe.png",
        runtime=runtime,
        offset=0,
        limit=100,
    )

    assert isinstance(result, ToolMessage), (
        f"Expected ToolMessage for PNG read, got {type(result)!r}"
    )
    return middleware, result


async def deepagents_model_boundary(
    middleware: FilesystemMiddleware,
    message: ToolMessage,
    profile: dict[str, Any],
) -> list[Any]:
    """Capture exactly what DeepAgents passes to the downstream model handler."""
    model = ChatAnyLLM(
        provider="openai",
        model="probe-model",
        api_key="probe-key",
    )
    model.profile = dict(profile)

    request = ModelRequest(model=model, messages=[message], tools=[])
    captured: dict[str, Any] = {}

    async def handler(bound_request: ModelRequest) -> object:
        captured["messages"] = list(bound_request.messages)
        return object()

    await middleware.awrap_model_call(request, handler)
    return captured["messages"]


async def capture_anyllm_outbound(
    message: ToolMessage,
) -> list[dict[str, Any]]:
    """Capture exact messages ChatAnyLLM is about to pass to any-llm."""
    model = ChatAnyLLM(
        provider="openai",
        model="probe-model",
        api_key="probe-key",
    )

    captured: dict[str, Any] = {}

    class ProbeComplete(Exception):
        pass

    async def fake_acompletion(*args: Any, **kwargs: Any) -> Any:
        captured["messages"] = copy.deepcopy(kwargs["messages"])
        raise ProbeComplete

    with patch("langchain_anyllm.chat_models.acompletion", new=fake_acompletion):
        try:
            await model.ainvoke([message])
        except ProbeComplete:
            pass

    assert "messages" in captured, "ChatAnyLLM never reached acompletion()"
    return captured["messages"]


async def probe() -> None:
    print("=" * 72)
    print("VERSIONS")
    print("=" * 72)

    for package in (
        "deepagents",
        "langchain",
        "langchain-core",
        "langchain-anyllm",
        "any-llm-sdk",
    ):
        print(f"{package:20} {package_version(package)}")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)

        middleware, image_message = await actual_deepagents_read_file(root)
        block = first_block(image_message)

        check(
            "DeepAgents read_file emits canonical image block",
            block.get("type") == "image"
            and isinstance(block.get("base64"), str)
            and block.get("mime_type") == "image/png",
            image_message.content_blocks,
        )

        original_content = copy.deepcopy(image_message.content)

        supported_messages = await deepagents_model_boundary(
            middleware,
            image_message,
            {"image_inputs": True},
        )
        supported = supported_messages[0]
        supported_block = first_block(supported)

        check(
            "DeepAgents keeps image when image_inputs=True",
            supported_block.get("type") == "image",
            supported.content_blocks,
        )

        unsupported_messages = await deepagents_model_boundary(
            middleware,
            image_message,
            {"image_inputs": False},
        )
        unsupported = unsupported_messages[0]
        unsupported_blocks = unsupported.content_blocks
        unsupported_is_placeholder = (
            len(unsupported_blocks) == 1
            and unsupported_blocks[0].get("type") == "text"
            and "does not support image content"
            in unsupported_blocks[0].get("text", "")
        )

        check(
            "DeepAgents replaces image with placeholder when image_inputs=False",
            unsupported_is_placeholder,
            unsupported_blocks,
        )

        check(
            "DeepAgents capability filtering does not rewrite original message",
            image_message.content == original_content
            and first_block(image_message).get("type") == "image",
            image_message.content_blocks,
        )

        tool_unsupported_messages = await deepagents_model_boundary(
            middleware,
            image_message,
            {
                "image_inputs": True,
                "image_tool_message": False,
            },
        )
        tool_unsupported = tool_unsupported_messages[0]

        check(
            "DeepAgents separately respects image_tool_message=False",
            tool_unsupported.content_blocks[0].get("type") == "text"
            and "does not support image content"
            in tool_unsupported.content_blocks[0].get("text", ""),
            tool_unsupported.content_blocks,
        )

        current_anyllm = await capture_anyllm_outbound(image_message)
        current_content = current_anyllm[0].get("content")

        check(
            "Current ChatAnyLLM drops canonical LangChain image block",
            current_content in ("", [], None),
            current_anyllm,
        )

        converted_block = convert_to_openai_image_block(block)

        check(
            "LangChain convert_to_openai_image_block produces image_url",
            converted_block.get("type") == "image_url"
            and converted_block.get("image_url", {})
            .get("url", "")
            .startswith("data:image/png;base64,"),
            converted_block,
        )

        converted_message = image_message.model_copy(
            update={"content": [converted_block]}
        )
        converted_anyllm = await capture_anyllm_outbound(converted_message)
        converted_content = converted_anyllm[0].get("content")

        survived = (
            isinstance(converted_content, list)
            and len(converted_content) == 1
            and converted_content[0].get("type") == "image_url"
        )

        check(
            "ChatAnyLLM preserves image after LangChain image_url conversion",
            survived,
            converted_anyllm,
        )

        existing_image_url = ToolMessage(
            name="read_file",
            tool_call_id="probe-existing-image-url",
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64," + PNG_BASE64,
                    },
                }
            ],
        )
        existing_anyllm = await capture_anyllm_outbound(existing_image_url)
        existing_content = existing_anyllm[0].get("content")

        check(
            "Existing image_url already survives ChatAnyLLM unchanged",
            isinstance(existing_content, list)
            and existing_content[0].get("type") == "image_url",
            existing_anyllm,
        )

        text_message = ToolMessage(
            name="read_file",
            tool_call_id="probe-text",
            content="hello from read_file",
        )
        text_anyllm = await capture_anyllm_outbound(text_message)

        check(
            "Plain text is unaffected",
            text_anyllm[0].get("content") == "hello from read_file",
            text_anyllm,
        )

        placeholder_anyllm = await capture_anyllm_outbound(unsupported)
        placeholder_content = placeholder_anyllm[0].get("content")

        check(
            "DeepAgents unsupported-image placeholder survives ChatAnyLLM",
            isinstance(placeholder_content, str)
            and "does not support image content" in placeholder_content,
            placeholder_anyllm,
        )

        mira_dir = root / ".mira"
        mira_dir.mkdir(exist_ok=True)
        (mira_dir / "models.yml").write_text(
            """models:
  vision-test:
    provider: openai
    model: probe-model
    image_inputs: true
""",
            encoding="utf-8",
        )

        registry = load_model_registry(root)
        issue_details = [
            getattr(issue, "details", str(issue))
            for issue in registry.issues
        ]

        check(
            "MIRA currently rejects image_inputs in model profile",
            "vision-test" in registry.invalid_names
            and any("image_inputs" in detail for detail in issue_details),
            {
                "profiles": list(registry.profiles),
                "invalid_names": registry.invalid_names,
                "issues": issue_details,
            },
        )

        print("\n" + "=" * 72)
        print("BOUNDARY SUMMARY")
        print("=" * 72)
        print(
            """
DeepAgents read_file
    PNG
     |
     v
canonical LangChain {"type": "image", "base64": ..., "mime_type": ...}
     |
     +-- image_inputs=False ------------------------+
     |                                              |
     |                                    DeepAgents placeholder
     |                                              |
     |                                              v
     |                                           AnyLLM
     |
     +-- image_inputs=True
             |
             v
       canonical image
             |
             v
       CURRENT AnyLLM
             |
             X  block dropped

Candidate boundary normalization:

canonical image
     |
     v
LangChain convert_to_openai_image_block()
     |
     v
{"type": "image_url", ...}
     |
     v
ChatAnyLLM
     |
     v
preserved
""".strip()
        )

    print("\n" + "=" * 72)

    if FAILURES:
        print(f"PROBE FAILED: {len(FAILURES)} check(s)")
        for failure in FAILURES:
            print(f"  - {failure}")
        raise SystemExit(1)

    print("ALL PROBE CHECKS PASSED")
    print("=" * 72)


if __name__ == "__main__":
    asyncio.run(probe())
