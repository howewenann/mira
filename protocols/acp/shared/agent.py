"""Shared ACP agent behavior over MIRA's public application API."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from acp import PROTOCOL_VERSION
from acp.exceptions import RequestError
from acp.schema import (
    AgentCapabilities,
    ClientCapabilities,
    Implementation,
    InitializeResponse,
    LoadSessionResponse,
    NewSessionResponse,
    PromptCapabilities,
    PromptResponse,
    SessionConfigOptionSelect,
    SessionConfigSelectOption,
    SessionMode,
    SessionModeState,
    SetSessionConfigOptionResponse,
    SetSessionModeResponse,
    TextContentBlock,
)

from config.version import __version__
from mira import MiraApplication, MiraSession
from protocols.acp.shared.frontend import ACPFrontend, InteractionCancelled, ReplyInChat


MIRA_MODE_OPTIONS = [
    SessionMode(id="act", name="Act", description="Execute through MIRA's action workflow."),
    SessionMode(id="plan", name="Plan", description="Use MIRA's formal planning workflow."),
]


class MiraAgent:
    """Implement the stock ACP Agent interface over workspace-scoped MIRA apps."""

    def __init__(self) -> None:
        self._applications: dict[str, MiraApplication] = {}
        self._application_locks: dict[str, asyncio.Lock] = {}
        self._session_workspaces: dict[str, Path] = {}
        self._session_modes: dict[str, str] = {}
        self._mira_sessions: dict[str, MiraSession] = {}
        self.frontend = ACPFrontend(
            snapshot=lambda session_id: self._mira_sessions[session_id].snapshot()
        )

    def on_connect(self, connection: Any) -> None:
        self.frontend.connection = connection

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        del protocol_version, client_capabilities, client_info, kwargs
        return InitializeResponse(
            protocol_version=PROTOCOL_VERSION,
            agent_capabilities=AgentCapabilities(
                load_session=True,
                prompt_capabilities=PromptCapabilities(),
            ),
            agent_info=Implementation(name="mira", title="MIRA", version=__version__),
        )

    async def new_session(
        self,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[Any] | None = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        del kwargs
        self._reject_external_configuration(additional_directories, mcp_servers)
        session_id = uuid4().hex
        self._initialize_session(session_id, self._workspace(cwd), "act")
        return NewSessionResponse(
            session_id=session_id,
            modes=self._mode_state(session_id),
            config_options=self._config_options(session_id),
        )

    async def load_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list[Any] | None = None,
        additional_directories: list[str] | None = None,
        **kwargs: Any,
    ) -> LoadSessionResponse:
        del kwargs
        self._reject_external_configuration(additional_directories, mcp_servers)
        workspace = self._workspace(cwd)
        with self.frontend.bind(session_id):
            application = await self._application(workspace)
            if not application.session_exists(session_id):
                self._forget_session(session_id)
                raise RequestError.resource_not_found(session_id)
            session = self._mira_sessions.get(session_id)
            if session is None or session.application is not application:
                session = await application.open_session(session_id=session_id)
                self._mira_sessions[session_id] = session
            mode = "plan" if session.snapshot().mode == "planning" else "act"
            self._initialize_session(session_id, workspace, mode)
            await self.frontend.replay(session_id, session.snapshot().transcript)
        return LoadSessionResponse(
            modes=self._mode_state(session_id),
            config_options=self._config_options(session_id),
        )

    async def prompt(
        self,
        session_id: str,
        prompt: list[Any],
        **kwargs: Any,
    ) -> PromptResponse:
        del kwargs
        if not prompt or any(not isinstance(block, TextContentBlock) for block in prompt):
            raise RequestError.invalid_params(
                {"prompt": "MIRA ACP currently supports text content only"}
            )
        text = "".join(block.text for block in prompt)
        try:
            with self.frontend.bind(session_id):
                session = await self._session(session_id)
                await session.prompt(text)
            return PromptResponse(stop_reason="end_turn")
        except ReplyInChat:
            return PromptResponse(stop_reason="end_turn")
        except (InteractionCancelled, asyncio.CancelledError):
            return PromptResponse(stop_reason="cancelled")
        finally:
            await self.frontend.flush(session_id)

    async def set_session_mode(
        self,
        session_id: str,
        mode_id: str,
        **kwargs: Any,
    ) -> SetSessionModeResponse:
        del kwargs
        if mode_id not in {"act", "plan"}:
            raise RequestError.invalid_params({"modeId": f"unknown MIRA mode: {mode_id}"})
        with self.frontend.bind(session_id):
            session = await self._session(session_id)
            await session.set_mode(mode_id)
        self._session_modes[session_id] = mode_id
        return SetSessionModeResponse()

    async def set_config_option(
        self,
        config_id: str,
        session_id: str,
        value: str | bool,
        **kwargs: Any,
    ) -> SetSessionConfigOptionResponse:
        del kwargs
        if config_id != "mode" or not isinstance(value, str):
            raise RequestError.invalid_params(
                {config_id: "MIRA ACP exposes only the ACT/PLAN mode selector"}
            )
        await self.set_session_mode(session_id, value)
        return SetSessionConfigOptionResponse(config_options=self._config_options(session_id))

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        del kwargs
        session = self._mira_sessions.get(session_id)
        if session is not None:
            await session.cancel()

    async def shutdown(self) -> None:
        """Delegate resource cleanup to MIRA, then stop ordered senders."""
        await asyncio.gather(
            *(application.shutdown() for application in self._applications.values()),
            return_exceptions=True,
        )
        await self.frontend.shutdown()

    async def _session(self, session_id: str) -> MiraSession:
        session = self._mira_sessions.get(session_id)
        if session is not None:
            return session
        workspace = self._session_workspaces.get(session_id)
        if workspace is None:
            raise RequestError.resource_not_found(session_id)
        application = await self._application(workspace)
        session = await application.open_session(session_id=session_id)
        requested_mode = self._session_modes.get(session_id, "act")
        if requested_mode != "act":
            await session.set_mode(requested_mode)
        self._mira_sessions[session_id] = session
        return session

    async def _application(self, workspace: Path) -> MiraApplication:
        key = self._workspace_key(workspace)
        application = self._applications.get(key)
        if application is not None:
            return application
        lock = self._application_locks.setdefault(key, asyncio.Lock())
        async with lock:
            application = self._applications.get(key)
            if application is None:
                application = await MiraApplication.start(
                    workspace=workspace,
                    frontend=self.frontend,
                )
                self._applications[key] = application
        return application

    def _initialize_session(self, session_id: str, workspace: Path, mode: str) -> None:
        self._session_workspaces[session_id] = workspace
        self._session_modes[session_id] = mode

    def _mode_state(self, session_id: str) -> SessionModeState:
        return SessionModeState(
            current_mode_id=self._session_modes.get(session_id, "act"),
            available_modes=MIRA_MODE_OPTIONS,
        )

    def _config_options(self, session_id: str) -> list[SessionConfigOptionSelect]:
        return [
            SessionConfigOptionSelect(
                id="mode",
                name="Session Mode",
                description="Select MIRA's ACT or formal PLAN workflow.",
                category="mode",
                type="select",
                current_value=self._session_modes.get(session_id, "act"),
                options=[
                    SessionConfigSelectOption(
                        value=mode.id,
                        name=mode.name,
                        description=mode.description,
                    )
                    for mode in MIRA_MODE_OPTIONS
                ],
            )
        ]

    def _forget_session(self, session_id: str) -> None:
        self._mira_sessions.pop(session_id, None)
        self._session_workspaces.pop(session_id, None)
        self._session_modes.pop(session_id, None)

    @staticmethod
    def _workspace(cwd: str | Path) -> Path:
        return Path(cwd).expanduser().resolve()

    @staticmethod
    def _workspace_key(workspace: Path) -> str:
        return os.path.normcase(str(workspace.resolve()))

    @staticmethod
    def _reject_external_configuration(
        additional_directories: list[str] | None,
        mcp_servers: list[Any] | None,
    ) -> None:
        unsupported = {}
        if additional_directories:
            unsupported["additionalDirectories"] = "MIRA manages workspace directories"
        if mcp_servers:
            unsupported["mcpServers"] = "MIRA manages MCP configuration and trust"
        if unsupported:
            raise RequestError.invalid_params(unsupported)

__all__ = ["MiraAgent"]
