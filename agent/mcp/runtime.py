"""Task-owned lifecycle for one persistent MCP client session."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(slots=True)
class _Command:
    operation: Literal["start", "stop", "shutdown"]
    result: asyncio.Future[Any]


class MCPServerRuntime:
    """Enter and exit one server session from the same long-lived task."""

    def __init__(
        self,
        name: str,
        context_factory: Callable[[], AbstractAsyncContextManager[Any]],
    ) -> None:
        self.name = name
        self._context_factory = context_factory
        self._commands: asyncio.Queue[_Command] = asyncio.Queue()
        self._owner: asyncio.Task[None] | None = None
        self._session: Any = None

    @property
    def open(self) -> bool:
        return self._session is not None

    async def start(self) -> Any:
        return await self._request("start")

    async def stop(self) -> None:
        await self._request("stop")

    async def shutdown(self) -> None:
        if self._owner is None:
            return
        await self._request("shutdown")
        owner, self._owner = self._owner, None
        await _wait_without_cancelling(owner)

    async def _request(self, operation: Literal["start", "stop", "shutdown"]) -> Any:
        if self._owner is None:
            self._owner = asyncio.create_task(self._run(), name=f"mcp-runtime-{self.name}")
        result = asyncio.get_running_loop().create_future()
        await self._commands.put(_Command(operation, result))
        return await _wait_without_cancelling(result)

    async def _run(self) -> None:
        stack: AsyncExitStack | None = None
        try:
            while True:
                command = await self._commands.get()
                try:
                    if command.operation == "start":
                        if stack is None:
                            candidate = AsyncExitStack()
                            try:
                                self._session = await candidate.enter_async_context(self._context_factory())
                            except BaseException:
                                await candidate.aclose()
                                raise
                            stack = candidate
                        _set_result(command.result, self._session)
                        continue

                    self._session = None
                    if stack is not None:
                        closing, stack = stack, None
                        await closing.aclose()
                    _set_result(command.result, None)
                    if command.operation == "shutdown":
                        return
                except BaseException as error:
                    self._session = None
                    _set_exception(command.result, error)
                    if command.operation == "shutdown":
                        return
        finally:
            self._session = None
            if stack is not None:
                try:
                    await stack.aclose()
                except BaseException:
                    pass
            while not self._commands.empty():
                command = self._commands.get_nowait()
                _set_exception(command.result, RuntimeError(f"MCP runtime {self.name} stopped"))


async def _wait_without_cancelling(awaitable: asyncio.Future[Any] | asyncio.Task[Any]) -> Any:
    """Finish an owner command even if its requesting UI worker is cancelled."""
    while True:
        try:
            return await asyncio.shield(awaitable)
        except asyncio.CancelledError:
            if awaitable.done():
                return awaitable.result()


def _set_result(future: asyncio.Future[Any], value: Any) -> None:
    if not future.done():
        future.set_result(value)


def _set_exception(future: asyncio.Future[Any], error: BaseException) -> None:
    if not future.done():
        future.set_exception(error)


__all__ = ["MCPServerRuntime"]
