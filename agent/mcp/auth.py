"""Standards-compliant OAuth support for remote HTTP MCP servers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import socket
import tempfile
import time
import webbrowser
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit, urlunsplit

import httpx
from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.client.auth.oauth2 import (
    build_oauth_authorization_server_metadata_discovery_urls,
    build_protected_resource_metadata_discovery_urls,
)
from mcp.shared.auth import (
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthMetadata,
    OAuthToken,
    ProtectedResourceMetadata,
)
from pydantic import AnyHttpUrl

from agent.mcp.models import MCPServerState

TOKEN_ROOT = Path.home() / ".mira" / "_state" / "mcp-tokens"
_TOKEN_FILE = "oauth.json"
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")
_BEARER_SECRET = re.compile(r"(?i)(\bBearer\s+)[^\s,;]+")
_NAMED_SECRET = re.compile(
    r"(?i)(access_token|refresh_token|client_secret|authorization_code|code)(\s*[=:]\s*)([^\s&,;]+)"
)
_URL_SECRET = re.compile(r"(?i)([?&](?:access_token|refresh_token|client_secret|code)=)[^&#\s]+")
_RESOURCE_METADATA = re.compile(r'(?i)\bresource_metadata\s*=\s*"([^"\r\n]+)"')


class OAuthLoginRequired(RuntimeError):
    """Raised when an OAuth flow would require explicit browser interaction."""


class FileTokenStorage(TokenStorage):
    """Small atomic JSON token store scoped to one normalized MCP identity."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.path = directory / _TOKEN_FILE
        self._lock = asyncio.Lock()

    async def get_tokens(self) -> OAuthToken | None:
        data = await self._read()
        raw = data.get("tokens")
        if not isinstance(raw, dict):
            return None
        try:
            token = OAuthToken.model_validate(raw)
            expires_at = data.get("expires_at")
            if isinstance(expires_at, (int, float)):
                token.expires_in = int(float(expires_at) - time.time())
            return token
        except (TypeError, ValueError):
            return None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        async with self._lock:
            data = await asyncio.to_thread(self._read_sync)
            data["tokens"] = tokens.model_dump(mode="json", exclude_none=True)
            if tokens.expires_in is None:
                data.pop("expires_at", None)
            else:
                data["expires_at"] = time.time() + int(tokens.expires_in)
            await asyncio.to_thread(self._write_sync, data)

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        data = await self._read()
        raw = data.get("client_info")
        if not isinstance(raw, dict):
            return None
        try:
            return OAuthClientInformationFull.model_validate(raw)
        except (TypeError, ValueError):
            return None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        async with self._lock:
            data = await asyncio.to_thread(self._read_sync)
            data["client_info"] = client_info.model_dump(mode="json", exclude_none=True)
            await asyncio.to_thread(self._write_sync, data)

    async def clear(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._clear_sync)

    async def _read(self) -> dict[str, Any]:
        async with self._lock:
            return await asyncio.to_thread(self._read_sync)

    def _read_sync(self) -> dict[str, Any]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def _write_sync(self, data: dict[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        _best_effort_mode(self.directory, 0o700)
        handle, temporary = tempfile.mkstemp(prefix=".oauth-", suffix=".tmp", dir=self.directory)
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(data, stream, sort_keys=True, separators=(",", ":"))
                stream.flush()
                os.fsync(stream.fileno())
            _best_effort_mode(Path(temporary), 0o600)
            os.replace(temporary, self.path)
            _best_effort_mode(self.path, 0o600)
        finally:
            try:
                Path(temporary).unlink(missing_ok=True)
            except OSError:
                pass

    def _clear_sync(self) -> None:
        try:
            shutil.rmtree(self.directory)
        except FileNotFoundError:
            pass


class MiraOAuthProvider(OAuthClientProvider):
    """Official MCP SDK provider with an explicit-interaction browser boundary."""

    def __init__(
        self,
        state: MCPServerState,
        *,
        interactive: bool,
        token_root: Path = TOKEN_ROOT,
        browser_opener: Callable[[str], bool] = webbrowser.open,
        callback_timeout: float = 300.0,
    ) -> None:
        self.interactive = interactive
        self.browser_opener = browser_opener
        self.callback_timeout = callback_timeout
        self.callback_host = "127.0.0.1"
        self.callback_port = _available_port(self.callback_host)
        self._callback_server: asyncio.AbstractServer | None = None
        self._callback_result: asyncio.Future[tuple[str, str | None]] | None = None
        redirect_uri = f"http://{self.callback_host}:{self.callback_port}/callback"
        self.storage = FileTokenStorage(server_token_directory(state, token_root=token_root))
        metadata = OAuthClientMetadata(
            client_name="MIRA",
            redirect_uris=[AnyHttpUrl(redirect_uri)],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="none",
        )
        super().__init__(
            server_url=str(state.config["url"]),
            client_metadata=metadata,
            storage=self.storage,
            redirect_handler=self._redirect,
            callback_handler=self._callback,
            timeout=callback_timeout,
        )

    async def _initialize(self) -> None:
        await super()._initialize()
        if self.context.current_tokens and self.context.current_tokens.expires_in is not None:
            self.context.update_token_expiry(self.context.current_tokens)

    async def _redirect(self, authorization_url: str) -> None:
        if not self.interactive:
            raise OAuthLoginRequired("Login required")
        loop = asyncio.get_running_loop()
        self._callback_result = loop.create_future()
        self._callback_server = await asyncio.start_server(
            self._receive_callback,
            self.callback_host,
            self.callback_port,
        )
        opened = await asyncio.to_thread(self.browser_opener, authorization_url)
        if not opened:
            await self._close_callback_server()
            raise OAuthLoginRequired("Could not open the browser")

    async def _callback(self) -> tuple[str, str | None]:
        if self._callback_result is None:
            raise OAuthLoginRequired("OAuth callback was not started")
        try:
            return await asyncio.wait_for(self._callback_result, timeout=self.callback_timeout)
        except TimeoutError as error:
            raise OAuthLoginRequired("Browser authorization timed out") from error
        finally:
            await self._close_callback_server()

    async def _receive_callback(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        result = self._callback_result
        try:
            request_line = (await asyncio.wait_for(reader.readline(), timeout=5.0)).decode("ascii", "replace")
            while True:
                line = await asyncio.wait_for(reader.readline(), timeout=5.0)
                if line in {b"\r\n", b"\n", b""}:
                    break
            parts = request_line.split(" ", 2)
            query = parse_qs(urlsplit(parts[1] if len(parts) > 1 else "").query)
            if query.get("error"):
                raise OAuthLoginRequired("Browser authorization was cancelled or denied")
            code = (query.get("code") or [""])[0]
            state = (query.get("state") or [None])[0]
            if not code:
                raise OAuthLoginRequired("Browser authorization returned no code")
            if result is not None and not result.done():
                result.set_result((code, state))
            body = b"Authentication complete. You may return to MIRA."
            status = b"200 OK"
        except BaseException as error:
            if result is not None and not result.done():
                result.set_exception(error)
            body = b"Authentication did not complete. Return to MIRA for details."
            status = b"400 Bad Request"
        writer.write(
            b"HTTP/1.1 " + status + b"\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: "
            + str(len(body)).encode("ascii") + b"\r\nConnection: close\r\n\r\n" + body
        )
        try:
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    async def _close_callback_server(self) -> None:
        if self._callback_server is not None:
            self._callback_server.close()
            await self._callback_server.wait_closed()
            self._callback_server = None


def server_token_directory(state: MCPServerState, *, token_root: Path = TOKEN_ROOT) -> Path:
    component = _SAFE_COMPONENT.sub("-", state.name).strip("-._") or "server"
    component = component[:40]
    name_digest = hashlib.sha256(state.name.encode("utf-8")).hexdigest()[:8]
    identity = normalized_server_url(str(state.config.get("url", "")))
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return token_root.expanduser() / f"{component}-{name_digest}-{digest}"


def has_persisted_login(state: MCPServerState, *, token_root: Path = TOKEN_ROOT) -> bool:
    return (server_token_directory(state, token_root=token_root) / _TOKEN_FILE).is_file()


async def forget_persisted_login(state: MCPServerState, *, token_root: Path = TOKEN_ROOT) -> None:
    await FileTokenStorage(server_token_directory(state, token_root=token_root)).clear()


def normalized_server_url(url: str) -> str:
    parsed = urlsplit(url.strip())
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower()
    port = parsed.port
    if port and not ((scheme == "https" and port == 443) or (scheme == "http" and port == 80)):
        host = f"{host}:{port}"
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit((scheme, host, path, parsed.query, ""))


def has_authorization_header(state: MCPServerState) -> bool:
    headers = state.config.get("headers", {})
    return isinstance(headers, dict) and any(str(name).casefold() == "authorization" for name in headers)


async def is_oauth_login_required(error: BaseException, state: MCPServerState) -> bool:
    """Narrowly classify an unauthenticated HTTP failure as MCP OAuth."""
    if state.transport != "http" or has_authorization_header(state):
        return False
    responses = list(_error_responses(error))
    denied = [response for response in responses if response.status_code in {401, 403}]
    if not denied:
        return False
    for response in denied:
        header = response.headers.get("www-authenticate", "")
        match = _RESOURCE_METADATA.search(header)
        if "bearer" in header.casefold() and match and _valid_metadata_url(match.group(1)):
            return True
    return await _discover_oauth_metadata(str(state.config.get("url", "")))


def is_known_oauth_failure(error: BaseException, state: MCPServerState) -> bool:
    if state.transport != "http" or has_authorization_header(state):
        return False
    return any(response.status_code in {401, 403} for response in _error_responses(error))


def sanitized_error(error: BaseException) -> str:
    text = " ".join(str(error).split())
    text = _BEARER_SECRET.sub(r"\1[redacted]", text)
    text = _NAMED_SECRET.sub(r"\1\2[redacted]", text)
    text = _URL_SECRET.sub(r"\1[redacted]", text)
    return f"{type(error).__name__}: {text}" if text else type(error).__name__


def _error_responses(error: BaseException) -> Iterator[httpx.Response]:
    seen: set[int] = set()
    pending: list[BaseException] = [error]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        response = getattr(current, "response", None)
        if isinstance(response, httpx.Response):
            yield response
        children = getattr(current, "exceptions", ())
        pending.extend(child for child in children if isinstance(child, BaseException))
        pending.extend(value for value in current.args if isinstance(value, BaseException))
        if isinstance(current.__cause__, BaseException):
            pending.append(current.__cause__)
        if isinstance(current.__context__, BaseException):
            pending.append(current.__context__)


async def _discover_oauth_metadata(server_url: str) -> bool:
    try:
        async with asyncio.timeout(5.0):
            async with httpx.AsyncClient(timeout=5.0, follow_redirects=False) as client:
                protected: ProtectedResourceMetadata | None = None
                for url in build_protected_resource_metadata_discovery_urls(None, server_url):
                    response = await client.get(url)
                    if response.status_code == 200:
                        protected = ProtectedResourceMetadata.model_validate(response.json())
                        if _same_origin(str(protected.resource), server_url):
                            break
                        protected = None
                if protected is None or not protected.authorization_servers:
                    return False
                authorization_server = str(protected.authorization_servers[0])
                for url in build_oauth_authorization_server_metadata_discovery_urls(authorization_server, server_url):
                    response = await client.get(url)
                    if response.status_code != 200:
                        continue
                    metadata = OAuthMetadata.model_validate(response.json())
                    if normalized_server_url(str(metadata.issuer)) == normalized_server_url(authorization_server):
                        return True
    except (TimeoutError, httpx.HTTPError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return False


def _valid_metadata_url(value: str) -> bool:
    parsed = urlsplit(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.hostname) and parsed.username is None


def _same_origin(left: str, right: str) -> bool:
    first, second = urlsplit(left), urlsplit(right)
    return (
        first.scheme.lower(),
        (first.hostname or "").lower(),
        _effective_port(first.scheme, first.port),
    ) == (
        second.scheme.lower(),
        (second.hostname or "").lower(),
        _effective_port(second.scheme, second.port),
    )


def _effective_port(scheme: str, port: int | None) -> int | None:
    if port is not None:
        return port
    return 443 if scheme.casefold() == "https" else 80 if scheme.casefold() == "http" else None


def _available_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((host, 0))
        return int(listener.getsockname()[1])


def _best_effort_mode(path: Path, mode: int) -> None:
    if os.name == "nt":
        return
    try:
        path.chmod(mode)
    except OSError:
        pass


__all__ = [
    "FileTokenStorage",
    "MiraOAuthProvider",
    "OAuthLoginRequired",
    "TOKEN_ROOT",
    "forget_persisted_login",
    "has_authorization_header",
    "has_persisted_login",
    "is_known_oauth_failure",
    "is_oauth_login_required",
    "normalized_server_url",
    "sanitized_error",
    "server_token_directory",
]
