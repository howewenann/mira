"""Native FastMCP OAuth selection for remote HTTP servers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import httpx2 as httpx
from fastmcp.client.auth import OAuth
from mcp.client.auth.utils import (
    build_oauth_authorization_server_metadata_discovery_urls,
    create_oauth_metadata_request,
    handle_auth_metadata_response,
)

TokenEndpointAuthMethod = Literal["client_secret_basic", "client_secret_post", "none"]
_DISCOVERY_TIMEOUT_SECONDS = 5.0


def select_token_endpoint_auth_method(
    supported_methods: Sequence[str] | None,
) -> TokenEndpointAuthMethod | None:
    """Choose the strongest supported DCR token authentication method."""
    if supported_methods is None:
        return None
    methods = set(supported_methods)
    if "client_secret_basic" in methods:
        return "client_secret_basic"
    if "client_secret_post" in methods:
        return "client_secret_post"
    return "none"


async def discover_token_endpoint_auth_method(
    server_url: str,
) -> TokenEndpointAuthMethod | None:
    """Discover public authorization-server metadata for one HTTP MCP URL."""
    urls = build_oauth_authorization_server_metadata_discovery_urls(None, server_url)
    async with httpx.AsyncClient(
        timeout=_DISCOVERY_TIMEOUT_SECONDS,
        follow_redirects=False,
    ) as client:
        for url in urls:
            try:
                response = await client.send(create_oauth_metadata_request(url))
            except httpx.HTTPError:
                continue
            should_continue, metadata = await handle_auth_metadata_response(response)
            if metadata is not None:
                return select_token_endpoint_auth_method(
                    metadata.token_endpoint_auth_methods_supported
                )
            if not should_continue:
                break
    return None


async def create_http_oauth(server_url: str) -> OAuth:
    """Build FastMCP OAuth with discovered DCR token authentication metadata."""
    method = await discover_token_endpoint_auth_method(server_url)
    client_metadata = {"token_endpoint_auth_method": method} if method is not None else None
    return OAuth(
        server_url,
        additional_client_metadata=client_metadata,
    )


__all__ = [
    "TokenEndpointAuthMethod",
    "create_http_oauth",
    "discover_token_endpoint_auth_method",
    "select_token_endpoint_auth_method",
]
