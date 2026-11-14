"""Local protocol fixture with ordinary MCP capabilities and portable skills."""

from pathlib import Path
import sys

from fastmcp import FastMCP
from fastmcp.skills.provider import SkillsDirectoryProvider
from fastmcp.skills.extension import SkillsExtension

server = FastMCP("MIRA skills regression")
if sys.argv[2] != "no-skills":
    provider = SkillsDirectoryProvider(roots=Path(sys.argv[1]), reload=True)
    server.add_provider(provider)
    server.add_extension(SkillsExtension(providers=[provider]))


@server.tool
def echo(value: str) -> str:
    """Return a value."""
    return value


@server.prompt
def greeting(name: str) -> str:
    """Greet someone."""
    return f"Hello {name}"


@server.resource("fixture://reference")
def reference() -> str:
    """An ordinary attachable resource."""
    return "Reference text"


server.run()
