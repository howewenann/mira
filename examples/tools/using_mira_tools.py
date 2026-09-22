"""Ordinary LangChain tools may consume MIRA's trusted runtime context."""

from langchain.tools import ToolRuntime
from langchain_core.tools import tool


@tool
async def convert_pdf(
    source: str,
    destination: str,
    runtime: ToolRuntime,
) -> str:
    """Convert a PDF to Markdown and save the result."""
    convert = runtime.context.tools["mcp__converter__pdf_to_markdown"]
    write = runtime.context.tools["write_file"]
    markdown = await convert.ainvoke({"path": source})
    await write.ainvoke({"file_path": destination, "content": markdown})
    return destination


@tool
async def research_topic(topic: str, runtime: ToolRuntime) -> str:
    """Ask the configured researcher subagent to investigate a topic."""
    researcher = runtime.context.agents["researcher"]
    return str(await researcher.ainvoke(f"Research this topic: {topic}"))
