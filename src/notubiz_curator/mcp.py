"""MCP (Model Context Protocol) integration utilities."""

from collections.abc import Callable
from typing import Any

from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksMCPClient
from pydantic import BaseModel


class ToolInfo(BaseModel):
    """Tool information for agent integration.

    Attributes:
        name: Tool name
            spec: JSON description of the tool (OpenAI Chat Completions format)
        exec_fn: Function that implements the tool logic

    """

    name: str
    spec: dict[str, Any]
    exec_fn: Callable[..., str]

    model_config = {"arbitrary_types_allowed": True}


def create_managed_exec_fn(server_url: str, tool_name: str, w: WorkspaceClient) -> Callable[..., str]:
    """Create an execution function for an MCP tool.

    Args:
        server_url: MCP server URL
        tool_name: Name of the tool
        w: Databricks workspace client

    Returns:
        Callable that executes the tool

    """

    def exec_fn(**kwargs: object) -> str:
        client = DatabricksMCPClient(server_url=server_url, workspace_client=w)
        response = client.call_tool(tool_name, kwargs)
        text = "\n".join(block.text for block in response.content if block.type == "text")
        if response.isError:
            raise RuntimeError(f"MCP tool {tool_name} failed: {text or 'no error details'}")
        if len(response.content) != sum(block.type == "text" for block in response.content):
            raise RuntimeError(f"MCP tool {tool_name} returned unsupported non-text content")
        return text

    return exec_fn


async def create_mcp_tools(w: WorkspaceClient, url_list: list[str]) -> list[ToolInfo]:
    """Create tools from MCP servers.

    Args:
        w: Databricks workspace client
        url_list: List of MCP server URLs

    Returns:
        List of ToolInfo objects

    """
    tools = []
    names = set()
    for server_url in url_list:
        mcp_client = DatabricksMCPClient(server_url=server_url, workspace_client=w)
        mcp_tools = mcp_client.list_tools()
        if not mcp_tools:
            raise RuntimeError(f"No tools discovered at MCP server {server_url}")
        for mcp_tool in mcp_tools:
            if mcp_tool.name in names:
                raise ValueError(f"Duplicate MCP tool name: {mcp_tool.name}")
            names.add(mcp_tool.name)
            input_schema = mcp_tool.inputSchema.copy() if mcp_tool.inputSchema else {}
            tool_spec = {
                "type": "function",
                "function": {
                    "name": mcp_tool.name,
                    "parameters": input_schema,
                    "description": mcp_tool.description or f"Tool: {mcp_tool.name}",
                },
            }
            exec_fn = create_managed_exec_fn(server_url, mcp_tool.name, w)
            tools.append(ToolInfo(name=mcp_tool.name, spec=tool_spec, exec_fn=exec_fn))
    return tools
