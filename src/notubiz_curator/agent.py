"""Notubiz agent implementing the MLflow ResponsesAgent interface."""

import asyncio
import json
import os
from collections.abc import Generator
from time import perf_counter
from typing import Any, Protocol, cast
from uuid import uuid4

import backoff
import mlflow
import nest_asyncio
import openai
from databricks.sdk import WorkspaceClient
from loguru import logger
from mlflow.entities import SpanType
from mlflow.pyfunc.model import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from notubiz_curator.config import ProjectConfig
from notubiz_curator.mcp import ToolInfo, create_mcp_tools
from notubiz_curator.memory import LakebaseMemory
from notubiz_curator.tools import create_search_documents_tool
from notubiz_curator.vector_search import LakebasePgVectorManager

_DEFAULT_MAX_ITERATIONS = 15


class MessageStore(Protocol):
    """Append and retrieve JSON-compatible messages for a session."""

    def load_messages(self, session_id: str) -> list[dict[str, Any]]:
        """Load a session's messages in order."""
        ...

    def save_messages(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        """Append new messages to a session."""
        ...


class NotubizAgent(ResponsesAgent):
    """ResponsesAgent for Notubiz documents, Genie, web search, and Lakebase memory."""

    def __init__(
        self,
        workspace: WorkspaceClient,
        llm_endpoint: str,
        system_prompt: str,
        tools: list[ToolInfo] | None = None,
        memory: MessageStore | None = None,
    ) -> None:
        """Initialize the Notubiz Agent."""
        nest_asyncio.apply()

        self.workspace_client = workspace
        self.llm_endpoint = llm_endpoint
        self.system_prompt = system_prompt
        self.memory = memory

        self._tools_dict = {tool.name: tool for tool in tools or []}
        if len(self._tools_dict) != len(tools or []):
            raise ValueError("Tool names must be unique")

        base_url = f"{workspace.config.host.rstrip('/')}/ai-gateway/mlflow/v1"

        def access_token() -> str:
            authorization = workspace.config.authenticate().get("Authorization", "")
            if not authorization.startswith("Bearer "):
                raise RuntimeError("Databricks authentication did not provide a bearer token")
            return authorization[len("Bearer ") :]

        self.model_serving_client = OpenAI(api_key=access_token, base_url=base_url)

    @classmethod
    def from_config(
        cls,
        config: ProjectConfig,
        workspace: WorkspaceClient | None = None,
        *,
        genie_space_id: str | None = None,
    ) -> "NotubizAgent":
        """Create a NotubizAgent synchronously from ProjectConfig."""
        nest_asyncio.apply()
        return asyncio.run(
            create_notubiz_agent(
                config,
                workspace or WorkspaceClient(),
                genie_space_id=genie_space_id,
            )
        )

    def get_tool_specs(self) -> list[dict[str, Any]]:
        """Return tool specifications for the Unified Responses API."""
        return [
            {
                "type": "function",
                "name": tool_info.spec["function"]["name"],
                "description": tool_info.spec["function"].get("description", ""),
                "parameters": tool_info.spec["function"].get("parameters", {"type": "object", "properties": {}}),
            }
            for tool_info in self._tools_dict.values()
        ]

    @mlflow.trace(span_type=SpanType.TOOL)
    def execute_tool(self, tool_name: str, args: dict[str, Any]) -> object:
        """Execute the specified tool with the given arguments."""
        if tool_name not in self._tools_dict:
            logger.warning("Model requested an unavailable tool: name={}", tool_name)
            return "Error: this tool is unavailable. Use available tools or answer from existing results."
        started = perf_counter()
        result = self._tools_dict[tool_name].exec_fn(**args)
        logger.info("Tool completed: name={}, seconds={:.2f}", tool_name, perf_counter() - started)
        return result

    @backoff.on_exception(backoff.expo, openai.RateLimitError)
    @mlflow.trace(span_type=SpanType.LLM)
    def call_llm(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Call the serving endpoint and return one Responses output item."""
        response = self.model_serving_client.responses.create(
            model=self.llm_endpoint,
            input=cast(Any, messages),
            tools=cast(Any, self.get_tool_specs()),
            max_output_tokens=1024,
        )
        for item in response.output:
            if getattr(item, "type", None) != "function_call":
                continue
            call_id = getattr(item, "call_id", None) or getattr(item, "id", str(uuid4()))
            return self.create_function_call_item(
                id=call_id,
                call_id=call_id,
                name=getattr(item, "name", ""),
                arguments=getattr(item, "arguments", "{}"),
            )

        return self.create_text_output_item(
            text=response.output_text or "",
            id=response.id or str(uuid4()),
        )

    def handle_tool_call(self, tool_call: dict[str, Any], messages: list[dict[str, Any]]) -> ResponsesAgentStreamEvent:
        """Execute a tool call, append its output to messages, and return a stream event."""
        args = json.loads(tool_call["arguments"])
        if not isinstance(args, dict):
            raise ValueError(f"Tool arguments for {tool_call['name']} must be a JSON object")
        result = str(self.execute_tool(tool_name=tool_call["name"], args=args))

        tool_call_output = self.create_function_call_output_item(tool_call["call_id"], result)
        messages.append(tool_call_output)
        return ResponsesAgentStreamEvent.model_validate({"type": "response.output_item.done", "item": tool_call_output})

    @mlflow.trace(span_type=SpanType.MEMORY, name="memory_load")
    def load_memory(self, session_id: str) -> list[dict[str, Any]]:
        """Load previous messages from Lakebase memory."""
        if self.memory:
            return self.memory.load_messages(session_id)
        return []

    @mlflow.trace(span_type=SpanType.MEMORY, name="memory_save")
    def save_memory(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        """Save new messages to Lakebase memory."""
        if self.memory:
            self.memory.save_messages(session_id, messages)

    def _extract_output_items(self, events: list[ResponsesAgentStreamEvent]) -> list[dict[str, Any]]:
        """Extract all Responses output items from completed stream events."""
        items: list[dict[str, Any]] = []
        for event in events:
            if event.type != "response.output_item.done":
                continue
            raw_item = getattr(event, "item", None)
            if raw_item is None:
                continue
            item = raw_item if isinstance(raw_item, dict) else raw_item.model_dump()
            items.append(item)
        return items

    def _run_tool_loop(
        self,
        messages: list[dict[str, Any]],
        max_iter: int = _DEFAULT_MAX_ITERATIONS,
    ) -> list[ResponsesAgentStreamEvent]:
        """Run the LLM <-> tool loop until the model stops or max_iter."""
        events: list[ResponsesAgentStreamEvent] = []
        for _ in range(max_iter):
            last_msg = messages[-1]
            if last_msg.get("type") == "message" and last_msg.get("role") == "assistant":
                return events
            if last_msg.get("type") == "function_call":
                tool_event = self.handle_tool_call(last_msg, messages)
                events.append(tool_event)
            else:
                llm_output = self.call_llm(messages)
                messages.append(llm_output)
                events.append(
                    ResponsesAgentStreamEvent.model_validate({"type": "response.output_item.done", "item": llm_output})
                )
        else:
            events.append(
                ResponsesAgentStreamEvent.model_validate(
                    {
                        "type": "response.output_item.done",
                        "item": self.create_text_output_item("Max iterations reached. Stopping.", str(uuid4())),
                    }
                )
            )
        return events

    @mlflow.trace(span_type=SpanType.CHAIN)
    def call_and_run_tools(
        self,
        request_input: list[dict[str, Any]],
        previous_messages: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
        session_id: str | None = None,
        max_iter: int = _DEFAULT_MAX_ITERATIONS,
    ) -> list[ResponsesAgentStreamEvent]:
        """Run a full conversation turn and persist new messages to memory when configured."""
        messages: list[dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        if previous_messages:
            messages.extend(previous_messages)
        messages.extend(request_input)

        mlflow.update_current_trace(
            tags={
                "git_sha": os.getenv("GIT_SHA", "local"),
                "deployment_name": os.getenv(
                    "DATABRICKS_APP_NAME",
                    os.getenv("MODEL_SERVING_ENDPOINT_NAME", "local"),
                ),
                "model_version": os.getenv("MODEL_VERSION", "local"),
            },
            metadata=({"mlflow.trace.session": session_id} if session_id else {}),
            client_request_id=request_id or str(uuid4()),
        )

        events = self._run_tool_loop(messages, max_iter=max_iter)

        if session_id and self.memory:
            self.save_memory(
                session_id,
                request_input + self._extract_output_items(events),
            )
        return events

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """Handle non-streaming Responses API requests."""
        events = list(self.predict_stream(request))
        return ResponsesAgentResponse(
            output=cast(Any, self._extract_output_items(events)),
            custom_outputs=request.custom_inputs,
        )

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict_stream(self, request: ResponsesAgentRequest) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Stream Responses API events for the incoming request."""
        custom = request.custom_inputs or {}
        session_id = custom.get("session_id")
        request_id = custom.get("request_id")

        if session_id is not None and self.memory is None:
            raise ValueError("A message store is required for a session")
        if session_id is not None and (not isinstance(session_id, str) or not session_id.strip()):
            raise ValueError("session_id must be a non-empty string")
        if request_id is not None and (not isinstance(request_id, str) or not request_id.strip()):
            raise ValueError("request_id must be a non-empty string")

        previous_messages = self.load_memory(session_id) if session_id and self.memory else []

        request_input = [item.model_dump() for item in request.input]
        events = self.call_and_run_tools(
            request_input=request_input,
            previous_messages=previous_messages,
            request_id=request_id,
            session_id=session_id,
        )
        yield from events

    def generate_message(
        self,
        message: str,
        max_iterations: int = _DEFAULT_MAX_ITERATIONS,
        *,
        session_id: str | None = None,
        history: list[ChatCompletionMessageParam] | None = None,
        request_id: str | None = None,
    ) -> str:
        """Send a text message using the ResponsesAgent implementation."""
        if session_id is not None and history is not None:
            raise ValueError("Provide either session_id or history, not both")
        if isinstance(max_iterations, bool) or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")

        request_messages: list[dict[str, Any]] = [dict(item) for item in history] if history else []
        request_messages.append({"role": "user", "content": message})
        custom_inputs: dict[str, Any] = {}
        if session_id is not None:
            custom_inputs["session_id"] = session_id
        if request_id is not None:
            custom_inputs["request_id"] = request_id

        response = self.predict(
            ResponsesAgentRequest.model_validate(
                {
                    "input": request_messages,
                    "custom_inputs": custom_inputs or None,
                }
            )
        )
        parts: list[str] = []
        for item in response.output:
            data = item.model_dump()
            if data.get("type") == "message" and data.get("role") == "assistant":
                for content_item in data.get("content", []):
                    if isinstance(content_item, dict) and isinstance(content_item.get("text"), str):
                        parts.append(content_item["text"])
        return "\n".join(parts).strip()


async def create_notubiz_agent(
    config: ProjectConfig,
    workspace: WorkspaceClient,
    *,
    genie_space_id: str | None = None,
) -> NotubizAgent:
    """Discover the managed tools and assemble the agent and Lakebase memory."""
    space_id = genie_space_id or config.genie_space_id
    if not space_id:
        raise ValueError("genie_space_id is required")
    if not config.lakebase_enabled or not config.lakebase_instance:
        raise ValueError("Lakebase must be enabled and configured")
    os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = config.warehouse_id
    mlflow.set_experiment(config.experiment_path)
    logger.info("Using MLflow experiment: {}", config.experiment_path)
    host = workspace.config.host.rstrip("/")
    urls = [
        f"{host}/api/2.0/mcp/genie/{space_id}",
        f"{host}/ai-gateway/mcp-services/system.ai.web_search",
    ]
    tools = [create_search_documents_tool(LakebasePgVectorManager(config, workspace))]
    logger.info("Discovering Genie and web search MCP tools")
    tools.extend(await create_mcp_tools(workspace, urls))
    logger.info("Registered agent tools: {}", [tool.name for tool in tools])
    memory = LakebaseMemory(config.lakebase_instance, workspace)
    logger.info("Initializing Lakebase session storage")
    memory.ensure_schema()
    logger.info("Lakebase session storage ready")
    return NotubizAgent(workspace, config.llm_endpoint, config.system_prompt, tools, memory)
