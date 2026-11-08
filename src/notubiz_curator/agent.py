"""Minimal agent using an OpenAI-compatible Databricks endpoint."""

import json
from time import perf_counter
from typing import Any, Protocol, cast

from databricks.sdk import WorkspaceClient
from loguru import logger
from openai import OpenAI, omit
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion_message_function_tool_call_param import (
    ChatCompletionMessageFunctionToolCallParam,
)

from notubiz_curator.config import ProjectConfig
from notubiz_curator.mcp import ToolInfo, create_mcp_tools
from notubiz_curator.memory import LakebaseMemory
from notubiz_curator.tools import create_search_documents_tool
from notubiz_curator.vector_search import LakebasePgVectorManager

_DEFAULT_MAX_ITERATIONS = 10
_MAX_WRAPUP_RETRIES = 2


class MessageStore(Protocol):
    """Append and retrieve JSON-compatible messages for a session."""

    def load_messages(self, session_id: str) -> list[dict[str, Any]]:
        """Load a session's messages in order."""
        ...

    def save_messages(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        """Append new messages to a session."""
        ...


class NotubizAgent:
    """Call a chat model and execute its requested function tools."""

    def __init__(
        self,
        workspace: WorkspaceClient,
        llm_endpoint: str,
        system_prompt: str,
        tools: list[ToolInfo] | None = None,
        memory: MessageStore | None = None,
    ) -> None:
        """Create an agent using the supplied workspace.

        Args:
            workspace: Authenticated Databricks workspace client.
            llm_endpoint: AI Gateway model service name.
            system_prompt: Instructions sent with each conversation.
            tools: Optional tools the model can call.
            memory: Optional message store for session-based conversations.

        """
        self.llm_endpoint = llm_endpoint
        self.system_prompt = system_prompt
        self.memory = memory

        self.tools = {tool.name: tool for tool in tools or []}
        if len(self.tools) != len(tools or []):
            raise ValueError("Tool names must be unique")
        self.tool_specs = [tool.spec for tool in self.tools.values()]

        base_url = f"{workspace.config.host.rstrip('/')}/ai-gateway/mlflow/v1"

        def access_token() -> str:
            authorization = workspace.config.authenticate().get("Authorization", "")
            if not authorization.startswith("Bearer "):
                raise RuntimeError("Databricks authentication did not provide a bearer token")
            return authorization[len("Bearer ") :]

        self.client = OpenAI(api_key=access_token, base_url=base_url)

    def generate_message(
        self,
        message: str,
        max_iterations: int = _DEFAULT_MAX_ITERATIONS,
        *,
        session_id: str | None = None,
        history: list[ChatCompletionMessageParam] | None = None,
    ) -> str:
        """Send a message and execute requested tools until the model responds.

        Pass a session ID to load its history from the configured memory and
        save each complete turn atomically after the model produces an answer.
        Alternatively, pass history directly to bypass memory entirely. History
        should not contain the system prompt. Passing neither starts a one-off
        conversation; passing both is not allowed.

        Args:
            message: User request to send to the model.
            max_iterations: Maximum model calls in the agent loop, including the wrap-up.
                An empty wrap-up can be retried twice without tools.
            session_id: Session to load and append; requires configured memory.
            history: Prior messages to use without reading or writing memory.

        Returns:
            The model's final text response.

        Raises:
            ValueError: If the session and history conflict, the store is absent,
                or the model requests an unsupported tool call type.
            RuntimeError: If the model provides no text or exhausts the limit.

        """
        if session_id is not None and history is not None:
            raise ValueError("Provide either session_id or history, not both")
        if session_id is not None and self.memory is None:
            raise ValueError("A message store is required for a session")
        if session_id is not None and not session_id.strip():
            raise ValueError("session_id must not be empty")
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or not 1 <= max_iterations <= 10:
            raise ValueError("max_iterations must be an integer between 1 and 10")

        logger.info("Starting agent turn: max_iterations={}, session_memory={}", max_iterations, session_id is not None)
        if session_id is not None and self.memory is not None:
            logger.info("Loading session history from Lakebase")
            previous = cast(list[ChatCompletionMessageParam], self.memory.load_messages(session_id))
            logger.info("Loaded {} session messages", len(previous))
        else:
            previous = list(history) if history is not None else []

        messages: list[ChatCompletionMessageParam] = [{"role": "system", "content": self.system_prompt}, *previous]
        user_message: ChatCompletionMessageParam = {"role": "user", "content": message}
        messages.append(user_message)
        new_messages: list[ChatCompletionMessageParam] = [user_message]
        empty_answer_retries = 0

        for iteration in range(max_iterations + _MAX_WRAPUP_RETRIES):
            final_call = iteration >= max_iterations - 1
            if iteration == max_iterations - 1:
                logger.info("Starting final wrap-up without tools")
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "You have spent several turns working on this task. No further tool calls are available. "
                            "Please provide your best final answer now based on what you have learned so far. "
                            "Cite sources where possible and explicitly state any limitations or missing information."
                        ),
                    }
                )
            elif iteration >= max_iterations:
                logger.warning(
                    "Retrying empty wrap-up without tools: attempt {}/{}",
                    iteration - max_iterations + 1,
                    _MAX_WRAPUP_RETRIES,
                )
                if iteration == max_iterations:
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "The previous wrap-up was empty. Write a non-empty final answer using only the "
                                "information available, and state clearly if it is insufficient or incomplete."
                            ),
                        }
                    )
            logger.info(
                "Model call {}/{}: endpoint={}, messages={}, tools_enabled={}, empty_retries={}",
                iteration + 1,
                max_iterations if iteration < max_iterations else max_iterations + _MAX_WRAPUP_RETRIES,
                self.llm_endpoint,
                len(messages),
                bool(self.tool_specs and not final_call),
                empty_answer_retries,
            )
            started = perf_counter()
            response = self.client.chat.completions.create(
                model=self.llm_endpoint,
                messages=messages,
                tools=self.tool_specs if self.tool_specs and not final_call else omit,  # type: ignore
                reasoning_effort="low" if iteration >= max_iterations else omit,
            )
            choice = response.choices[0]
            assistant_message = choice.message
            usage = response.usage
            logger.info(
                "Model response: iteration={}, seconds={:.2f}, model={}, finish_reason={}, "
                "answer_chars={}, tool_calls={}, prompt_tokens={}, completion_tokens={}, request_id={}",
                iteration + 1,
                perf_counter() - started,
                response.model,
                choice.finish_reason,
                len(assistant_message.content or ""),
                len(assistant_message.tool_calls or []),
                usage.prompt_tokens if usage else None,
                usage.completion_tokens if usage else None,
                response._request_id,
            )

            if final_call and assistant_message.tool_calls:
                raise RuntimeError("Model requested another tool after the allowed iterations for tool calls.")
            if not assistant_message.tool_calls:
                if not assistant_message.content or not assistant_message.content.strip():
                    if iteration == max_iterations + _MAX_WRAPUP_RETRIES - 1:
                        raise RuntimeError("Model returned an empty final answer within the allowed iterations.")
                    empty_answer_retries += 1
                    logger.warning("Model returned an empty final answer; retrying if calls remain.")
                    if empty_answer_retries == 1 and not final_call:
                        messages.append(
                            {
                                "role": "system",
                                "content": (
                                    "Your previous response was empty. Provide a non-empty answer to the user's "
                                    "question using the available information. If a Genie query is still running, "
                                    "poll it before answering. Cite sources where possible and state limitations."
                                ),
                            }
                        )
                    continue

                final_message: ChatCompletionMessageParam = {"role": "assistant", "content": assistant_message.content}
                new_messages.append(final_message)
                if session_id is not None and self.memory is not None:
                    logger.info("Saving {} new session messages to Lakebase", len(new_messages))
                    self.memory.save_messages(session_id, [dict(item) for item in new_messages])
                    logger.info("Session messages saved")
                return assistant_message.content

            calls: list[ChatCompletionMessageFunctionToolCallParam] = []

            for tool_call in assistant_message.tool_calls:
                if tool_call.type != "function":
                    raise ValueError(f"Unsupported tool call type: {tool_call.type}")

                calls.append(
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                )
            tool_request: ChatCompletionMessageParam = {
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": calls,
            }
            messages.append(tool_request)
            new_messages.append(tool_request)

            for call in calls:
                name = call["function"]["name"]
                logger.info("Executing tool: name={}, iteration={}", name, iteration + 1)
                arguments: object = json.loads(call["function"]["arguments"])
                if not isinstance(arguments, dict):
                    raise ValueError(f"Tool arguments for {name} must be a JSON object")
                if name not in self.tools:
                    logger.warning("Model requested an unavailable tool: name={}", name)
                    result = "Error: this tool is unavailable. Use available tools or answer from existing results."
                else:
                    started = perf_counter()
                    result = self.tools[name].exec_fn(**arguments)
                    logger.info(
                        "Tool completed: name={}, seconds={:.2f}, result_chars={}",
                        name,
                        perf_counter() - started,
                        len(str(result)),
                    )
                tool_message: ChatCompletionMessageParam = {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": str(result),
                }
                messages.append(tool_message)
                new_messages.append(tool_message)

        raise RuntimeError("Model did not provide a final answer within the allowed iterations.")


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
