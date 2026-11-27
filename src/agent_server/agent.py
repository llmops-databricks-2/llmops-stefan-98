"""MLflow AgentServer handlers for the Databricks App deployment."""

import os
from collections.abc import Generator

from agent_server.auth import setup_auth
from notubiz_curator.config import ProjectConfig, get_env

cfg = ProjectConfig.from_yaml("project_config.yml", env=get_env())
setup_auth(cfg)

if os.getenv("GENIE_SPACE_ID"):
    cfg.genie_space_id = os.environ["GENIE_SPACE_ID"]

import mlflow  # noqa: E402
from mlflow.genai.agent_server import invoke, stream  # noqa: E402
from mlflow.types.responses import (  # noqa: E402
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from notubiz_curator.agent import NotubizAgent  # noqa: E402

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(cfg.experiment_path)

agent = NotubizAgent.from_config(cfg)


@invoke()
def invoke_handler(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    """Handle non-streaming Responses API requests."""
    return agent.predict(request)


@stream()
def stream_handler(request: ResponsesAgentRequest) -> Generator[ResponsesAgentStreamEvent, None, None]:
    """Stream Responses API events from the existing agent implementation."""
    yield from agent.predict_stream(request)
