"""Configuration management for Notubiz Curator."""

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    """Project configuration model."""

    catalog: str = Field(..., description="Unity Catalog name")
    db_schema: str = Field(..., description="Schema name", alias="schema")
    volume: str = Field(..., description="Volume name")
    llm_endpoint: str = Field(..., description="AI Gateway chat model service name")
    embedding_endpoint: str = Field(
        ..., description="Embedding serving endpoint for ai_query, AI Search and model resources"
    )
    embedding_service: str = Field(..., description="AI Gateway embedding model service name for the OpenAI client")
    experiment_path: str = Field(..., description="MLflow experiment path for this environment")
    warehouse_id: str = Field(..., description="Warehouse ID")
    lakebase_instance: str | None = Field(None, description="Lakebase project ID")
    lakebase_enabled: bool = Field(False, description="Whether Lakebase is enabled for this environment")
    usage_policy_id: str | None = Field(None, description="Budget policy for serverless resources")
    genie_space_id: str | None = Field(None, description="Genie space ID for MCP integration")
    system_prompt: str = Field(
        default=(
            "You help users find and understand public documents from Wetterskip Fryslân. "
            "Use document search for document contents, Genie tools for structured data and counts, "
            "and web search when current external information is needed for fact-checking. "
            "For document search, use concise search terms and request no more than 20 results. "
            "After reviewing relevant results, provide a clear, non-empty answer rather than repeatedly searching. "
            "Cite actual source URLs where available, distinguish document evidence from web sources, "
            "and be explicit when the results are insufficient or cannot establish completeness. "
            "Do not infer that no documents exist from an inconclusive search. Do not invent facts or citations."
        ),
        description="System prompt for the agent",
    )

    model_config = {"populate_by_name": True}

    @classmethod
    def from_yaml(cls, config_path: str, env: str = "dev") -> "ProjectConfig":
        """Load configuration from YAML file.

        Args:
            config_path: Path to the YAML configuration file
            env: Environment name (dev, acc, prd)

        Returns:
            ProjectConfig instance

        """
        if env not in ["prd", "acc", "dev"]:
            raise ValueError(f"Invalid environment: {env}. Expected 'prd', 'acc', or 'dev'")

        with open(config_path, encoding="utf-8") as config_file:
            config_data = yaml.safe_load(config_file)

        if env not in config_data:
            raise ValueError(f"Environment '{env}' not found in config file")

        environment_config = dict(config_data[env])
        environment_config.setdefault("usage_policy_id", config_data.get("usage_policy_id"))
        return cls(**environment_config)

    @property
    def schema(self) -> str:
        """Alias for db_schema for backward compatibility."""
        return self.db_schema

    @property
    def full_schema_name(self) -> str:
        """Get fully qualified schema name."""
        return f"{self.catalog}.{self.db_schema}"

    @property
    def full_volume_path(self) -> str:
        """Get fully qualified volume path."""
        return f"{self.catalog}.{self.schema}.{self.volume}"


def load_config(config_path: str = "project_config.yml", env: str = "dev") -> ProjectConfig:
    """Load project configuration.

    Args:
        config_path: Path to configuration file
        env: Environment name

    Returns:
        ProjectConfig instance

    """
    if not Path(config_path).is_absolute():
        current = Path.cwd()
        for _ in range(3):
            candidate = current / config_path
            if candidate.exists():
                config_path = str(candidate)
                break
            current = current.parent

    return ProjectConfig.from_yaml(config_path, env)


def get_env() -> str:
    """Get current environment from environment variable.

    Returns:
        Environment name (dev, acc, or prd)

    """
    return os.getenv("ENV", "dev")
