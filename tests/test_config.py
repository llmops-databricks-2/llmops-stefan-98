"""Test the project configuration for each deployment environment.

`ci.yml` runs every test under tests/ on each pull request and each push to main; add yours here.
"""

from pathlib import Path

import pytest

from notubiz_curator.config import ProjectConfig

PROJECT_CONFIG = Path(__file__).resolve().parents[1] / "project_config.yml"


@pytest.mark.parametrize("env", ["dev", "acc", "prd"])
def test_project_config_loads(env: str) -> None:
    """Load the provisioned settings without requiring optional lecture resources."""
    cfg = ProjectConfig.from_yaml(str(PROJECT_CONFIG), env)

    assert cfg.full_volume_path == f"llmops_{env}.stefan_98.notubiz_files"
    assert cfg.warehouse_id == "566c64409b1743cf"
    assert cfg.experiment_path == f"/llmops/stefan-98-{env}"
    assert cfg.embedding_endpoint == "databricks-gte-large-en"
    assert cfg.embedding_service == "course_ops.gateway.embed"
    assert cfg.lakebase_instance == "lb-llmops-stefan-98"
    assert cfg.lakebase_enabled is (env == "dev")
