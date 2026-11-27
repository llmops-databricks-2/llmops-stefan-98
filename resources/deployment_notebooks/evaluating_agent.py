# Databricks notebook source
# /// script
# dependencies = [
#   "../..",
# ]
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Evaluate the Notubiz agent
# MAGIC
# MAGIC Runs the reviewed cases from `eval_inputs.txt` against the deployed agent dependencies.

# COMMAND ----------

"""MLflow GenAI evaluation for the Notubiz agent."""

import asyncio

import nest_asyncio
from databricks.sdk import WorkspaceClient
from loguru import logger

from notubiz_curator.agent import create_notubiz_agent
from notubiz_curator.config import get_env, load_config
from notubiz_curator.evaluation import evaluate_agent

nest_asyncio.apply()
dbutils.widgets.text("environment", get_env())  # noqa: F821
dbutils.widgets.text("genie_space_id", "")  # noqa: F821
dbutils.widgets.text("config_path", "project_config.yml")  # noqa: F821
dbutils.widgets.text("eval_inputs_path", "eval_inputs.txt")  # noqa: F821

env = dbutils.widgets.get("environment").strip()  # noqa: F821
config_path = dbutils.widgets.get("config_path").strip()  # noqa: F821
eval_inputs_path = dbutils.widgets.get("eval_inputs_path").strip()  # noqa: F821
cfg = load_config(config_path=config_path, env=env)
genie_space_id = dbutils.widgets.get("genie_space_id").strip() or cfg.genie_space_id  # noqa: F821
if not genie_space_id:
    raise ValueError("Set the genie_space_id widget to the deployed Genie space ID.")

workspace = WorkspaceClient()
agent = asyncio.run(create_notubiz_agent(cfg, workspace, genie_space_id=genie_space_id))
results = evaluate_agent(agent, eval_inputs_path)
logger.info("Evaluation metrics: {}", results.metrics)
logger.info("Evaluation run ID: {}", results.run_id)
