# Databricks notebook source
# /// script
# dependencies = [
#   "../..",
# ]
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Test the Notubiz agent
# MAGIC
# MAGIC Run the cells in order. Set the `genie_space_id` widget to the ID of the deployed Genie
# MAGIC space; `environment` defaults to `dev`. The job supplies these values and the path to
# MAGIC `project_config.yml` automatically. The package and configuration must be available.
# MAGIC
# MAGIC The synced Lakebase chunks and indexes, Genie space and both MCP services must already
# MAGIC be available. The workspace identity needs the necessary model, Lakebase, Genie, SQL
# MAGIC warehouse, table and web-search permissions. These tests invoke external services and
# MAGIC may incur costs; the Genie warehouse may start. A new session is stored in Lakebase.
# MAGIC If memory uses a separate service principal, set all three `LAKEBASE_SP_*` environment
# MAGIC variables securely before creating the agent; otherwise the workspace identity is used.

# COMMAND ----------

"""Interactive tests for the Notubiz agent and its tools."""

import asyncio
import inspect
from uuid import uuid4

import nest_asyncio
from databricks.sdk import WorkspaceClient
from loguru import logger

from notubiz_curator.agent import create_notubiz_agent
from notubiz_curator.config import get_env, load_config

nest_asyncio.apply()
logger.info("Agent source: {}", inspect.getfile(create_notubiz_agent))
dbutils.widgets.text("environment", get_env())  # noqa: F821
dbutils.widgets.text("genie_space_id", "")  # noqa: F821
dbutils.widgets.text("config_path", "project_config.yml")  # noqa: F821
env = dbutils.widgets.get("environment").strip()  # noqa: F821
config_path = dbutils.widgets.get("config_path").strip()  # noqa: F821
cfg = load_config(config_path=config_path, env=env)
genie_space_id = dbutils.widgets.get("genie_space_id").strip() or cfg.genie_space_id  # noqa: F821
if not genie_space_id:
    raise ValueError("Set the genie_space_id widget to the deployed Genie space ID.")

w = WorkspaceClient()
agent = asyncio.run(create_notubiz_agent(cfg, w, genie_space_id=genie_space_id))
session_id = f"notubiz-test-{uuid4()}"
logger.info("Environment: {}; session: {}", env, session_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Hybrid document search
# MAGIC
# MAGIC Inspect the sources and dates in the answer; hybrid search returns matching chunks,
# MAGIC not a guarantee that every relevant commitment has been found.

# COMMAND ----------

first_question = "Zet voor mij alle toezeggingen van het dagelijks bestuur over veenweide sinds 2024 op een rijtje."
logger.info("Starting question 1: hybrid document search")
answer = agent.generate_message(first_question, session_id=session_id)
logger.info("Hybrid search answer: {}", answer)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Genie: structured document data

# COMMAND ----------

logger.info("Starting question 2: Genie")
answer = agent.generate_message(
    "Gebruik Genie om te bepalen hoeveel documenten er per organisatie in de documenttabel staan.",
    session_id=session_id,
)
logger.info("Genie answer: {}", answer)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Web search: current external information
# MAGIC
# MAGIC This sends the question to web search. Check that the answer provides relevant web sources.

# COMMAND ----------

logger.info("Starting question 3: web search")
answer = agent.generate_message(
    "Zoek op het internet: wat is Notubiz?",
    session_id=session_id,
)
logger.info("Web search answer: {}", answer)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Session memory
# MAGIC
# MAGIC Ask the agent to recall the first question in the same session.

# COMMAND ----------

recall_question = "Wat was mijn eerste vraag in dit gesprek? Herhaal die vraag zo letterlijk mogelijk."
logger.info("Starting question 4: session memory")
answer = agent.generate_message(recall_question, session_id=session_id)
logger.info("Memory answer: {}", answer)
