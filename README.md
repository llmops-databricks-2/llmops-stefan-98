# Notubiz agent

A Databricks agent for questions about public Notubiz documents from Wetterskip Fryslân.
It helps board members, employees, elected representatives and citizens find and understand
information without manually searching through documents.

[NotuBiz](https://www.notubiz.nl/) provides a platform for public-sector decision-making,
including meeting agendas and documents. [OpenBesluitvorming](https://openbesluitvorming.nl/)
makes public decision-making information searchable across Dutch municipalities, provinces and
water authorities. This project retrieves Wetterskip Fryslân's NotuBiz documents through the
OpenBesluitvorming API.

The current solution processes **documents only**. Meetings and other data are not ingested as
separate sources, although meeting details may be present as document metadata.

## Notubiz agent

1. **Bootstrap — manual.** The `notubiz-bootstrap` job retrieves the available historical
   document snapshot and writes it to `notubiz_documents`. Running it again replaces the existing table.
2. **Ingestion — daily.** The `notubiz-ingestion` job retrieves the 50 most recent documents
   each day and adds new or changed documents to `notubiz_documents`.
3. **Chunks and embeddings — manual.** The `notubiz-chunks` job splits documents into text
   fragments and creates embeddings. The result is the `notubiz_chunks` table.
4. **Lakebase sync — manual.** The `notubiz-lakebase-sync` job synchronizes the chunks and
   embeddings to Lakebase and creates search indexes. Lakebase Search must be enabled.
5. **Agent.** The `NotubizAgent` uses these building blocks to answer questions:
   - **Document search:** local hybrid search in Lakebase using text and embeddings.
   - **Genie:** MCP tools for structured questions about document metadata and counts,
     using the existing SQL warehouse.
   - **Web search:** the `system.ai.web_search` MCP tool for additional information from the web.
   - **Memory:** Lakebase stores conversation history per session, allowing the agent to reuse
     previous questions and answers.

The agent makes at most ten model calls in its normal loop. Tools, including document search
and Genie polling, remain available until the final call. The final call is a tool-free wrap-up
based on the information gathered so far. If the wrap-up is empty, the agent retries it up to
twice without tools (at most twelve model calls in total). If it still receives no answer, the
turn fails instead of saving an empty response. Wrap-up retries request low reasoning effort;
session history is kept unchanged. Repeated searches can increase latency, cost and context size.
A session ID is not an access control
mechanism; access restrictions and retention periods must be configured separately.

## Key considerations

- The current embedding model, `databricks-gte-large-en`, is English-language. A multilingual
  model should be evaluated for Dutch and Frisian documents. Switching models requires
  rebuilding the embeddings and search index.
- New documents are **not automatically searchable**: chunking and Lakebase sync are manual.
  The chunk job creates a snapshot and does not overwrite an existing chunk table; refresh the
  chunks into a new table and explicitly switch consumers to it afterwards.
- Lakebase is currently enabled only for `dev`.

## Getting started

Settings are in [`project_config.yml`](project_config.yml); jobs and the Genie space are in
[`resources/`](resources/). Validate the Databricks bundle with:

```bash
databricks bundle validate -t dev
```

Deploy the bundle and run the jobs in the order above. To test all three tools and session
memory, run the unscheduled `testing_agent` job:

```bash
databricks bundle run -t dev testing_agent
```

The job runs [`testing_agent.py`](resources/deployment_notebooks/testing_agent.py) and passes the
deployed Genie space ID automatically. It makes live service calls and may start the SQL warehouse.
Agent logs show model-call outcomes, tool names, search result counts and memory steps without
logging tool arguments or document excerpts.
For interactive use, set the notebook's `genie_space_id` widget to the deployed space ID.
Example in a Databricks notebook:

```python
from databricks.sdk import WorkspaceClient
from notubiz_curator.agent import create_notubiz_agent
from notubiz_curator.config import load_config

agent = await create_notubiz_agent(
    load_config(env="dev"),
    WorkspaceClient(),
    genie_space_id="<genie-space-id>",
)
answer = agent.generate_message("What do the documents say about water quality?", session_id="session-1")
print(answer)
```

The identity in use needs access to the models, Lakebase, the Genie space, the SQL warehouse
and the document table. Web search also requires Unity Gateway and the relevant permissions.
Memory can use the same identity or a separate service principal via `LAKEBASE_SP_CLIENT_ID`,
`LAKEBASE_SP_CLIENT_SECRET` and `LAKEBASE_SP_HOST`.
