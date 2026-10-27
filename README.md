# llmops-stefan-98

Your repository for the LLMOps course on Databricks. Everything you need in
the course workspace was provisioned for you before you got this repo, and every id is already in
`project_config.yml`: you configure nothing.

## What was provisioned for you

| What | Value |
|---|---|
| Your schema in development, which you manage | `llmops_dev.stefan_98` |
| Your schema in acceptance, which you read | `llmops_acc.stefan_98` |
| Your schema in production, which you read | `llmops_prd.stefan_98` |
| Suggested volume name (created by the Notubiz job) | `notubiz_files`, such as `llmops_dev.stefan_98.notubiz_files` |
| SQL warehouse id | `114dd69b6fa40891` |
| Usage policy id | `4e4dddd7-4129-322f-9c85-3f7053f06f8b` |
| MLflow experiments | Dev: `/llmops/stefan_98-dev`; acc: `/llmops/stefan_98-acc`; prd: `/llmops/stefan_98-prd` |
| Chat model service | `course_ops.gateway.chat` |
| Embedding model service | `course_ops.gateway.embed` |

Both settings name model services of the course’s AI Gateway, with a monthly budget per student.
Chat routes to GPT-OSS-120B; embeddings route to GTE Large. Call both through the gateway
with the `databricks-openai` package:

```python
from databricks_openai import DatabricksOpenAI

client = DatabricksOpenAI(use_ai_gateway=True)
client.chat.completions.create(
    model="course_ops.gateway.chat",
    messages=[...],
    reasoning_effort="low",
    max_tokens=1024,
)
client.embeddings.create(model="course_ops.gateway.embed", input="A paper abstract")
```

The `databricks-openai` package is not included in this repo’s dependencies. The same APIs
are available through the installed `openai` client with the workspace base URL
`/ai-gateway/mlflow/v1` and Databricks authentication.

`embedding_endpoint` is a gateway model service despite the existing field name. Generate
vectors through its embeddings API before storing them in pgvector. It is not a managed
Vector Search index’s `embedding_model_endpoint_name`.

## Lakebase

The `lakebase_instance` value in `project_config.yml` identifies your Lakebase project.
All three environments share it; your Unity Catalog schemas remain separate.
Onboarding does not create an application database or install pgvector. Create the database
and extensions you need during the course. A Vector Search endpoint is not provisioned.

For a dedicated student project, you and your three deployment service principals have project
management access and OAuth Postgres roles with database administration privileges. The project
carries your usage policy and student cost tag. Its initial compute range is 0.5–2 CU, with
suspension after five idle minutes. Creating extra branches or computes can increase its cost.

## What is in the repo

- `src/notubiz_curator/`: configuration and OpenBesluitvorming ingestion helpers.
  Add the other course modules as you work through the lectures.
- `lecture_1/`, `lecture_2/` and so on, when you add them: copy a lecture's folder from the course
  code as it is, to the root of this repo, where its notebooks find `project_config.yml` one level up.
  Commit it with the hooks installed (below): the first commit stops once, after the hooks have
  formatted the notebooks and fixed what ruff can, and goes through when you add their changes and
  commit again. Ruff rules are configured in `pyproject.toml`.
- `databricks.yml`: bundle targets for `dev`, `acc`, and `prd`. No job is defined yet.
- `src/notubiz_curator/ingestion.py`: retrieves and normalizes the 50 most recent Wetterskip Fryslân
  Notubiz document records without downloading PDFs or writing Delta data.
- `pyproject.toml` and `version.txt`: the package and its version, bumped as the lectures ask.
- `tests/`: your tests; `test_config.py` checks that `project_config.yml` loads.
- `.github/workflows/ci.yml`: on every pull request and push to `main`, installs the package with
  uv and runs the pre-commit hooks and the tests.
- `.github/CODEOWNERS`: who reviews a pull request to `main`, which needs one approving review.

## Getting started

```bash
uv sync --extra dev
uvx pre-commit install
databricks bundle validate -t dev
```

## Public Governance Document Assistant

The assistant serves board members, employees, elected representatives, and citizens of Wetterskip
Fryslân who need reliable answers from public meeting documents without manually reviewing agendas
and attachments. Its minimal corpus will contain the 50 most recent Notubiz records with extracted
`markdownText` and metadata including document title, meeting date, meeting identifier, and a public
PDF source URL.

`fetch_documents(50)` returns storage-neutral `NotubizDocument` records, one complete source
document per record. A later job can turn their `as_record()` output into a Spark DataFrame, write
it to `notubiz_documents`, then split
`document_text` into retrieval chunks. The current focus is only this ingestion pipeline; no
Gateway, model-service, embedding, or agent code is included.

Run the regular ingestion locally or from a Python task with:

```bash
uv run python -m notubiz_curator.scripts.ingest_notubiz --document-count 50
```

The script writes to `llmops_dev.stefan_98.notubiz_documents` by default. Configure a Databricks
task with the same parameters when deployed:

```bash
python -m notubiz_curator.scripts.ingest_notubiz \
  --environment dev \
  --config-path project_config.yml \
  --table-name notubiz_documents \
  --databricks-profile course
```

When run locally, `--databricks-profile` selects the Databricks Connect profile. It can also be set
once with `DATABRICKS_CONFIG_PROFILE`; otherwise the script fails rather than using the unrelated
`DEFAULT` profile. In a Databricks job, omit it and the runtime Spark session is used automatically.

This script always performs a Delta `MERGE`: it inserts documents whose `document_id` is new and
updates only existing records whose source metadata or document text changed. It does not duplicate
unchanged documents or replace the full corpus.

`resources/notubiz_ingestion.jobs.yml` defines two serverless-environment-5 DAB jobs:

- `notubiz_ingestion` is named `notubiz-ingestion-${bundle.target}` and runs daily at 00:00
  (`Europe/Amsterdam`), merging the 50 most recent documents.
- `notubiz_bootstrap` has no schedule and must be started manually. It reads every page of the
  OpenBesluitvorming snapshot feed and deliberately replaces `notubiz_documents` once. Run it before
  enabling the daily ingestion job to initialize the historic corpus.

Each environment logs to its own MLflow experiment and UC trace tables in its matching catalog. You can manage the dev experiment and write dev traces; acc and prd are read-only for you. Each deploy service principal manages its own environment’s experiment and writes to that environment’s schema.
