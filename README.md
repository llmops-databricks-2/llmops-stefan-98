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
| Embedding serving endpoint | `databricks-gte-large-en` |

The chat setting names a model service of the course's AI Gateway. The embedding setting uses the
Databricks serving endpoint directly, both for chunk embeddings and search queries. The gateway chat
service can be called with the `databricks-openai` package:

```python
from databricks_openai import DatabricksOpenAI

client = DatabricksOpenAI(use_ai_gateway=True)
client.chat.completions.create(
    model="course_ops.gateway.chat",
    messages=[...],
    reasoning_effort="low",
    max_tokens=1024,
)
```

The `databricks-openai` package is not included in this repo’s dependencies. The same APIs
are available through the installed `openai` client with the workspace base URL
`/ai-gateway/mlflow/v1` and Databricks authentication.

`embedding_endpoint` identifies the serving endpoint used to generate vectors. It is not a managed
Vector Search index's `embedding_model_endpoint_name`.

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
- `databricks.yml`: bundle targets for `dev`, `acc`, and `prd`; job definitions are in `resources/`.
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
document per record. The ingestion scripts store them in `notubiz_documents`; a separate,
explicitly run script can create embedded retrieval chunks as described below.

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

`resources/notubiz_ingestion.jobs.yml` defines serverless-environment-5 DAB jobs, including:

- `notubiz_ingestion` is named `notubiz-ingestion-${bundle.target}` and runs daily at 00:00
  (`Europe/Amsterdam`), merging the 50 most recent documents.
- `notubiz_bootstrap` has no schedule and must be started manually. It reads every page of the
  OpenBesluitvorming snapshot feed and deliberately replaces `notubiz_documents` once. Run it before
  enabling the daily ingestion job to initialize the historic corpus.

Each environment logs to its own MLflow experiment and UC trace tables in its matching catalog. You can manage the dev experiment and write dev traces; acc and prd are read-only for you. Each deploy service principal manages its own environment’s experiment and writes to that environment’s schema.

## Notubiz chunks and embeddings

The `ChunkingProcessor` in `src/notubiz_curator/chunking_processor.py` implements the processing
steps in one class, following the course's `DataProcessor` structure. The one-time script creates
the processor and calls `process_chunks()`. It reads the existing `notubiz_documents` Delta table, cleans its
`document_text` Markdown, splits on headings and then into smaller text segments, and calls the
configured embedding model with `ai_query`. It creates `notubiz_chunks`, with one row per chunk:

- `chunk_id`, `chunk_index`, `text`, and `section`;
- `document_id`, `document_title`, `meeting_id`, `meeting_date`, and `source_url`;
- `embedding`, an array of floats.

The title and section are included in the embedded text as well as being available as metadata.
Empty or missing document text produces no chunks and is counted in the log. The script does not
call `ai_parse_document`: the source is already Markdown, not a PDF. Like the course example,
Spark uses a Python UDF and `posexplode` to create chunks, then `ai_query` generates embeddings
using `embedding_endpoint`. The job logs when chunking and embedding generation start and finish.
Embedding generation is one Spark operation, so there is no per-batch progress log.

Chunking reads a fixed Delta table version and writes the chunks to a temporary staging table before
generating embeddings. A long-running job needs the source version to remain available until chunking
finishes; do not vacuum away its source files during that time. Processing the full corpus can take a
long time and consume course budget.

Chunks contain at most 1,200 characters including title and section, with up to 100 characters of
overlap within a section. Using roughly four characters per token, 1,200 characters is about 300
tokens in typical English text. This is only an estimate: Dutch and other text can tokenize
differently, so the character limit does **not** guarantee a maximum of 512 tokens or any model's
context limit. Evaluate chunk sizes against your actual documents and model.

The configured embedding endpoint is `databricks-gte-large-en`. The job requires permission to call
the configured endpoint.

Run locally with Databricks Connect and the course profile:

```bash
uv sync
uv run python -m notubiz_curator.scripts.create_notubiz_chunks --databricks-profile course
```

In Databricks, install the package and run the same module without `--databricks-profile`. The
optional `--environment`, `--config-path`, `--source-table`, `--target-table` and `--max-chars`
arguments select a different configuration or limit. The character limit cannot exceed 1,200.

The bundle also defines the unscheduled `notubiz-chunks-${bundle.target}` job in
`resources/notubiz_ingestion.jobs.yml`. After deploying the bundle, start it manually in
Databricks. It uses the target environment's `notubiz_documents` and creates `notubiz_chunks`.
No job is started automatically by deployment.

This is a **full snapshot**, not an incremental job. It refuses to overwrite an existing target
table and only creates the final table after all embedding calls and staging writes succeed.
An interrupted or failed run can leave a uniquely named `notubiz_chunks_staging_<id>` table for
inspection and manual cleanup. To refresh after
document changes, choose a new target table and
explicitly switch consumers after checking it. The bundled job uses the default target name;
to refresh to a new name, change its parameters or run the script manually with `--target-table`.

Open `notebooks/notubiz_chunk_statistics.py` in Databricks to inspect chunk lengths and verify
that they stay within the character limit. It does not measure tokens. The notebook reads the configured environment
(`ENV`, default `dev`) and does not create embeddings.

### Remarks: embedding language

The existing `embedding_endpoint: "databricks-gte-large-en"` is an **English-language** embedding
model. It is used here as
requested, but a multilingual embedding model is preferable for predominantly Dutch questions and
documents, and should also be evaluated on any Frisian content. Switching models requires
rebuilding the embeddings and search index.

## Lakebase Synced Table and vector search

The manual `notubiz-chunks-${bundle.target}` job creates the Delta chunks table. The separate manual
`notubiz-lakebase-sync-${bundle.target}` job creates or refreshes `chunks_synced` in Lakebase:

1. It validates `notubiz_chunks`, its unique `chunk_id` values and 1,024-dimensional embeddings,
   and enables Delta Change Data Feed.
2. It installs the course's `lakebase_vector` extension and creates
   `llmops_dev.stefan_98.chunks_synced` with `chunk_id` as the primary key and `embedding` as
   `vector(1024)`. It uses the configured usage policy for the sync pipeline.
3. It waits for the initial sync or the specific requested update, then creates the `lakebase_ann`
   vector index and the full-text GIN index. The Synced Table is a read-only copy; write changes to
   the Delta source.

The scheduling policy is **TRIGGERED**, as in the course notebook. Creation starts the first sync;
later source changes are copied only when the sync job is run again. The sync job does not create
new chunks: the current chunking job is a one-time snapshot, so updating `notubiz_documents` alone
does not update the synced chunks.

Dev, acc and prd currently share the same Lakebase project, database and PostgreSQL schema. Only dev
has `lakebase_enabled: true`; acc and prd fail before accessing Lakebase until separate destinations
are configured. The job and pipeline consume serverless and Lakebase budget.

After deploying the bundle, run the chunk job first if `notubiz_chunks` does not yet exist, then
start the Lakebase sync job. Alternatively, with Databricks Connect and the course profile:

```bash
uv run python -m notubiz_curator.scripts.sync_lakebase --environment dev --databricks-profile course
```

`LakebasePgVectorManager` in `src/notubiz_curator/vector_search.py` supports cosine and hybrid
search, with an optional `meeting_id` filter. It embeds questions with the same configured endpoint
as the documents and returns chunk text and Notubiz metadata:

```python
from notubiz_curator.config import load_config
from notubiz_curator.vector_search import LakebasePgVectorManager

manager = LakebasePgVectorManager(load_config(env="dev"))
results = manager.search("Wat is het beleid voor waterkwaliteit?", num_results=5)
hybrid_results = manager.hybrid_search("waterkwaliteit", meeting_id="meeting-id")
```

Hybrid search uses PostgreSQL's Dutch text configuration, rather than the English configuration in
the course example. As in the course, the manager creates the GIN index; the sync module creates the
ANN index separately. Credentials are refreshed for each database connection. The course's optional
`LAKEBASE_ENDPOINT`, `PGHOST`, and `PGDATABASE` settings are supported only when they still point to
the configured project, production branch, and `databricks_postgres` database. `PGHOST` must be the
unpooled endpoint host: Lakebase rejects OAuth authentication through the pooled host.
Lakebase Search must be enabled on the project before installing
`lakebase_vector` and creating the course's `lakebase_ann` index. Enabling it is a separate,
irreversible project setting and can restart compute. The notebook's `ai_search` comparison itself
is not run by the job.
