# Deploying the memory service to Cloud Run

This deploys the FastAPI service (`serve/app.py`) to Google Cloud Run. Two embedding
backends are supported: `local` (`all-MiniLM-L6-v2`, baked into the image) and `vertex`
(Vertex AI text-embeddings). The store is in-memory and single-instance — see
[Persistence](#persistence) before treating it as stateful production.

## Public surface (read-only) vs admin (Bearer)

The deployed service exposes a **read-only, rate-limited** public surface over a **bundled
synthetic corpus** (never user data):

| method + path | auth | purpose |
|---|---|---|
| `POST /recall` | public (rate-limited) | ranked retrieval → `{ records: [...] }` |
| `GET  /stats`  | public (rate-limited) | `{ active, superseded }` |
| `GET  /health` | public | liveness + active backend |
| `POST /admin/rebuild` | **Bearer** | rebuild the index from the active set → `{ rebuilt }` |
| `POST /remember` `/ingest` `/consolidate` `/forget` | **Bearer** | mutate the in-memory store |

Auth is an app-layer Bearer token read from `ADMIN_TOKEN`. It is **never baked into the
image** — it is injected at deploy time from Secret Manager. When `ADMIN_TOKEN` is unset,
the admin/mutating routes return `503` (disabled) rather than being left open.

There is **no background maintenance thread**: Cloud Run freezes CPU on idle instances, so
an in-process timer would not fire reliably. The index is rebuilt on write and on demand
via `POST /admin/rebuild`, which a **Cloud Scheduler** job calls on a cadence (below).

> The `vertex` backend calls a live Google Cloud API and was **not** exercised by the
> author; validate it against your own project. Pricing figures from `bench/costmodel.py`
> use published rates that change — verify before quoting.

## 0. Prerequisites

- A GCP project with billing enabled — https://console.cloud.google.com/billing
- The `gcloud` CLI — https://cloud.google.com/sdk/docs/install
- Docker is **not** required (Cloud Run builds from source via Cloud Build).

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
export PROJECT_ID=$(gcloud config get-value project)
export REGION=us-central1
```

Enable the APIs (Cloud Run + Cloud Build + Secret Manager + Cloud Scheduler; add Vertex
only for the vertex backend):

```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  secretmanager.googleapis.com cloudscheduler.googleapis.com
# vertex backend only:
gcloud services enable aiplatform.googleapis.com
```

Create the admin token as a secret (used by `/admin/rebuild` and the mutating routes):

```bash
openssl rand -hex 32 | tr -d '\n' | gcloud secrets create memory-admin-token --data-file=-
# grant the Cloud Run runtime service account read access to it
export SA="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')-compute@developer.gserviceaccount.com"
gcloud secrets add-iam-policy-binding memory-admin-token \
  --member="serviceAccount:$SA" --role="roles/secretmanager.secretAccessor"
```

Docs: Cloud Run quickstart — https://cloud.google.com/run/docs/quickstarts/deploy-container

## 1. Deploy (local embeddings — default, no Vertex needed)

From the repo root (the `Dockerfile` is detected automatically):

```bash
gcloud run deploy agent-memory-service \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --cpu 2 --memory 2Gi \
  --concurrency 8 \
  --min-instances 0 --max-instances 1 \
  --set-env-vars RATE_LIMIT_PER_MINUTE=60,INDEX_TYPE=flat \
  --set-secrets ADMIN_TOKEN=memory-admin-token:latest
```

- `--max-instances 1` bounds cost **and** keeps the in-memory store and the per-instance
  rate limiter coherent (the store is not shared across instances — see Persistence).
- `--memory 2Gi` gives torch + MiniLM room to load; `--cpu 2` speeds the cold-start
  embedding warmup. `--concurrency 8` keeps per-request CPU contention low on the local
  backend (embedding is CPU-bound).
- `RATE_LIMIT_PER_MINUTE` caps requests per client IP (set `0` to disable). The first
  build takes a few minutes (it bakes the MiniLM model).

Get the URL and smoke-test the public surface:

```bash
export URL=$(gcloud run services describe agent-memory-service \
  --region "$REGION" --format='value(status.url)')
curl -s "$URL/health"
curl -s -X POST "$URL/recall" -H 'content-type: application/json' \
  -d '{"query":"Who is my current manager?","k":3}'
curl -s "$URL/stats"
```

## 2. Scheduled index rebuild (Cloud Scheduler)

The rebuild cadence is external (no in-process threads). Cloud Scheduler hits the
Bearer-guarded `/admin/rebuild` on a schedule:

```bash
export ADMIN_TOKEN=$(gcloud secrets versions access latest --secret=memory-admin-token)
gcloud scheduler jobs create http memory-rebuild \
  --location "$REGION" \
  --schedule "*/30 * * * *" \
  --uri "$URL/admin/rebuild" \
  --http-method POST \
  --headers "Authorization=Bearer ${ADMIN_TOKEN}" \
  --attempt-deadline 60s

# trigger once now and confirm a successful invocation
gcloud scheduler jobs run memory-rebuild --location "$REGION"
gcloud scheduler jobs describe memory-rebuild --location "$REGION" \
  --format='value(status.lastAttemptTime,state)'
```

The job config stores the token in its header. For a stricter posture, make the whole
service IAM-authenticated and drive the job with an **OIDC token** from a dedicated
service account holding `roles/run.invoker` (`--oidc-service-account-email`), instead of
`--allow-unauthenticated` + app-layer Bearer. Cloud Scheduler HTTP targets —
https://cloud.google.com/scheduler/docs/creating

## 3. Deploy (Vertex embeddings)

Grant the service's runtime service account access to Vertex, then deploy with the
backend env vars. By default Cloud Run uses the Compute Engine default service account:

```bash
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:$SA" --role="roles/aiplatform.user"

gcloud run deploy agent-memory-service \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --cpu 1 --memory 1Gi --concurrency 20 --min-instances 0 --max-instances 1 \
  --set-env-vars EMBEDDINGS_BACKEND=vertex,GOOGLE_CLOUD_PROJECT=$PROJECT_ID,VERTEX_LOCATION=$REGION,RATE_LIMIT_PER_MINUTE=60 \
  --set-secrets ADMIN_TOKEN=memory-admin-token:latest
```

With Vertex, embedding runs off-box, so `--concurrency` can be higher. Confirm the
backend is active:

```bash
curl -s "$URL/health"   # -> {"status":"ok","embeddings_backend":"vertex",...}
```

Vertex embeddings model/pricing — https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings

## 4. Measure p50/p99 and cost against the live service

```bash
# real deployed latency + throughput
python bench/loadtest.py --url "$URL" --seed 500 --requests 1000 --concurrency 32

# $/1k requests from the measured p50 (add --vertex if using the vertex backend)
python bench/costmodel.py --latency-ms <p50_from_loadtest> --vcpu 2 --memory-gib 2
```

Cloud Run pricing — https://cloud.google.com/run/pricing · Vertex pricing — https://cloud.google.com/vertex-ai/pricing

## Dependency lock

The image installs from a fully-pinned (exact `==`) `requirements.txt` — a `pip-compile`
lock of `requirements.in`, with CPU-only torch from the PyTorch index. Regenerate it in a
linux/py3.12 container:

```bash
make lock     # docker run python:3.12-slim pip-compile
```

Add `--generate-hashes` (and re-enable `pip install --require-hashes` in the Dockerfile)
when building from a network that allows the PyTorch CDN wheel download.

## Persistence

The service holds memories in process RAM on a single instance. On scale-to-zero or
multi-instance autoscaling, state is not shared or retained. This deployment
demonstrates the **serving, latency, and cost path**; durable memory is the production
extension, via either:

- **pgvector on Supabase / Cloud SQL** — persist `MemoryItem`s and their embeddings; back the retrieval path with a SQL ANN index.
- **Vertex AI Vector Search** — a managed ANN index for the embedding store (https://cloud.google.com/vertex-ai/docs/vector-search/overview).

Both replace the in-memory `VectorIndex` behind the same `MemoryService` surface
(`get_active_vectors()` / `rebuild_index()`).

## Teardown

```bash
gcloud scheduler jobs delete memory-rebuild --location "$REGION" --quiet
gcloud run services delete agent-memory-service --region "$REGION" --quiet
gcloud secrets delete memory-admin-token --quiet
```
