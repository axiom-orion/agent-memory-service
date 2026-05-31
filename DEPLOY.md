# Deploying the memory service to Cloud Run

This deploys the FastAPI service (`serve/app.py`) to Google Cloud Run. Two embedding
backends are supported: `local` (`all-MiniLM-L6-v2`, baked into the image) and `vertex`
(Vertex AI text-embeddings). The store is in-memory and single-instance — see
[Persistence](#persistence) before treating it as stateful production.

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

Enable the APIs (Cloud Run + Cloud Build; add Vertex only for the vertex backend):

```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com
# vertex backend only:
gcloud services enable aiplatform.googleapis.com
```

Docs: Cloud Run quickstart — https://cloud.google.com/run/docs/quickstarts/deploy-container

## 1. Deploy (local embeddings — default, no Vertex needed)

From the repo root (the `Dockerfile` is detected automatically):

```bash
gcloud run deploy agent-memory-service \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --cpu 1 --memory 1Gi \
  --concurrency 8 \
  --min-instances 0 --max-instances 3
```

`--concurrency 8` keeps per-request CPU contention low (embedding is CPU-bound on the
local backend). The first build takes a few minutes (it bakes the MiniLM model).

Get the URL and smoke-test it:

```bash
export URL=$(gcloud run services describe agent-memory-service \
  --region "$REGION" --format='value(status.url)')
curl -s "$URL/healthz"
```

## 2. Deploy (Vertex embeddings)

Grant the service's runtime service account access to Vertex, then deploy with the
backend env vars. By default Cloud Run uses the Compute Engine default service account:

```bash
export SA="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')-compute@developer.gserviceaccount.com"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:$SA" --role="roles/aiplatform.user"

gcloud run deploy agent-memory-service \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --cpu 1 --memory 1Gi --concurrency 20 --max-instances 3 \
  --set-env-vars EMBEDDINGS_BACKEND=vertex,GOOGLE_CLOUD_PROJECT=$PROJECT_ID,VERTEX_LOCATION=$REGION
```

With Vertex, embedding runs off-box, so `--concurrency` can be higher. Confirm the
backend is active:

```bash
curl -s "$URL/healthz"   # -> {"status":"ok","embeddings_backend":"vertex"}
```

Vertex embeddings model/pricing — https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings

## 3. Measure p50/p99 and cost against the live service

```bash
# real deployed latency + throughput
python bench/loadtest.py --url "$URL" --seed 500 --requests 1000 --concurrency 32

# $/1k requests from the measured p50 (add --vertex if using the vertex backend)
python bench/costmodel.py --latency-ms <p50_from_loadtest> --vcpu 1 --memory-gib 1
```

Cloud Run pricing — https://cloud.google.com/run/pricing · Vertex pricing — https://cloud.google.com/vertex-ai/pricing

## Persistence

The service holds memories in process RAM on a single instance. On scale-to-zero or
multi-instance autoscaling, state is not shared or retained. This deployment
demonstrates the **serving, latency, and cost path**; durable memory is the production
extension, via either:

- **pgvector on Supabase / Cloud SQL** — persist `MemoryItem`s and their embeddings; back the retrieval path with a SQL ANN index.
- **Vertex AI Vector Search** — a managed ANN index for the embedding store (https://cloud.google.com/vertex-ai/docs/vector-search/overview).

Both replace the in-memory `VectorIndex` behind the same `MemoryService` surface.

## Teardown

```bash
gcloud run services delete agent-memory-service --region "$REGION"
```
