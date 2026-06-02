#!/usr/bin/env bash
# One-command Cloud Run deploy for the agent memory service.
#
# Deploys the read-only public /recall demo (bundled synthetic corpus), wires a Cloud
# Scheduler job to rebuild the index on a cadence, smoke-tests the live endpoints, and
# writes the live URL into README.md. Idempotent: re-running updates in place.
#
# Prereqs (see DEPLOY.md): gcloud authed; PROJECT_ID with billing; APIs enabled.
# Usage:
#   PROJECT_ID=my-proj REGION=us-central1 scripts/deploy_cloudrun.sh
set -euo pipefail

: "${PROJECT_ID:?set PROJECT_ID (gcloud project with billing enabled)}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-agent-memory-service}"
SECRET="${SECRET:-memory-admin-token}"
RATE_LIMIT="${RATE_LIMIT_PER_MINUTE:-60}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

gcloud config set project "$PROJECT_ID" >/dev/null

echo "==> Enabling APIs"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  secretmanager.googleapis.com cloudscheduler.googleapis.com >/dev/null

echo "==> Ensuring admin-token secret ($SECRET)"
if ! gcloud secrets describe "$SECRET" >/dev/null 2>&1; then
  openssl rand -hex 32 | tr -d '\n' | gcloud secrets create "$SECRET" --data-file=-
fi
SA="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')-compute@developer.gserviceaccount.com"
gcloud secrets add-iam-policy-binding "$SECRET" \
  --member="serviceAccount:$SA" --role="roles/secretmanager.secretAccessor" >/dev/null

echo "==> Deploying $SERVICE to Cloud Run ($REGION)"
gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --cpu 2 --memory 2Gi \
  --concurrency 8 \
  --min-instances 0 --max-instances 1 \
  --set-env-vars "RATE_LIMIT_PER_MINUTE=${RATE_LIMIT},INDEX_TYPE=flat" \
  --set-secrets "ADMIN_TOKEN=${SECRET}:latest"

URL="$(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)')"
echo "==> Live at $URL"

echo "==> Wiring Cloud Scheduler rebuild job (every 30 min)"
ADMIN_TOKEN="$(gcloud secrets versions access latest --secret="$SECRET")"
if gcloud scheduler jobs describe memory-rebuild --location "$REGION" >/dev/null 2>&1; then
  gcloud scheduler jobs update http memory-rebuild --location "$REGION" \
    --uri "$URL/admin/rebuild" --http-method POST \
    --update-headers "Authorization=Bearer ${ADMIN_TOKEN}" --attempt-deadline 60s
else
  gcloud scheduler jobs create http memory-rebuild --location "$REGION" \
    --schedule "*/30 * * * *" --uri "$URL/admin/rebuild" --http-method POST \
    --headers "Authorization=Bearer ${ADMIN_TOKEN}" --attempt-deadline 60s
fi
gcloud scheduler jobs run memory-rebuild --location "$REGION" >/dev/null || true

echo "==> Smoke test"
curl -fsS "$URL/health" && echo
curl -fsS -X POST "$URL/recall" -H 'content-type: application/json' \
  -d '{"query":"Who is my current manager?","k":3}' && echo
curl -fsS "$URL/stats" && echo

echo "==> Writing live URL into README.md"
python - "$URL" <<'PY'
import re, sys, pathlib
url = sys.argv[1]
p = pathlib.Path("README.md")
text = p.read_text(encoding="utf-8")
block = (
    "<!-- LIVE_URL_START -->\n"
    f"**Live demo:** [`{url}`]({url}) — a rate-limited, read-only `POST /recall` over a "
    "bundled synthetic corpus.\n\n```bash\n"
    f"curl -s -X POST {url}/recall -H 'content-type: application/json' "
    "-d '{\"query\":\"Who is my current manager?\",\"k\":3}'\n"
    f"curl -s {url}/stats\n```\n"
    "<!-- LIVE_URL_END -->"
)
text = re.sub(r"<!-- LIVE_URL_START -->.*?<!-- LIVE_URL_END -->", block, text, flags=re.S)
p.write_text(text, encoding="utf-8")
print("README.md updated with live URL")
PY

echo "==> Done. Review README.md, then commit."
