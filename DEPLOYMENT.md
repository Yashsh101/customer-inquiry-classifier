# Deployment

This project has two deployable surfaces:

- Streamlit demo UI.
- FastAPI inference API.

## Streamlit Community Cloud

1. Create a Streamlit app from this repository.
2. Set the entrypoint to `streamlit_app.py`.
3. Add secrets or environment variables:
   - `API_BASE_URL` if using a separate hosted API.
   - `ROUTING_CONF_THRESHOLD` if the app trains/loads locally.
4. Confirm the app can load `models/classifier_v2.joblib` or train a model on startup.

## Render API

The repository includes `render.yaml`.

1. Create a Render Blueprint from the repository.
2. Set environment variables from `.env.example`.
3. Use:

```bash
uvicorn app.api:app --host 0.0.0.0 --port $PORT
```

4. Verify:

```bash
curl https://<service-url>/health
curl -X POST https://<service-url>/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"I need help with a refund"}'
```

## Docker

```bash
docker build -t customer-inquiry-classifier .
docker run -p 8000:8000 --env-file .env customer-inquiry-classifier
```

## Production Checklist

- Keep `ENABLE_LLM_FALLBACK=false` unless an API key and cost limits are configured.
- Monitor low-confidence escalation rate.
- Rebuild the model when categories or incoming language patterns change.
- Store larger model artifacts in releases or external storage if they grow beyond Git-friendly size.
