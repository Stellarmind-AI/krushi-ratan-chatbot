# Deployment Guide

This guide is for deploying the current codebase so the latest chatbot behavior is preserved:

- the updated dashboard WebSocket callback flow works
- SQL remains disabled in production
- the deployed frontend points to the correct deployed backend

## 1. Deploy the latest code on both services

The frontend and backend must run the same latest code version.

- Backend must include:
  - `app/websocket/chat_handler.py`
  - `app/services/agent/orchestrator.py`
  - `app/services/agent/knowledge_handler.py`
- Frontend must include:
  - `streamlit_app/dashboard.py`

If only the dashboard is updated but the deployed backend is still old, you will keep seeing old behavior.

## 2. Use these backend environment values

Set the deployed backend `.env` like this:

```env
APP_HOST=0.0.0.0
APP_PORT=8000
ENVIRONMENT=production
LOG_LEVEL=INFO
ENABLE_SQL_FLOW=false
```

Keep your existing DB and API key values as needed.

Why this matters:

- `ENABLE_SQL_FLOW=false` is what keeps SQL disabled in the current release.
- That behavior is enforced in `app/services/agent/orchestrator.py`.

## 3. Use these frontend environment values

For the deployed Streamlit app, set:

```env
HEALTH_CHECK_URL=https://test-ai.krushiratn.com/health
WS_CHAT_URL=wss://test-ai.krushiratn.com/ws/chat
```

Replace `test-ai.krushiratn.com` with your real deployed backend domain if it changes.

Why this matters:

- The dashboard now resolves backend URLs from env/secrets instead of hardcoding localhost.
- That logic is in `streamlit_app/dashboard.py`.

## 4. Install with the updated requirements

The dashboard needs packages that were previously missing from `requirements.txt`.

Run:

```bash
pip install -r requirements.txt
```

This now includes:

- `streamlit`
- `websockets`
- `requests`

## 5. Run the backend as a single worker

This is mandatory for the current callback / clarification flow.

Run:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Do not use multiple workers for this release, for example do not run:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Why this matters:

- clarification state is stored in memory in `app/websocket/chat_handler.py`
- pending clarification is saved in `app/websocket/chat_handler.py`
- callback resolution reads it back in `app/websocket/chat_handler.py`

If a second WebSocket request lands on another worker or another server instance, the callback can fail with:

`No pending clarification for this session. Please send your question again.`

## 6. Run the Streamlit frontend

Run:

```bash
streamlit run streamlit_app/dashboard.py --server.port 8501
```

If you use Streamlit Cloud or another managed host, set `HEALTH_CHECK_URL` and `WS_CHAT_URL` in that platform's secrets or environment settings.

## 7. If using Nginx or a reverse proxy, enable WebSocket forwarding

Your proxy must support WebSocket upgrade on `/ws/chat`.

Minimum Nginx behavior required:

- forward `/health` to the FastAPI service
- forward `/ws/chat` to the FastAPI service
- pass `Upgrade` and `Connection` headers for WebSocket traffic

If WebSocket upgrade is not enabled, the dashboard may show connected health checks but chat will still fail.

## 8. Deployment checklist

Before marking deployment complete, verify all of these:

1. Backend and frontend are both updated to the latest code.
2. `ENABLE_SQL_FLOW=false` is set on the deployed backend.
3. Streamlit is using `WS_CHAT_URL=wss://<your-domain>/ws/chat`.
4. The backend is running with a single worker.
5. `https://<your-domain>/health` returns `200`.
6. Ask a SQL-style question such as `kapas bhav today`.
7. Confirm the reply is the SQL-disabled fallback, not a DB result.
8. Ask an app-navigation question and confirm it still answers correctly.

## 9. Exact deployment sequence

Use this order:

1. Pull the latest code on the server.
2. Update backend `.env` with `ENABLE_SQL_FLOW=false`.
3. Update frontend env/secrets with `HEALTH_CHECK_URL` and `WS_CHAT_URL`.
4. Reinstall dependencies with `pip install -r requirements.txt`.
5. Restart FastAPI as a single worker.
6. Restart Streamlit.
7. Test one SQL-type query and one navigation query.

## 10. Most likely reason deployment still shows old answers

If deployment still behaves like the old version, the usual cause is one of these:

- the deployed backend is still running old code
- `ENABLE_SQL_FLOW` is still `true` on the server
- Streamlit is still pointing to the wrong backend URL
- the backend is running with multiple workers and the callback state is getting lost
