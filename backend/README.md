# Tensorfield Backend

Minimal read-only SafeTensors inspection API.

## Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000
```

## Smoke test

```bash
curl http://127.0.0.1:8000/v1/health

curl -X POST http://127.0.0.1:8000/v1/models/inspect \
  -H "Content-Type: application/json" \
  -d '{"path": "/path/to/model.safetensors"}'

curl "http://127.0.0.1:8000/v1/tensor-slice?tensor=model.layers.0.self_attn.q_proj.weight&row=0&column=0&rows=4&columns=4"
```
