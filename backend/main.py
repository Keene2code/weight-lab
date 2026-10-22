import math
import os
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from safetensors import safe_open

app = FastAPI(title="Tensorfield Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Single attached model held in process memory
_attached_path: Optional[str] = None
_tensor_index: dict = {}


class InspectRequest(BaseModel):
    path: str


@app.get("/v1/health")
def health():
    return {"status": "ok"}


@app.post("/v1/models/inspect")
def inspect_model(req: InspectRequest):
    global _attached_path, _tensor_index

    path = req.path

    if not path.endswith(".safetensors"):
        raise HTTPException(400, "Only .safetensors files are supported")
    if not os.path.exists(path):
        raise HTTPException(400, f"File not found: {path}")
    if os.path.isdir(path):
        raise HTTPException(400, "Path must be a file, not a directory")

    tensors = []
    with safe_open(path, framework="numpy", device="cpu") as f:
        for name in f.keys():
            meta = f.get_slice(name)
            shape = list(meta.get_shape())
            dtype = str(f.get_tensor(name).dtype) if len(shape) == 0 else _get_dtype(f, name, shape)
            tensors.append({"name": name, "shape": shape, "dtype": dtype})

    _attached_path = path
    _tensor_index = {t["name"]: t for t in tensors}

    return {
        "id": "local-model",
        "name": os.path.basename(path),
        "tensors": tensors,
    }


@app.get("/v1/tensor-slice")
def tensor_slice(
    tensor: str = Query(...),
    row: int = Query(0, ge=0),
    column: int = Query(0, ge=0),
    rows: int = Query(4, ge=1, le=16),
    columns: int = Query(4, ge=1, le=16),
):
    if _attached_path is None:
        raise HTTPException(400, "No model attached. Call POST /v1/models/inspect first.")

    if tensor not in _tensor_index:
        raise HTTPException(404, f"Tensor not found: {tensor}")

    meta = _tensor_index[tensor]
    if len(meta["shape"]) != 2:
        raise HTTPException(400, "Only rank-2 tensors are supported for slicing")

    total_rows, total_cols = meta["shape"]

    if row >= total_rows or column >= total_cols:
        raise HTTPException(400, "Row or column origin is out of bounds")

    rows = min(rows, total_rows - row)
    columns = min(columns, total_cols - column)

    if rows * columns > 256:
        raise HTTPException(400, "Slice exceeds 256-element limit")

    with safe_open(_attached_path, framework="numpy", device="cpu") as f:
        full = f.get_tensor(tensor)

    window = full[row:row + rows, column:column + columns].astype(float)

    values = window.tolist()
    for r in values:
        for v in r:
            if math.isnan(v) or math.isinf(v):
                raise HTTPException(500, "Tensor contains NaN or Infinity values")

    return {
        "tensor": tensor,
        "origin": [row, column],
        "shape": [rows, columns],
        "values": values,
    }


def _get_dtype(f, name: str, shape: list) -> str:
    if not shape:
        return "unknown"
    try:
        t = f.get_tensor(name)
        return str(t.dtype)
    except Exception:
        return "unknown"
