import json
import math
import os
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from safetensors import safe_open

MAX_HEADER_BYTES = 100 * 1024 * 1024

MODEL_ROOT = os.path.realpath(
    os.environ.get("TENSORFIELD_MODEL_ROOT", os.path.expanduser("~/models"))
)

_origins = os.environ.get(
    "TENSORFIELD_ALLOWED_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173",
).split(",")

app = FastAPI(title="Tensorfield Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_attached_path: Optional[str] = None
_tensor_index: dict = {}


class InspectRequest(BaseModel):
    path: str


def read_safetensors_header(path: str) -> dict:
    file_size = os.path.getsize(path)

    with open(path, "rb") as f:
        length_bytes = f.read(8)
        if len(length_bytes) != 8:
            raise HTTPException(400, "Invalid SafeTensors header")

        header_length = int.from_bytes(length_bytes, "little")

        if header_length <= 0 or header_length > MAX_HEADER_BYTES or header_length > file_size - 8:
            raise HTTPException(400, "Invalid SafeTensors header length")

        header_bytes = f.read(header_length)
        if len(header_bytes) != header_length:
            raise HTTPException(400, "Truncated SafeTensors header")

    try:
        header = json.loads(header_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise HTTPException(400, "Malformed SafeTensors metadata")

    if not isinstance(header, dict):
        raise HTTPException(400, "Invalid SafeTensors metadata")

    return header


def resolve_path(raw: str) -> str:
    path = os.path.realpath(os.path.expanduser(raw))

    if not path.lower().endswith(".safetensors"):
        raise HTTPException(400, "Only .safetensors files are supported")

    if not os.path.isfile(path):
        raise HTTPException(400, "SafeTensors file was not found")

    if os.path.commonpath([MODEL_ROOT, path]) != MODEL_ROOT:
        raise HTTPException(403, "Model path is outside the configured model directory")

    return path


@app.get("/v1/health")
def health():
    return {"status": "ok"}


@app.post("/v1/models/inspect")
def inspect_model(req: InspectRequest):
    global _attached_path, _tensor_index

    path = resolve_path(req.path)
    header = read_safetensors_header(path)

    tensors = []
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(metadata, dict):
            continue

        shape = metadata.get("shape")
        dtype = metadata.get("dtype")

        if not isinstance(shape, list) or not isinstance(dtype, str):
            raise HTTPException(400, f"Invalid metadata for tensor: {name}")

        tensors.append({"name": name, "shape": shape, "dtype": dtype})

    # Validate the file is a well-formed SafeTensors checkpoint
    with safe_open(path, framework="numpy", device="cpu"):
        pass

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
        tensor_view = f.get_slice(tensor)
        window = tensor_view[row: row + rows, column: column + columns]

    window = np.asarray(window, dtype=np.float64)

    if not np.isfinite(window).all():
        raise HTTPException(500, "Tensor slice contains NaN or Infinity")

    return {
        "tensor": tensor,
        "origin": [row, column],
        "shape": [rows, columns],
        "values": window.tolist(),
    }
