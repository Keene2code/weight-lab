import json
import os
import struct
import tempfile
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("TENSORFIELD_MODEL_ROOT", tempfile.gettempdir())

from main import app, read_safetensors_header

client = TestClient(app)


def make_safetensors(tensors: dict[str, np.ndarray]) -> bytes:
    """Build a minimal valid SafeTensors file in memory."""
    metadata = {}
    offset = 0
    data_parts = []

    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        nbytes = arr.nbytes
        metadata[name] = {
            "dtype": "F32",
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        data_parts.append(arr.tobytes())
        offset += nbytes

    header_bytes = json.dumps(metadata).encode("utf-8")
    length = struct.pack("<Q", len(header_bytes))
    return length + header_bytes + b"".join(data_parts)


def write_safetensors(tensors: dict[str, np.ndarray], path: str):
    with open(path, "wb") as f:
        f.write(make_safetensors(tensors))


@pytest.fixture()
def model_file(tmp_path):
    arr = np.arange(16, dtype=np.float32).reshape(4, 4) * 0.1
    path = str(tmp_path / "model.safetensors")
    write_safetensors({"layer.weight": arr}, path)
    return path, arr


# --- health ---

def test_health():
    r = client.get("/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# --- inspect ---

def test_inspect_returns_metadata(model_file):
    path, _ = model_file
    r = client.post("/v1/models/inspect", json={"path": path})
    assert r.status_code == 200
    data = r.json()
    assert data["name"] == "model.safetensors"
    tensors = {t["name"]: t for t in data["tensors"]}
    assert "layer.weight" in tensors
    assert tensors["layer.weight"]["shape"] == [4, 4]
    assert tensors["layer.weight"]["dtype"] == "F32"


def test_inspect_never_calls_get_tensor(model_file):
    path, _ = model_file
    with patch("main.safe_open") as mock_open:
        mock_ctx = mock_open.return_value.__enter__.return_value
        mock_ctx.get_tensor.side_effect = AssertionError("get_tensor must not be called")
        r = client.post("/v1/models/inspect", json={"path": path})
    assert r.status_code == 200


def test_inspect_rejects_missing_file(tmp_path):
    r = client.post("/v1/models/inspect", json={"path": str(tmp_path / "nope.safetensors")})
    assert r.status_code == 400


def test_inspect_rejects_non_safetensors(tmp_path):
    p = tmp_path / "model.bin"
    p.write_bytes(b"\x00")
    r = client.post("/v1/models/inspect", json={"path": str(p)})
    assert r.status_code == 400


def test_inspect_rejects_truncated_header(tmp_path):
    p = tmp_path / "bad.safetensors"
    p.write_bytes(struct.pack("<Q", 9999) + b"short")
    r = client.post("/v1/models/inspect", json={"path": str(p)})
    assert r.status_code == 400


def test_inspect_rejects_malformed_json(tmp_path):
    p = tmp_path / "bad.safetensors"
    junk = b"not json!!!"
    p.write_bytes(struct.pack("<Q", len(junk)) + junk)
    r = client.post("/v1/models/inspect", json={"path": str(p)})
    assert r.status_code == 400


# --- slice ---

def test_slice_returns_correct_values(model_file):
    path, arr = model_file
    client.post("/v1/models/inspect", json={"path": path})
    r = client.get("/v1/tensor-slice?tensor=layer.weight&row=0&column=0&rows=4&columns=4")
    assert r.status_code == 200
    data = r.json()
    assert data["origin"] == [0, 0]
    assert data["shape"] == [4, 4]
    values = np.array(data["values"])
    np.testing.assert_allclose(values, arr.astype(np.float64), rtol=1e-5)


def test_slice_first_window(model_file):
    path, arr = model_file
    client.post("/v1/models/inspect", json={"path": path})
    r = client.get("/v1/tensor-slice?tensor=layer.weight&row=0&column=0&rows=2&columns=2")
    assert r.status_code == 200
    assert r.json()["shape"] == [2, 2]


def test_slice_final_window(model_file):
    path, arr = model_file
    client.post("/v1/models/inspect", json={"path": path})
    r = client.get("/v1/tensor-slice?tensor=layer.weight&row=2&column=2&rows=4&columns=4")
    assert r.status_code == 200
    assert r.json()["shape"] == [2, 2]  # clamped to boundary


def test_slice_out_of_bounds(model_file):
    path, _ = model_file
    client.post("/v1/models/inspect", json={"path": path})
    r = client.get("/v1/tensor-slice?tensor=layer.weight&row=100&column=0&rows=4&columns=4")
    assert r.status_code == 400


def test_slice_exceeds_element_limit(model_file):
    path, _ = model_file
    client.post("/v1/models/inspect", json={"path": path})
    r = client.get("/v1/tensor-slice?tensor=layer.weight&row=0&column=0&rows=16&columns=16")
    assert r.status_code == 422  # query param max=16 triggers validation


def test_slice_rejects_nan(tmp_path):
    arr = np.full((4, 4), float("nan"), dtype=np.float32)
    path = str(tmp_path / "nan.safetensors")
    write_safetensors({"layer.weight": arr}, path)
    client.post("/v1/models/inspect", json={"path": path})
    r = client.get("/v1/tensor-slice?tensor=layer.weight&row=0&column=0&rows=4&columns=4")
    assert r.status_code == 500


def test_slice_no_model_attached():
    import main
    main._attached_path = None
    main._tensor_index = {}
    r = client.get("/v1/tensor-slice?tensor=anything&row=0&column=0")
    assert r.status_code == 400


def test_source_file_unchanged(model_file):
    import hashlib
    path, _ = model_file
    with open(path, "rb") as f:
        before = hashlib.sha256(f.read()).hexdigest()
    client.post("/v1/models/inspect", json={"path": path})
    client.get("/v1/tensor-slice?tensor=layer.weight&row=0&column=0&rows=4&columns=4")
    with open(path, "rb") as f:
        after = hashlib.sha256(f.read()).hexdigest()
    assert before == after
