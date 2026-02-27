import hashlib
import tempfile
import json 

import aiofiles

from pathlib import Path
from fastapi import UploadFile
from langsmith import traceable


# ─────────────────────────────────────────
# FILE FINGERPRINT
# ─────────────────────────────────────────
def _file_fingerprint(path: str)-> dict:
    p = Path(path)
    h = hashlib.sha256()

    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    stat = p.stat()

    return {
        "sha256": h.hexdigest(),
        "size": stat.st_size, 
    }        

def _index_key(
        pdf_path: str, 
        chunk_size: int, 
        chunk_overlap: int, 
        embed_model_name: str
    ) -> str:

    fingerprint = _file_fingerprint(pdf_path)

    meta = {
        "pdf_fingerprint": fingerprint["sha256"],
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": embed_model_name,
        "format": "v1",
    }

    return hashlib.sha256(
        json.dumps(meta, sort_keys=True).encode("utf-8")
        ).hexdigest()


# ─────────────────────────────────────────
# Load & split PDF
# ─────────────────────────────────────────
@traceable(name="load_pdf")
async def pdf_load(upload: UploadFile) -> str:
    """Save uploaded PDF and return temp path."""

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp_path = tmp.name

    async with aiofiles.open(tmp_path, "wb") as f:
       # reads the upload in 1MB chunks and writes each chunk to a file
       while chunk := await upload.read(1024 * 1024):
        await f.write(chunk)

    return tmp_path  