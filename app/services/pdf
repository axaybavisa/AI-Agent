import hashlib
import tempfile
import json 

import aiofiles

from pathlib import Path
from fastapi import UploadFile
from typing import Dict
from langsmith import traceable


# ─────────────────────────────────────────
# FILE FINGERPRINT
# ─────────────────────────────────────────
class IndexKeyGenerator:
    
    def __init__(
            self,
            pdf_path: str,
            chunk_size: int,
            chunk_overlap: int,
            embed_model_name: str
    ):
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embed_model_name = embed_model_name

    def _file_fingerprint(self)-> Dict[str, str]:
        p = Path(self.pdf_path)
        h = hashlib.sha256()

        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)

        stat = p.stat()

        return {
            "sha256": h.hexdigest(),
            "size": stat.st_size, 
        }

    def generate_index_key(self) -> str:

        fingerprint = self._file_fingerprint()

        meta = {
            "pdf_fingerprint": fingerprint["sha256"],
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model": self.embed_model_name,
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