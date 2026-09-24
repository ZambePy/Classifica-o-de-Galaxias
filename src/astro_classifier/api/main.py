"""API de inferencia (FastAPI).

Dois modos, escolhidos por ASTRO_API_MODE:

  mock  - responde sem modelo nenhum. E o modo do dia 1: permite que o
          dashboard seja construido em paralelo ao treino.
  real  - carrega os checkpoints da cascata e classifica de verdade.

Subir:  uvicorn astro_classifier.api.main:app --reload
Docs:   http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import io
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from astro_classifier import __version__
from astro_classifier.api.mock import mock_prediction
from astro_classifier.api.schemas import (
    CONTRACT_VERSION,
    HealthResponse,
    PredictionResponse,
)
from astro_classifier.config import resolve_device

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB

app = FastAPI(
    title="GALAXIA - API de classificacao astronomica",
    version=__version__,
    description=(
        "Classificacao hierarquica de objetos astronomicos a partir de imagens. "
        f"Versao do contrato: {CONTRACT_VERSION}."
    ),
)

# O dashboard roda em outra porta (ex: Vite em :5173). Sem CORS, o navegador
# bloqueia a chamada. Em producao, restrinja allow_origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _mode() -> str:
    return os.getenv("ASTRO_API_MODE", "mock").lower()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Usado pelo dashboard para saber se o backend esta de pe e em que modo."""
    mode = _mode()
    models: list[str] = []
    if mode == "real":
        from astro_classifier.inference.pipeline import get_pipeline

        models = get_pipeline().loaded_levels()
    return HealthResponse(
        status="ok",
        contract_version=CONTRACT_VERSION,
        mode=mode,
        device=resolve_device(),
        models_loaded=models,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """Classifica uma imagem. Campo multipart obrigatorio: `file`."""
    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Arquivo vazio.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Imagem maior que o limite de {MAX_UPLOAD_BYTES // (1024*1024)} MB.",
        )

    # Valida que e mesmo uma imagem antes de qualquer processamento.
    try:
        image = Image.open(io.BytesIO(content))
        image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=415, detail="Arquivo nao reconhecido como imagem."
        ) from exc

    if _mode() == "mock":
        return mock_prediction(content)

    from astro_classifier.inference.pipeline import get_pipeline

    return get_pipeline().predict_bytes(content)
