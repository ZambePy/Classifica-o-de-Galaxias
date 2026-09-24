"""Testes do contrato da API.

Estes testes sao o que garante que o dashboard do seu colega nao quebra. Se
alguem mudar um campo da resposta sem avisar, eles falham aqui - no seu
computador, antes do commit, e nao na apresentacao.

Rodam inteiramente em modo mock: nao precisam de GPU, checkpoint nem
dataset. Por isso funcionam tambem no CI do GitHub.
"""

from __future__ import annotations

import io
import os

import pytest
from fastapi.testclient import TestClient
from PIL import Image

os.environ["ASTRO_API_MODE"] = "mock"

from astro_classifier.api.main import app  # noqa: E402
from astro_classifier.api.schemas import CONTRACT_VERSION  # noqa: E402
from astro_classifier.taxonomy import GALAXY_CLASSES, NEBULA_CLASSES, OBJECT_CLASSES  # noqa: E402


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def make_image(color: tuple[int, int, int] = (20, 20, 60), size: int = 128) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (size, size), color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_responde_ok(client):
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["contract_version"] == CONTRACT_VERSION
    assert body["mode"] == "mock"


def test_predict_devolve_o_contrato_completo(client):
    response = client.post("/predict", files={"file": ("teste.png", make_image(), "image/png")})
    assert response.status_code == 200

    body = response.json()
    assert body["mock"] is True
    assert body["object"] in OBJECT_CLASSES
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["summary_pt"]
    assert body["inference_ms"] > 0

    # O aviso de dominio nunca pode faltar - o dashboard depende dele.
    assert set(body["domain"]) == {"out_of_domain", "score", "threshold", "message_pt"}
    assert isinstance(body["domain"]["out_of_domain"], bool)


def test_subtipo_coerente_com_o_objeto(client):
    """Regra da cascata: 'other' nao tem subtipo; galaxia so recebe subtipo de
    galaxia; nebulosa so recebe subtipo de nebulosa."""
    for i in range(30):
        image = make_image(color=(i * 7 % 256, i * 13 % 256, i * 29 % 256))
        body = client.post("/predict", files={"file": ("t.png", image, "image/png")}).json()

        if body["object"] == "other":
            assert body["subtype"] is None
        elif body["object"] == "galaxy":
            assert body["subtype"] in GALAXY_CLASSES
        else:
            assert body["subtype"] in NEBULA_CLASSES


def test_probabilidades_somam_um_e_vem_ordenadas(client):
    body = client.post("/predict", files={"file": ("t.png", make_image(), "image/png")}).json()

    for level in body["levels"]:
        probs = [s["probability"] for s in level["scores"]]
        assert probs == sorted(probs, reverse=True), "scores devem vir do maior para o menor"
        assert sum(probs) == pytest.approx(1.0, abs=1e-6)
        assert level["predicted"] == level["scores"][0]["label"]
        assert level["confidence"] == pytest.approx(level["scores"][0]["probability"])


def test_mesma_imagem_devolve_mesma_resposta(client):
    """O mock e deterministico de proposito: permite que o dashboard tenha
    testes estaveis contra a API antes do modelo existir."""
    image = make_image()
    first = client.post("/predict", files={"file": ("t.png", image, "image/png")}).json()
    second = client.post("/predict", files={"file": ("t.png", image, "image/png")}).json()

    assert first["object"] == second["object"]
    assert first["subtype"] == second["subtype"]
    assert first["confidence"] == pytest.approx(second["confidence"])


def test_arquivo_que_nao_e_imagem_e_recusado(client):
    response = client.post("/predict", files={"file": ("nota.txt", b"isto nao e uma imagem", "text/plain")})
    assert response.status_code == 415


def test_arquivo_vazio_e_recusado(client):
    response = client.post("/predict", files={"file": ("vazio.png", b"", "image/png")})
    assert response.status_code == 400


def test_resumo_em_portugues_combina_objeto_e_subtipo():
    from astro_classifier.api.mock import build_summary_pt

    assert build_summary_pt("galaxy", "spiral") == "Galaxia espiral"
    assert build_summary_pt("nebula", "planetary") == "Nebulosa planetaria"
    assert build_summary_pt("other", None) == "Outro objeto"
