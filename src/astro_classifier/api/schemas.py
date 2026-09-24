"""Contrato da API - a fronteira entre o lado de IA e o lado de dashboard.

Este arquivo e o documento mais importante do projeto para a dupla: enquanto
ele nao mudar, os dois lados evoluem sem se quebrar. Mudou aqui? Avise o
outro e suba a versao em `contract_version`.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

CONTRACT_VERSION = "1.0"


class ClassScore(BaseModel):
    """Uma classe e sua probabilidade."""

    label: str = Field(..., description="Nome interno da classe, ex: 'spiral'")
    label_pt: str = Field(..., description="Nome legivel em portugues, para exibir")
    probability: float = Field(..., ge=0.0, le=1.0)


class LevelPrediction(BaseModel):
    """Saida de um dos modelos da cascata."""

    level: str = Field(..., description="object | galaxy | nebula")
    predicted: str = Field(..., description="Classe vencedora")
    predicted_pt: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    scores: list[ClassScore] = Field(..., description="Todas as classes, ordenadas por probabilidade")
    model_version: str = Field(..., description="Versao do checkpoint que produziu isto")


class DomainCheck(BaseModel):
    """Aviso de dominio: a imagem se parece com o que o modelo viu no treino?

    Os modelos sao treinados em recortes de surveys (SDSS/DSS). Uma
    astrofoto processada do Hubble e outro dominio - o resultado pode ser
    confiante e errado ao mesmo tempo. Quando `out_of_domain` e True o
    dashboard DEVE exibir o aviso, nao apenas o rotulo.
    """

    out_of_domain: bool
    score: float = Field(..., description="Confianca maxima observada (MSP)")
    threshold: float = Field(..., description="Limiar calibrado em dados fora do dominio")
    message_pt: str


class PredictionResponse(BaseModel):
    """Resposta completa de POST /predict."""

    contract_version: str = CONTRACT_VERSION
    mock: bool = Field(False, description="True quando a API responde sem modelo real carregado")

    # Resumo plano - o que o dashboard mostra em destaque.
    object: str = Field(..., description="galaxy | nebula | other")
    confidence: float = Field(..., ge=0.0, le=1.0)
    subtype: str | None = Field(None, description="None quando object == 'other'")
    subtype_confidence: float | None = None
    summary_pt: str = Field(..., description="Frase pronta para exibir, ex: 'Galaxia espiral'")

    # Detalhe por nivel - para quem quiser mostrar o grafico completo.
    levels: list[LevelPrediction]

    domain: DomainCheck
    inference_ms: float = Field(..., description="Tempo de inferencia em milissegundos")


class HealthResponse(BaseModel):
    status: str
    contract_version: str
    mode: str = Field(..., description="mock | real")
    device: str
    models_loaded: list[str]


class ErrorResponse(BaseModel):
    error: str
    detail: str
