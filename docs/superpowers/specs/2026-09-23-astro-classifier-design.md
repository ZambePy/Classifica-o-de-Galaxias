# Classificador hierárquico de objetos astronômicos — design

**Data:** 2026-09-23
**Status:** aprovado, em implementação
**Equipe:** dupla — um lado de IA/dados/treino, outro de dashboard/interface

---

## 1. Objetivo

Construir um sistema que recebe uma imagem e responde **que tipo de objeto
astronômico** ela contém, com um segundo nível de detalhe (morfologia da
galáxia, tipo da nebulosa) e um **aviso explícito** quando a imagem está fora
do domínio em que o modelo foi treinado.

Contexto: projeto acadêmico de dois estudantes de curso técnico de IA, feito
por interesse próprio e para currículo, com a possibilidade de evoluir para
um produto. Sem prazo rígido — o que favorece investir em reprodutibilidade e
em uma estrutura que aguente crescer.

### Critérios de sucesso

1. Um modelo treinado, avaliado e **reprodutível** por nível da cascata.
2. Métricas por classe (não só acurácia) e matriz de confusão para cada um.
3. API estável que o dashboard consome sem conhecer nada de PyTorch.
4. Dataset de nebulosas construído a partir de catálogos citáveis.
5. Sistema que admite não saber, em vez de sempre dar uma resposta confiante.

### Fora de escopo

Competir com modelos astronômicos especializados; detecção/segmentação (a
entrada é uma imagem com um objeto, não um campo a ser varrido); dados
espectrais; treino distribuído.

---

## 2. Decisão de arquitetura: três modelos em cascata

```
                      imagem
                        │
                  ┌─────▼──────┐
                  │  NÍVEL 1   │  galaxy / nebula / other
                  └─────┬──────┘
              ┌─────────┴─────────┐
        ┌─────▼────┐        ┌─────▼─────┐
        │ NÍVEL 2a │        │ NÍVEL 2b  │
        │ spiral   │        │ emission  │
        │ elliptic │        │ reflection│
        │ irregular│        │ planetary │
        └──────────┘        │ snr       │
                            └───────────┘
        ("other" encerra a cascata — não tem subtipo)
```

### Alternativas consideradas

| Abordagem | Por que não foi escolhida agora |
|---|---|
| **B — backbone único, três cabeças** (multi-tarefa) | Exige loss mascarada e enfrenta desbalanceamento entre tarefas (61k galáxias × ~3k nebulosas). Vira a Fase 2 e um experimento comparativo. |
| **C — modelo plano de 8 classes** | Descarta a estrutura hierárquica, que é o valor acadêmico do projeto, e agrava o desbalanceamento. |

### Por que a cascata

- **Datasets heterogêneos.** Cada nível recebe a estratégia que precisa (pesos
  de classe, augmentation, resolução, congelamento de backbone) sem contaminar
  os outros. O `level3_nebula.yaml` é visivelmente diferente do
  `level2_galaxy.yaml`, e isso é a decisão funcionando.
- **Trabalho incremental.** O modelo de galáxias fica pronto enquanto a coleta
  de nebulosas ainda roda. Nada trava nada — decisivo para uma dupla.
- **Depuração.** Três modelos pequenos e legíveis, em vez de uma loss mascarada.

### Custo assumido: erro em cascata

Uma galáxia classificada como nebulosa no nível 1 recebe um subtipo de
nebulosa e fica **duplamente errada**. Isso não é contornado — é **medido e
reportado**. A resposta da API expõe `levels` com o detalhe de cada etapa,
para que o dashboard mostre onde a decisão foi tomada. A avaliação de ponta a
ponta (acurácia da folha) deve aparecer no trabalho ao lado da acurácia por
nível.

---

## 3. Domínio e o aviso de fora-de-domínio

**O problema.** Os modelos treinam em recortes de levantamentos (DSS2/SDSS):
imagens fracas, fundo escuro, objeto pequeno e centralizado. O dashboard
aceita upload livre — alguém vai enviar uma astrofoto processada do Hubble,
uma selfie, ou um print de tela.

Uma rede com softmax **não sabe dizer "não sei"**. Ela distribui 100% entre as
classes que conhece. Sem detector, o sistema mente com confiança.

**Solução adotada:** MSP — *Maximum Softmax Probability* (Hendrycks & Gimpel,
ICLR 2017). O limiar é calibrado no conjunto de **validação**, escolhido para
manter 95% das imagens legítimas acima dele. Abaixo do limiar, a resposta vem
com `out_of_domain: true` e o dashboard exibe o aviso.

O trade-off é explícito e deve estar no texto: aceitamos marcar 5% das imagens
boas como suspeitas para pegar as ruins.

**Evolução (Fase 2):** energy-based (Liu et al., 2020) e Mahalanobis (Lee et
al., 2018). Comparar os três contra o baseline MSP é um experimento pronto.

---

## 4. Componentes

| Módulo | Responsabilidade | Depende de |
|---|---|---|
| `taxonomy.py` | classes e hierarquia — fonte única de verdade | nada |
| `paths.py` | dados fora do OneDrive, sob uma raiz configurável | `.env` |
| `config.py` | configuração de experimento em YAML | `taxonomy` |
| `data/catalogs.py` | catálogos VizieR → (nome, ra, dec, tipo) | astroquery |
| `data/cutouts.py` | coordenada → imagem, via hips2fits | requests |
| `data/splits.py` | divisão estratificada e reprodutível | sklearn |
| `data/datasets.py` | Dataset do PyTorch a partir de CSV | torch |
| `models/` | backbone + cabeça linear; checkpoint auto-descritivo | torchvision |
| `training/loops.py` | loop de treino, early stopping, AMP | torch |
| `evaluation/` | métricas, matriz de confusão, Grad-CAM | sklearn, matplotlib |
| `ood/msp.py` | detector de fora-de-domínio | numpy |
| `inference/pipeline.py` | **a cascata** — junta tudo | todos acima |
| `api/` | contrato HTTP e modo mock | fastapi |

**Regra de acoplamento:** `api/` não conhece PyTorch. `taxonomy.py` não conhece
nada. A cascata é o único ponto onde os três modelos se encontram.

---

## 5. Decisões de dados

- **Rotação livre (0–360°) como augmentation.** No céu não existe "em pé" — a
  orientação de uma galáxia na imagem é arbitrária, definida pela posição do
  telescópio. É uma simetria real do problema, não ruído artificial.
- **Cor pouco alterada.** Em nebulosas a cor carrega sinal físico (H-α
  avermelhado em emissão, azul espalhado em reflexão). Distorcer saturação e
  matiz apagaria a feature mais discriminativa. Só brilho e contraste leves.
- **Macro-F1 como métrica de seleção.** Com classes desbalanceadas, um modelo
  que só chuta a majoritária tem acurácia alta e macro-F1 baixo. Selecionar
  por macro-F1 impede que esse modelo inútil seja salvo.
- **Ambíguas descartadas.** No Galaxy Zoo, galáxias sem consenso entre os
  voluntários saem do dataset em vez de serem forçadas numa classe.
- **Checkpoints auto-descritivos.** Pesos + config + lista de classes no mesmo
  arquivo, com verificação ao carregar. Evita o pior bug possível: mudar a
  ordem das classes e receber predições silenciosamente trocadas.

---

## 6. Contrato com o dashboard

API REST (FastAPI), com **modo mock desde o dia 1**: a API responde no formato
final, com dados sintéticos determinísticos, antes de qualquer modelo existir.
O dashboard é construído em paralelo ao treino.

Detalhes em [`docs/api.md`](../../api.md). O contrato é versionado
(`CONTRACT_VERSION`) e coberto por testes que falham se um campo mudar sem
intenção.

---

## 7. Plano de execução

**Fase 1 — infraestrutura** *(concluída)*
Estrutura, config, API mock, testes, CI.

**Fase 2 — dados**
Galaxy Zoo baixado e rotulado; nebulosas coletadas dos catálogos; classe
`other` montada; **inspeção visual de amostras**; splits gerados.

**Fase 3 — baselines**
Treinar os três níveis com ResNet18. Métricas por classe, matrizes de
confusão, curvas.

**Fase 4 — cascata completa**
Avaliação de ponta a ponta, medição do erro em cascata, calibração do detector
de domínio contra um conjunto realmente fora do domínio, Grad-CAM no
dashboard.

**Fase 5 — estudo comparativo** *(o trabalho acadêmico)*
Backbones (ResNet18 × EfficientNet-B0 × ResNet50); cascata (A) × multi-tarefa
(B); rótulo majoritário × soft labels; MSP × energy × Mahalanobis.

---

## 8. Riscos

| Risco | Gravidade | Mitigação |
|---|---|---|
| Coleta de nebulosas rende poucas imagens ou de má qualidade | **alta** | `--inspect` e `--limit` antes do download completo; inspeção visual obrigatória; ajuste de FOV por classe |
| IDs do VizieR mudarem entre edições | média | `inspect_catalog()`; falha de um catálogo não aborta os outros |
| `irregular` e `reflection` com F1 muito baixo por falta de dados | média | `min_per_class` remove classes inviáveis com aviso; reportar honestamente |
| Modelo aprender atalho (fundo, artefato) em vez do objeto | média | Grad-CAM na avaliação, não só no dashboard |
| 6 GB de VRAM limitarem os experimentos | baixa | AMP ligado; batch menor para backbones grandes |
| Dados dentro do OneDrive travarem a máquina | **alta** | `ASTRO_DATA_ROOT` fora da pasta sincronizada; `data/` no `.gitignore` |

---

## 9. O que este projeto deliberadamente não faz

Não tenta bater modelos astronômicos especializados. O objetivo é um
**baseline reprodutível e honesto**, documentado a ponto de outra pessoa
repetir os números — e, a partir dele, comparações controladas. Um baseline
sólido com limitações declaradas vale academicamente mais que um número alto
sem procedência.
