# GALAXIA — classificador hierárquico de objetos astronômicos

Sistema de visão computacional que recebe a imagem de um objeto no céu e
responde **o que ele é** — galáxia, nebulosa ou outra coisa — e, quando é
galáxia ou nebulosa, **de que tipo**. Quando a imagem não se parece com nada
que o modelo aprendeu, ele diz isso em vez de inventar uma resposta.

```
                            imagem
                              │
                     ┌────────▼────────┐
                     │     NÍVEL 1     │
                     │ galáxia/nebulosa│
                     │      /outro     │
                     └────────┬────────┘
                 ┌────────────┴────────────┐
          ┌──────▼──────┐           ┌──────▼──────┐
          │  NÍVEL 2a   │           │  NÍVEL 2b   │
          │  espiral    │           │  emissão    │
          │  elíptica   │           │  reflexão   │
          │  irregular  │           │  planetária │
          └─────────────┘           │  remanesc.  │
                                    └─────────────┘
```

Projeto acadêmico de dois estudantes de curso técnico de IA. O objetivo não é
competir com modelos astronômicos especializados, e sim construir um
**baseline reprodutível e honesto**, documentado a ponto de outra pessoa
repetir os números.

---

## Índice

- [O que torna este projeto diferente](#o-que-torna-este-projeto-diferente)
- [Começando em 5 minutos](#começando-em-5-minutos)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Como o projeto funciona](#como-o-projeto-funciona)
- [Fluxo completo de trabalho](#fluxo-completo-de-trabalho)
- [Divisão de trabalho na dupla](#divisão-de-trabalho-na-dupla)
- [Decisões técnicas e o porquê](#decisões-técnicas-e-o-porquê)
- [Limitações conhecidas](#limitações-conhecidas)
- [Roteiro](#roteiro)
- [Referências](#referências)

---

## O que torna este projeto diferente

**1. O dataset de nebulosas é construído, não baixado.**
Para galáxias existe o Galaxy Zoo. Para nebulosas não existe equivalente. O
projeto monta o seu a partir de **catálogos astronômicos publicados** (listas
de objetos com coordenadas, compiladas por astrônomos ao longo de décadas) e
de um **serviço de recorte do céu**:

```
catálogo (nome, ra, dec, tipo)  →  hips2fits  →  imagem rotulada
```

**2. O sistema pode dizer "não sei".**
Uma rede com softmax sempre distribui 100% entre as classes que conhece — uma
foto da Lua vira "nebulosa planetária, 97%". Aqui há um detector de
fora-de-domínio que marca a resposta como não confiável quando a imagem não se
parece com os dados de treino.

**3. A hierarquia é real, não cosmética.**
Três modelos independentes, cada um com seu dataset e seus hiperparâmetros. O
erro em cascata que isso gera é medido e reportado, não escondido.

---

## Começando em 5 minutos

**Pré-requisitos:** Python 3.10+, Git. GPU NVIDIA é recomendada para treinar
(o projeto foi desenvolvido numa GTX 1660 Ti de 6 GB), mas **não é necessária**
para rodar a API em modo mock nem os testes.

```powershell
git clone <url-do-repositorio>
cd "REDE NEURAL GALAXIA"

powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
```

O script cria o `.venv`, instala o PyTorch com CUDA, instala as dependências,
cria o `.env` e monta a árvore de dados. Em Linux/macOS: `bash scripts/setup_env.sh`.

Confirme que está tudo de pé:

```powershell
.\.venv\Scripts\Activate.ps1
pytest
```

Suba a API (modo mock — responde sem nenhum modelo treinado):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\serve_api.ps1
```

Abra <http://127.0.0.1:8000/docs> e faça upload de uma imagem qualquer. Você
recebe uma resposta completa, no formato final, imediatamente.

> **Por que um modo mock?** Porque o dashboard pode ser construído por inteiro
> **antes** de existir qualquer modelo treinado. Numa dupla, essa é a maior
> alavanca de produtividade que existe: ninguém fica esperando ninguém.

---

## Estrutura do repositório

```
REDE NEURAL GALAXIA/
├── configs/                      um YAML por modelo da cascata
│   ├── level1_object.yaml        galáxia / nebulosa / outro
│   ├── level2_galaxy.yaml        morfologia de galáxia
│   └── level3_nebula.yaml        tipo de nebulosa
│
├── src/astro_classifier/
│   ├── taxonomy.py               ⭐ classes e hierarquia — fonte da verdade
│   ├── config.py                 configuração de experimento (YAML)
│   ├── paths.py                  onde cada coisa mora no disco
│   │
│   ├── data/
│   │   ├── catalogs.py           catálogos VizieR → coordenadas
│   │   ├── cutouts.py            coordenadas → imagens (hips2fits)
│   │   ├── datasets.py           Dataset do PyTorch
│   │   ├── transforms.py         pré-processamento e augmentation
│   │   └── splits.py             divisão train/val/test estratificada
│   │
│   ├── models/                   backbone + cabeça linear
│   ├── training/loops.py         loop de treino, early stopping, AMP
│   ├── evaluation/               métricas, matriz de confusão, Grad-CAM
│   ├── ood/msp.py                detector de fora-de-domínio
│   ├── inference/pipeline.py     ⭐ a cascata em funcionamento
│   └── api/                      contrato HTTP + modo mock
│
├── scripts/                      tudo que se roda pela linha de comando
├── tests/                        33 testes, sem GPU, rodam em segundos
├── docs/
│   ├── api.md                    ⭐ contrato para quem faz o dashboard
│   ├── datasets.md               ⭐ de onde vem cada imagem
│   └── superpowers/specs/        o design do sistema
└── notebooks/                    exploração e inspeção visual
```

**Os dados não ficam aqui.** Imagens, checkpoints e resultados vivem em
`ASTRO_DATA_ROOT` (padrão `C:/astro-data`), configurado no `.env`.

> ⚠️ **Se este repositório estiver dentro do OneDrive**, isso não é opcional.
> O Galaxy Zoo sozinho tem ~61 mil imagens; dentro de uma pasta sincronizada, o
> cliente do OneDrive tenta enviar cada arquivo e degrada a máquina inteira.

---

## Como o projeto funciona

### A cascata

O nível 1 decide o tipo de objeto. **O resultado dele escolhe qual modelo do
nível 2 roda.** Se o nível 1 disser `other`, a cascata para — não existe
subtipo de "outro".

Isso está implementado em `inference/pipeline.py`, e a resposta da API expõe
cada etapa separadamente (campo `levels`), para que o dashboard mostre onde a
decisão foi tomada.

### O aviso de domínio

Os modelos são treinados em **recortes de levantamentos astronômicos**
(DSS2/SDSS): imagens fracas, fundo escuro, objeto pequeno no centro. O
dashboard, porém, aceita upload livre.

```
   TREINO (DSS2/SDSS)          USO REAL (Hubble)
     256×256, fraca              4000×4000, vibrante
     fundo escuro                cores saturadas
     objeto pequeno              objeto preenche o quadro
                    └──── domain shift ────┘
```

O detector (MSP) compara a confiança máxima do nível 1 com um limiar calibrado
na validação. Abaixo do limiar, a resposta vem com `out_of_domain: true` e o
dashboard exibe o aviso junto do resultado.

### Grad-CAM — onde o modelo olhou

`evaluation/gradcam.py` gera o mapa de calor das regiões que sustentaram a
predição. Não é enfeite: **é como você descobre que o modelo aprendeu a coisa
errada.** Se o mapa de uma "galáxia espiral" acende no ruído do fundo em vez
dos braços espirais, sua acurácia alta é um artefato do dataset.

---

## Fluxo completo de trabalho

### 1. Dados

```powershell
# Nebulosas — SEMPRE inspecione os catálogos antes de baixar milhares de imagens
python scripts\build_nebula_dataset.py --inspect
python scripts\build_nebula_dataset.py --limit 20     # teste
python scripts\build_nebula_dataset.py                # completo (leva horas)

# Galáxias — baixe o Galaxy Zoo do Kaggle primeiro, depois:
python scripts\prepare_galaxy_zoo.py `
    --solutions C:\astro-data\raw\galaxy_zoo\training_solutions_rev1.csv `
    --images    C:\astro-data\raw\galaxy_zoo\images_training_rev1 `
    --copy
```

> **Depois de baixar, olhe as imagens.** Recortes vazios e campo de visão
> errado são comuns e envenenam o treino em silêncio. Um notebook que mostra 50
> imagens por classe numa grade vale mais que confiar no catálogo.
> Detalhes em [`docs/datasets.md`](docs/datasets.md).

### 2. Splits

```powershell
python scripts\make_splits.py --level object --cap-per-class 4000
python scripts\make_splits.py --level galaxy
python scripts\make_splits.py --level nebula
```

O teto no nível 1 evita que dezenas de milhares de galáxias afoguem alguns
milhares de nebulosas.

### 3. Treino

```powershell
python scripts\train.py --config configs\level1_object.yaml
python scripts\train.py --config configs\level2_galaxy.yaml
python scripts\train.py --config configs\level3_nebula.yaml
```

Cada execução grava em `ASTRO_DATA_ROOT/runs/<nome>/`: a config exata usada,
métricas por época, curvas de treino e a matriz de confusão. O melhor
checkpoint (por macro-F1) vai para `checkpoints/`.

### 4. Avaliação e calibração

```powershell
python scripts\evaluate.py --config configs\level1_object.yaml
```

Mede no conjunto de **teste** e, para o nível 1, calibra o detector de domínio.

> **Não toque no conjunto de teste antes do fim.** Se você ajustar
> hiperparâmetros olhando o teste, ele deixa de medir generalização — e o
> número que você publicar estará inflado.

### 5. API com modelos reais

```powershell
powershell -ExecutionPolicy Bypass -File scripts\serve_api.ps1 -Mode real
```

---

## Divisão de trabalho na dupla

```
  LADO DE IA / DADOS              LADO DE DASHBOARD
  ─────────────────               ─────────────────
  coleta de dados                 upload de imagem
  treino dos modelos              gráficos de probabilidade
  avaliação e métricas            aviso de fora-de-domínio
  calibração do detector          visualização do Grad-CAM
          │                              │
          └────────► POST /predict ◄─────┘
                  contrato v1.0
                  docs/api.md
```

**O contrato é o acordo.** Enquanto ele não mudar, cada lado evolui sozinho sem
quebrar o outro — e como a API responde em modo mock desde o primeiro dia, o
dashboard não espera o treino terminar.

Se precisar mudar um campo: avise a outra pessoa, suba `CONTRACT_VERSION` em
`api/schemas.py` e atualize `docs/api.md`. Os testes em
`tests/test_api_contract.py` falham até que a mudança seja consciente.

---

## Decisões técnicas e o porquê

| Decisão | Por quê |
|---|---|
| **Três modelos independentes**, não um multi-tarefa | Os datasets são muito diferentes em tamanho e origem; separados, cada um recebe a estratégia que precisa. E o modelo de galáxias fica pronto enquanto a coleta de nebulosas ainda roda. |
| **Transfer learning**, não CNN do zero | Com ~3 mil imagens de nebulosa, uma rede do zero decora o treino. Features de baixo nível do ImageNet transferem bem mesmo para imagens de telescópio. |
| **Rotação livre 0–360°** no augmentation | No céu não existe "em pé" — a orientação é definida pela posição do telescópio. É uma simetria real do problema. |
| **Cor quase intocada** no augmentation | A cor carrega sinal físico (H-α avermelhado em emissão, azul espalhado em reflexão). Distorcê-la apagaria a feature mais discriminativa. |
| **Macro-F1**, não acurácia, para escolher o modelo | Com classes desbalanceadas, chutar sempre a majoritária dá acurácia alta e macro-F1 baixo. |
| **Galáxias ambíguas descartadas** | No Galaxy Zoo, quando os voluntários não concordam, o rótulo é ruído. Descartar encolhe o dataset e melhora a qualidade. |
| **Checkpoints auto-descritivos** | Pesos + config + classes no mesmo arquivo, verificados ao carregar. Evita o pior bug do projeto: mudar a ordem das classes e receber predições silenciosamente trocadas. |
| **Precisão mista (AMP)** ligada | Nos 6 GB da GTX 1660 Ti, corta a memória pela metade e acelera o treino. |
| **Dados fora do repositório** | Sincronização e versionamento de 61 mil imagens travam a máquina e estouram qualquer cota. |

---

## Limitações conhecidas

Declaradas de propósito — um baseline com limitações honestas vale mais que um
número alto sem procedência.

- **Erro em cascata.** Uma galáxia classificada como nebulosa no nível 1
  recebe um subtipo de nebulosa e fica duplamente errada. Isso é medido e
  reportado, não contornado.
- **`irregular` é a classe mais frágil.** O Galaxy Zoo 2 não tem pergunta
  direta para "irregular"; a aproximação usada mistura galáxias irregulares
  com fusões, anéis e artefatos.
- **Nebulosas de reflexão são poucas.** O catálogo de van den Bergh tem ~150
  objetos. Pode não dar para treinar essa classe de forma confiável.
- **Domínio restrito.** Treinado em recortes do DSS2/SDSS. Astrofotos
  processadas são outro domínio — daí o detector existir.
- **MSP é o baseline mais simples** de detecção de fora-de-domínio. Funciona,
  mas há métodos melhores (energy-based, Mahalanobis) previstos para a Fase 2.
- **Campos de visão por classe são um chute inicial informado**, ajustável
  depois de inspecionar as imagens.

---

## Roteiro

- [x] **Fase 1 — infraestrutura**: estrutura, config, API mock, testes, CI
- [ ] **Fase 2 — dados**: Galaxy Zoo, coleta de nebulosas, classe `other`,
      inspeção visual, splits
- [ ] **Fase 3 — baselines**: treinar os três níveis com ResNet18
- [ ] **Fase 4 — cascata completa**: avaliação ponta a ponta, erro em cascata,
      calibração do detector, Grad-CAM no dashboard
- [ ] **Fase 5 — estudo comparativo** *(o trabalho acadêmico)*:
  - backbones: ResNet18 × EfficientNet-B0 × ResNet50
  - cascata × multi-tarefa (backbone compartilhado)
  - rótulo majoritário × **soft labels** (treinar com a distribuição de votos
    dos voluntários — a pergunta de pesquisa mais interessante do projeto)
  - MSP × energy-based × Mahalanobis

---

## Desenvolvimento

```powershell
pytest                              # 33 testes, sem GPU, ~4 segundos
ruff check src tests scripts        # lint
```

Os testes cobrem taxonomia, contrato da API, splits e detector de domínio —
tudo que não precisa de GPU. O CI do GitHub roda os dois a cada push, sem
instalar o PyTorch (rápido e suficiente).

---

## Referências

**Dados**
- Galaxy Zoo 2 — Willett et al. (2013), *MNRAS* 435, 2835
- Sharpless (1959), *Catalogue of HII Regions*, ApJS 4, 257
- van den Bergh (1966), *Catalogue of Reflection Nebulae*, AJ 71, 990
- Acker et al. (1992), *Strasbourg-ESO Catalogue of Galactic Planetary Nebulae*
- Green (2009), *A Catalogue of Galactic Supernova Remnants*
- VizieR / hips2fits — CDS, Observatoire astronomique de Strasbourg

**Métodos**
- Selvaraju et al. (2017), *Grad-CAM*, ICCV
- Hendrycks & Gimpel (2017), *A Baseline for Detecting Misclassified and
  Out-of-Distribution Examples in Neural Networks*, ICLR
- Liu et al. (2020), *Energy-based Out-of-distribution Detection*, NeurIPS
- He et al. (2016), *Deep Residual Learning for Image Recognition*, CVPR

---

## Licença

MIT — ver [LICENSE](LICENSE).

O código é MIT. **As imagens obtidas de levantamentos astronômicos e catálogos
seguem os termos de cada fonte** — verifique-os antes de redistribuir dados.
