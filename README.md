# GALAXIA — classificador hierárquico de objetos astronômicos

Recebe a imagem de um objeto no céu e responde **o que ele é** — galáxia,
nebulosa ou outra coisa — e, quando é galáxia ou nebulosa, **de que tipo**.
Quando a imagem não se parece com nada que o modelo aprendeu, ele diz isso em
vez de inventar uma resposta.

```
                            imagem
                              │
                     ┌────────▼────────┐
                     │     NÍVEL 1     │   acurácia 0,982
                     │ galáxia/nebulosa│
                     │      /outro     │
                     └────────┬────────┘
                 ┌────────────┴────────────┐
          ┌──────▼──────┐           ┌──────▼──────┐
          │  NÍVEL 2a   │           │  NÍVEL 2b   │
          │  espiral    │           │  emissão    │
          │  elíptica   │  0,944    │  reflexão   │  0,839
          │  irregular  │           │  planetária │
          └─────────────┘           │  remanesc.  │
                                    └─────────────┘

              ponta a ponta: 90,9% de rótulo final correto
```

Projeto acadêmico de dois estudantes de curso técnico de IA. O objetivo não é
competir com modelos astronômicos especializados, e sim construir um
**baseline reprodutível e honesto** — documentado a ponto de outra pessoa
repetir os números, e com as limitações medidas em vez de escondidas.

---

## Índice

- [Resultados](#resultados)
- [O que torna este projeto diferente](#o-que-torna-este-projeto-diferente)
- [Começando](#começando)
- [Como usar](#como-usar)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Como o sistema funciona](#como-o-sistema-funciona)
- [Reproduzindo do zero](#reproduzindo-do-zero)
- [Divisão de trabalho na dupla](#divisão-de-trabalho-na-dupla)
- [Decisões técnicas](#decisões-técnicas)
- [Limitações conhecidas](#limitações-conhecidas)
- [Roteiro](#roteiro)
- [Referências](#referências)

---

## Resultados

| modelo | backbone | n | acurácia | macro-F1 | MCC | ROC AUC |
|---|---|---:|---:|---:|---:|---:|
| **nível 1** · objeto | resnet50 | 1.064 | 0,9821 | 0,9815 | 0,9732 | 0,9947 |
| **nível 2** · galáxia | resnet18 | 5.659 | 0,9438 | 0,8963 | 0,8964 | 0,9899 |
| **nível 3** · nebulosa | efficientnet_b0 | 409 | 0,8386 | 0,8105 | 0,7745 | 0,9471 |
| **cascata** ponta a ponta | — | 1.064 | **0,9088** | — | — | — |

Das 375 galáxias do teste, **todas as 375 foram classificadas corretamente**,
com precisão de 1,000 — nada que não fosse galáxia foi chamado de galáxia.

Dataset: **66.409 imagens**, das quais 37.716 galáxias (Galaxy Zoo), 2.737
nebulosas de seis catálogos e 1.984 em `other`.

> **Mas leia a [ablação por oclusão](docs/results.md#ablação-por-oclusão--o-resultado-central)
> antes de acreditar no nível 3.** Com o objeto desfocado, aquele modelo
> mantém 88,6% da acurácia — mais do que mantém sem o fundo. Ele decide pelo
> contexto, não pela nebulosa. O de galáxias faz o oposto: desfocado o objeto,
> cai abaixo de chutar.

Métricas completas, matrizes de confusão, experimentos e análise de vieses em
**[`docs/results.md`](docs/results.md)**.

---

## O que torna este projeto diferente

**1. O dataset de nebulosas é construído, não baixado — e está publicado.**
Para galáxias existe o Galaxy Zoo. Para nebulosas não existe equivalente. O
projeto monta o seu a partir de **seis catálogos astronômicos publicados** —
listas de objetos com coordenadas, compiladas por astrônomos ao longo de
décadas — e de um serviço de recorte do céu:

```
catálogo (nome, ra, dec, tipo)  →  hips2fits  →  imagem rotulada
```

O resultado são **2.737 nebulosas** com coordenada, tipo, enquadramento e
procedência, em [`docs/dataset/`](docs/dataset/). As imagens pertencem aos
levantamentos e não são redistribuíveis, mas a curadoria é nossa e cabe em
205 KB — qualquer pessoa reconstrói o dataset idêntico rodando um comando.

O [cartão do dataset](docs/dataset/dataset_card.md) documenta os filtros, a
deduplicação e **o viés medido**, com a instrução de reportá-lo junto com
qualquer acurácia obtida a partir dele.

**2. O sistema pode dizer "não sei".**
Uma rede com softmax sempre distribui 100% entre as classes que conhece — uma
foto do Hubble vira "nebulosa planetária, 97%". Aqui há um detector de
fora-de-domínio, **medido contra três levantamentos diferentes** (AUROC de
0,70 a 0,90) e validado com uma foto real do Hubble.

**3. Os vieses são medidos, não supostos.**
O projeto identificou e quantificou um viés de contexto no próprio dataset, por
**quatro métodos independentes**: ablação por oclusão, Grad-CAM, distribuição de
latitude galáctica e um grupo de controle com céu vazio.

O mais forte é a ablação, que tem controle interno:

```
             só objeto nítido   só fundo nítido
galáxia            98,8%              48,2%     ← abaixo do chute
nebulosa           83,1%              88,6%     ← o fundo basta
```

Galáxias vêm do SDSS, em alta latitude galáctica — nenhuma correlação possível
entre posição e morfologia. Nebulosas vêm do DSS2, no plano da Via Láctea, onde
o fundo correlaciona com a classe por acidente de geografia.

---

## Começando

**Pré-requisitos:** Python 3.10+, Git. GPU NVIDIA é recomendada para treinar,
mas **não é necessária** para rodar a API em modo mock nem os testes.

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
pytest            # 108 testes, sem GPU, ~10 segundos
```

> ⚠️ **Os dados não ficam no repositório.** Imagens, checkpoints e resultados
> vivem em `ASTRO_DATA_ROOT` (padrão `C:/astro-data`), configurado no `.env`.
> Se este repositório estiver dentro de uma pasta sincronizada (OneDrive,
> Dropbox), isso não é opcional — o Galaxy Zoo sozinho tem 61 mil imagens.

---

## Como usar

### Classificar imagens pela linha de comando

```powershell
python scripts\predict.py foto.jpg              # uma imagem
python scripts\predict.py C:\minhas_fotos       # uma pasta
python scripts\predict.py foto.jpg --gradcam    # + mapa de calor
python scripts\predict.py foto.jpg --json       # saída JSON
```

```
VII_20_1.jpg
  Nebulosa de reflexao  96.3%

  object  (modelo object-e30)
    ► Nebulosa                   ███████████████████████░  96.3%
      Outro objeto               █░░░░░░░░░░░░░░░░░░░░░░░   2.2%
      Galaxia                    ░░░░░░░░░░░░░░░░░░░░░░░░   1.5%

  nebula  (modelo nebula-e18)
    ► Nebulosa de reflexao       █████████████████░░░░░░░  72.1%
      Nebulosa de emissao        ██████░░░░░░░░░░░░░░░░░░  25.6%
```

Numa pasta, o resumo final diz quantas foram marcadas como fora do domínio.

### API HTTP

```powershell
powershell -ExecutionPolicy Bypass -File scripts\serve_api.ps1 -Mode real
```

Documentação interativa em <http://127.0.0.1:8000/docs> — dá para testar
upload de imagem pelo navegador.

**Modo `mock`** (padrão) responde sem nenhum modelo treinado, no formato
final. Existe para que o dashboard seja construído em paralelo ao treino —
numa dupla, é a maior alavanca de produtividade que existe.

Contrato completo em **[`docs/api.md`](docs/api.md)**.

---

## Estrutura do repositório

```
REDE NEURAL GALAXIA/
│
├── configs/                        um YAML por modelo da cascata
│   ├── level1_object.yaml          galáxia / nebulosa / outro
│   ├── level2_galaxy.yaml          morfologia de galáxia
│   ├── level2_galaxy_soft.yaml     ↑ variante com soft labels (experimento)
│   └── level3_nebula.yaml          tipo de nebulosa
│
├── src/astro_classifier/
│   ├── taxonomy.py                 ⭐ classes e hierarquia — fonte da verdade
│   ├── config.py                   configuração de experimento
│   ├── paths.py                    onde cada coisa mora no disco
│   │
│   ├── data/
│   │   ├── catalogs.py             catálogos VizieR → coordenadas + tamanho
│   │   ├── cutouts.py              coordenadas → imagens (hips2fits)
│   │   ├── sky_sampling.py         amostragem de céu por latitude galáctica
│   │   ├── datasets.py             Dataset do PyTorch, com soft labels
│   │   ├── transforms.py           pré-processamento e augmentation
│   │   └── splits.py               ⭐ atribuição global, consistente entre níveis
│   │
│   ├── models/                     backbone + cabeça linear
│   ├── training/
│   │   ├── loops.py                treino, early stopping, sanidade numérica
│   │   └── soft_labels.py          KL-divergência contra votos humanos
│   ├── evaluation/
│   │   ├── metrics.py              o pacote completo de métricas
│   │   ├── confusion.py            matrizes e curvas
│   │   ├── gradcam.py              onde o modelo olhou
│   │   └── cascade.py              ⭐ avaliação ponta a ponta
│   ├── ood/msp.py                  detector de fora-de-domínio
│   ├── inference/tta.py            test-time augmentation (8 simetrias)
│   ├── inference/pipeline.py       ⭐ a cascata em funcionamento
│   └── api/                        contrato HTTP + modo mock
│
├── scripts/
│   │   ── dados ──
│   ├── build_nebula_dataset.py     catálogos → imagens de nebulosa
│   ├── build_other_dataset.py      aglomerados + campos de céu sorteados
│   ├── build_ood_set.py            conjunto fora-de-domínio, 3 levantamentos
│   ├── madrugada.sh                bateria completa, sem supervisão
│   ├── fetch_non_astronomical.py   fotos comuns (STL-10)
│   ├── prepare_galaxy_zoo.py       37 probabilidades → 3 classes + soft labels
│   ├── inspect_dataset.py          ⭐ grades visuais — não pule
│   ├── make_splits.py              train/val/test consistente entre níveis
│   │   ── treino e avaliação ──
│   ├── train.py                    treina um nível
│   ├── train_all.ps1               os três em sequência, sem supervisão
│   ├── evaluate.py                 teste + Grad-CAM + calibração do detector
│   ├── evaluate_cascade.py         ⭐ o sistema de ponta a ponta
│   ├── evaluate_ood.py             detector de erro e de domínio
│   ├── analyze_shortcut.py         viés de contexto: 3 evidências
│   ├── analyze_occlusion.py        ⭐ ablação: o objeto ou o fundo decide?
│   ├── run_experiments.py          bateria: sementes, backbones, TTA
│   │   ── uso ──
│   ├── predict.py                  classificação pela linha de comando
│   ├── export_dataset.py           publica o catálogo em docs/dataset/
│   ├── serve_api.ps1               sobe a API
│   └── setup_env.ps1 / .sh         prepara o ambiente
│
├── tests/                          108 testes, sem GPU, ~10 segundos
├── docs/
│   ├── dataset/                    ⭐ o catálogo publicado
│   │   ├── nebulae_catalog.csv     2.737 objetos, reconstrutível
│   │   └── dataset_card.md         proveniência, filtros, vieses
│   ├── results.md                  ⭐ todas as medições
│   ├── datasets.md                 ⭐ de onde vem cada imagem
│   ├── api.md                      ⭐ contrato para o dashboard
│   └── superpowers/specs/          o design do sistema
└── notebooks/
```

---

## Como o sistema funciona

### A cascata

O nível 1 decide o tipo de objeto. **O resultado dele escolhe qual modelo do
nível 2 roda.** Se o nível 1 disser `other`, a cascata para — não existe
subtipo de "outro".

A resposta da API expõe cada etapa separadamente (campo `levels`), então o
dashboard mostra *onde* a decisão foi tomada.

**Custo medido:** 7,3 pontos de acurácia entre o nível 1 (0,982) e o rótulo
final (0,909). Desses, 1,8% são amostras que chegaram ao submodelo errado.

### O aviso de domínio

```
   TREINO (DSS2/SDSS)          USO REAL (Hubble)
     256×256, fraca              4000×4000, vibrante
     fundo escuro                falsa-cor saturada
     objeto pequeno              objeto preenche o quadro
                    └──── domain shift ────┘
```

O detector compara a confiança máxima do nível 1 com um limiar calibrado.
Abaixo dele, a resposta vem com `out_of_domain: true`.

**Validado com uma foto real dos Pilares da Criação:** classificou errado,
mas avisou. A mesma nebulosa em recorte DSS2 sai correta com 97,7%.

### Grad-CAM

`evaluation/gradcam.py` mostra onde o modelo olhou. Não é enfeite: **foi assim
que descobrimos que o modelo reconhecia o plano galáctico em vez do
remanescente de supernova.** Ver [`docs/results.md`](docs/results.md).

---

## Reproduzindo do zero

```powershell
# 1. Nebulosas — automático, ~40 min (6 catálogos, deduplicados)
python scripts\build_nebula_dataset.py --inspect     # confira antes
python scripts\build_nebula_dataset.py

# 2. Classe "other"
python scripts\build_other_dataset.py
python scripts\fetch_non_astronomical.py

# 3. Galáxias — baixe o Galaxy Zoo do Kaggle antes
python scripts\prepare_galaxy_zoo.py `
    --solutions C:\astro-data\raw\galaxy_zoo\training_solutions_rev1.csv `
    --images    C:\astro-data\raw\galaxy_zoo\images_training_rev1

# 4. OLHE AS IMAGENS. Não pule esta etapa.
python scripts\inspect_dataset.py --class-dir nebulae

# 5. Splits consistentes entre níveis
python scripts\make_splits.py --all --from-index galaxies_index.csv

# 6. Treino e avaliação — ~80 min
powershell -ExecutionPolicy Bypass -File scripts\train_all.ps1

# 7. Análises
python scripts\evaluate_cascade.py
python scripts\analyze_shortcut.py
python scripts\build_ood_set.py
python scripts\evaluate_ood.py --ood-dir C:\astro-data\ood\mellinger
```

> **A etapa 4 não é opcional.** A coleta por coordenada é cega: baixa o pedaço
> de céu que o catálogo aponta, sem verificar se o objeto aparece. Foi a
> inspeção visual que revelou que **nenhuma** das 20 planetárias amostradas
> mostrava a nebulosa — o campo de visão estava errado. Nenhuma métrica de
> treino teria mostrado isso.

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

**O contrato é o acordo.** Enquanto ele não mudar, cada lado evolui sozinho. E
como a API responde em modo mock desde o primeiro dia, o dashboard não espera
o treino terminar.

Para mudar um campo: avise a outra pessoa, suba `CONTRACT_VERSION` em
`api/schemas.py` e atualize `docs/api.md`. Os testes falham até que a mudança
seja consciente.

---

## Decisões técnicas

| decisão | por quê |
|---|---|
| **Um backbone por nível**, escolhido por medida | ResNet50 no nível 1 (+1,1 pt), ResNet18 no nível 2 (empata e treina 6× mais rápido), EfficientNet-B0 no nível 3 (+4,8 pts com ⅓ dos parâmetros). |
| **Três modelos em cascata**, não multi-tarefa | Os datasets são muito diferentes em tamanho (26 mil galáxias × 1,2 mil nebulosas); separados, cada um recebe a estratégia que precisa. |
| **Splits globais**, decididos uma vez | Dividir cada nível independentemente vazou 218 das 300 galáxias do teste para o treino do nível 2. |
| **Transfer learning**, não CNN do zero | Com ~1,2 mil imagens de nebulosa, uma rede do zero decora o treino. |
| **Rotação livre 0–360°** no augmentation | No céu não existe "em pé" — a orientação é definida pela posição do telescópio. É uma simetria real do problema. |
| **Cor quase intocada** | A cor carrega sinal físico: Hα avermelhado em emissão, azul espalhado em reflexão. |
| **Campo de visão por objeto** | As planetárias têm mediana de 9″ e os remanescentes de 18′ — 120× maiores. Um valor fixo não serve aos dois. |
| **Macro-F1** para escolher o checkpoint | Com classes desbalanceadas, chutar a majoritária dá acurácia alta e macro-F1 baixo. |
| **`sqrt_balanced`** nos pesos de classe | O peso cheio (5,5×) fazia o modelo prever `irregular` em excesso: recall 0,886 com precisão 0,579. |
| **Precisão mista DESLIGADA** | Nesta GPU (GTX 1660 Ti, sem tensor cores) o AMP é 4,6× mais lento e o cuDNN produz NaN a partir de batch 64. |
| **Deduplicação por coordenada** | Catálogos se sobrepõem; o LBN inclui dezenas de nebulosas de Sharpless com outra nomenclatura. |
| **Dados fora do repositório** | 61 mil imagens não se versionam. |

---

## Limitações conhecidas

Declaradas de propósito — um baseline com limitações medidas vale mais que um
número alto sem procedência. Detalhes em [`docs/results.md`](docs/results.md).

1. **O nível 3 decide pelo contexto**, não pela nebulosa — 80% do desempenho
   sobrevive ao desfoque do objeto. Seus números não medem reconhecimento.
2. **`irregular` é limitado pelo rótulo** (precisão 0,732). O Galaxy Zoo 2 não
   tem pergunta direta para essa classe.
3. **`supernova_remnant` tem sinal óptico fraco** — F1 0,575, e desaba nas duas
   condições da ablação.
4. **DSS2 é um levantamento dos anos 1990.** Para galáxias havia opção melhor
   (DECaLS, PanSTARRS); só as nebulosas galácticas exigem o DSS2.
5. **Sem benchmark externo.** Os números não são comparáveis com a literatura
   enquanto o mesmo protocolo não rodar num dataset com resultados publicados.
6. **Três sementes.** Nível 3 tem desvio de ±0,012 na acurácia: diferenças
   menores que ~2 pontos ali são ruído.

---

## Roteiro

- [x] **Infraestrutura**: estrutura, config, API mock, testes, CI
- [x] **Dados**: Galaxy Zoo, 6 catálogos de nebulosa, classe `other`, inspeção
- [x] **Baselines**: os três níveis treinados e avaliados
- [x] **Cascata completa**: avaliação ponta a ponta, erro em cascata medido
- [x] **Detector de domínio**: calibrado e medido em 3 levantamentos
- [x] **Análise de viés**: Grad-CAM, geografia, grupo de controle
- [x] **Experimento com soft labels**
- [x] **Múltiplas sementes** com média e desvio
- [x] **Comparação de backbones**: EfficientNet-B0 ganha 4,8 pts no nível 3 com 1/3 dos parâmetros
- [x] **Test-time augmentation**: +0,6 pt de acurácia no nível 1
- [x] **Ablação por oclusão**: isola a contribuição do objeto e do contexto
- [ ] **Benchmark contra trabalhos publicados** (Galaxy10 DECaLS)
- [ ] **Trocar o survey das galáxias** para DECaLS ou PanSTARRS
- [ ] **Cascata × multi-tarefa** (backbone compartilhado)
- [ ] **Métodos melhores de OOD**: energy-based, Mahalanobis

---

## Desenvolvimento

```powershell
pytest                              # 108 testes, sem GPU, ~10 s
ruff check src tests scripts        # lint
```

Os testes cobrem taxonomia, contrato da API, splits, métricas, cascata,
detector, soft labels e nomes de arquivo — tudo que não precisa de GPU. O CI
roda os dois a cada push, sem instalar o PyTorch.

Vários testes existem por causa de bugs reais que passaram despercebidos.
Estão documentados no próprio arquivo de teste, com o sintoma original.

---

## Referências

**Dados**
- Galaxy Zoo 2 — Willett et al. (2013), *MNRAS* 435, 2835
- Sharpless (1959), *Catalogue of HII Regions*, ApJS 4, 257
- van den Bergh (1966), *Catalogue of Reflection Nebulae*, AJ 71, 990
- Lynds (1965), *Catalogue of Bright Nebulae*, ApJS 12, 163
- Rodgers, Campbell & Whiteoak (1960), *Catalogue of Hα-emission regions*
- Acker et al. (1992), *Strasbourg-ESO Catalogue of Galactic Planetary Nebulae*
- Green (2009), *A Catalogue of Galactic Supernova Remnants*
- Harris (1996, ed. 2010), *Catalog of Milky Way Globular Clusters*
- VizieR e hips2fits — CDS, Observatoire astronomique de Strasbourg

**Métodos**
- He et al. (2016), *Deep Residual Learning for Image Recognition*, CVPR
- Selvaraju et al. (2017), *Grad-CAM*, ICCV
- Hendrycks & Gimpel (2017), *A Baseline for Detecting Misclassified and
  Out-of-Distribution Examples in Neural Networks*, ICLR
- Hinton et al. (2015), *Distilling the Knowledge in a Neural Network*
- Liu et al. (2020), *Energy-based Out-of-distribution Detection*, NeurIPS

---

## Licença

MIT — ver [LICENSE](LICENSE).

O código é MIT. **As imagens obtidas de levantamentos astronômicos e catálogos
seguem os termos de cada fonte** — verifique-os antes de redistribuir dados.
