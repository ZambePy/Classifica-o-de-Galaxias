# GALAXIA — classificador hierárquico de objetos astronômicos

Recebe a imagem de um objeto no céu e responde **o que ele é** — galáxia,
nebulosa ou outra coisa — e, quando é galáxia ou nebulosa, **de que tipo**.
Quando a imagem não se parece com nada que o modelo aprendeu, ele diz isso em
vez de inventar uma resposta.

```
                            imagem
                              │
                     ┌────────▼────────┐
                     │     NÍVEL 1     │   acurácia 0,966
                     │ galáxia/nebulosa│
                     │      /outro     │
                     └────────┬────────┘
                 ┌────────────┴────────────┐
          ┌──────▼──────┐           ┌──────▼──────┐
          │  NÍVEL 2a   │           │  NÍVEL 2b   │
          │  espiral    │           │  emissão    │
          │  elíptica   │  0,946    │  reflexão   │  0,854
          │  irregular  │           │  planetária │
          └─────────────┘           │  remanesc.  │
                                    └─────────────┘

              ponta a ponta: 89,4% de rótulo final correto
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
| **nível 1** · objeto | resnet50 | 1.048 | 0,9656 | 0,9639 | 0,9482 | 0,9892 |
| **nível 2** · galáxia | resnet18 | 5.656 | 0,9459 | 0,8998 | 0,9005 | 0,9905 |
| **nível 3** · nebulosa | efficientnet_b0 | 411 | 0,8540 | 0,8316 | 0,7999 | 0,9623 |
| **cascata** ponta a ponta | — | 1.048 | **0,8941** | — | — | — |

Das 375 galáxias do teste, **todas as 375 foram classificadas corretamente**,
com precisão de 1,000 — nada que não fosse galáxia foi chamado de galáxia.

Dataset: **42.437 imagens rotuladas** — 37.716 galáxias (Galaxy Zoo), 2.737
nebulosas de oito catálogos e 1.984 em `other`. Há mais 23.862 imagens do
Galaxy Zoo em disco que **não** entram em nenhum conjunto: são as que a votação
não rotulou com confiança suficiente. Total em disco: 66.299.

> **Mas leia a [ablação por oclusão](docs/results.md#ablação-por-oclusão--o-resultado-central)
> antes de acreditar no nível 3.** Com o objeto desfocado, aquele modelo
> mantém 90% da acurácia — mais do que mantém sem o fundo (70%). Ele decide
> pelo contexto, não pela nebulosa. O de galáxias faz o oposto: desfocado o
> objeto, cai abaixo de chutar.

**Benchmark externo:** rodando o mesmo pipeline na taxonomia oficial de 10
classes do **Galaxy10 DECaLS** — dataset público, tarefa idêntica à de trabalhos
publicados — sai **0,8764 de acurácia** (0,8541 balanceada, MCC 0,8606, n=2.669).
É o único número deste repositório comparável com a literatura; os demais usam
taxonomia própria.

Métricas completas, matrizes de confusão, experimentos e análise de vieses em
**[`docs/results.md`](docs/results.md)**.

---

## O que torna este projeto diferente

**1. O dataset de nebulosas é construído, não baixado — e está publicado.**
Para galáxias existe o Galaxy Zoo. Para nebulosas não existe equivalente. O
projeto monta o seu a partir de **oito catálogos astronômicos publicados** —
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
fora-de-domínio, **medido contra três levantamentos diferentes** e testado com a
mesma nebulosa fotografada por quatro instrumentos.

Três métodos foram comparados no mesmo conjunto, todos sem retreinar nada, e a
diferença entre eles é grande: o baseline da área (MSP) deixa passar **2 de
cada 3** imagens de fora, e a distância de Mahalanobis no espaço de features
deixa passar **1 em 7**.

| método | AUROC | fora do domínio aceito |
|---|---:|---:|
| MSP — Hendrycks & Gimpel 2017 | 0,727 | 66,7% |
| energia — Liu et al. 2020 | 0,764 | 61,7% |
| **Mahalanobis — Lee et al. 2018** | **0,976** | **13,8%** |

A API ainda responde com o **MSP por padrão**, de propósito: trocar o método
muda a escala do campo `domain.score`, que é parte do contrato com o dashboard.
Liga-se com `ASTRO_OOD_METHOD=mahalanobis` — ver [`docs/api.md`](docs/api.md).

**3. Os vieses são medidos, não supostos.**
O projeto identificou e quantificou um viés de contexto no próprio dataset, por
**quatro métodos independentes**: ablação por oclusão, Grad-CAM, distribuição de
latitude galáctica e um grupo de controle com céu vazio.

O mais forte é a ablação, que tem controle interno:

```
             só objeto nítido   só fundo nítido
galáxia            99,0%              47,2%     ← abaixo do chute
nebulosa           69,5%              90,0%     ← o fundo basta
```

Galáxias vêm do SDSS, em alta latitude galáctica — nenhuma correlação possível
entre posição e morfologia. Nebulosas vêm do DSS2, no plano da Via Láctea, onde
o fundo correlaciona com a classe por acidente de geografia.

Das quatro evidências, a ablação é a única com controle interno, e é por isso
que ela carrega a conclusão. A auditoria final descobriu que o **grupo de
controle estava contaminado** — 15% dos "campos sem nebulosa" continham uma
nebulosa catalogada dentro do recorte de 30′. Está corrigido no código e o
efeito está quantificado em
[`docs/results.md`](docs/results.md#o-viés-do-plano-galáctico); a conclusão sobre
o atalho se sustenta pela ablação, não por aquele número.

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
pytest            # 204 testes, sem GPU, ~20 segundos
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
VII_20_120.jpg
  Nebulosa de emissao  97.5%

  object  (modelo object-e27)
    ► Nebulosa                   ███████████████████████░  97.5%
      Galaxia                    ░░░░░░░░░░░░░░░░░░░░░░░░   1.3%
      Outro objeto               ░░░░░░░░░░░░░░░░░░░░░░░░   1.2%

  nebula  (modelo nebula-e24)
    ► Nebulosa de emissao        ███████████████████████░  94.1%
      Nebulosa planetaria        █░░░░░░░░░░░░░░░░░░░░░░░   2.6%
      Nebulosa de reflexao       ░░░░░░░░░░░░░░░░░░░░░░░░   1.9%
      Remanescente de supernova  ░░░░░░░░░░░░░░░░░░░░░░░░   1.4%
```

Em consoles que não aceitam esses caracteres — o terminal padrão do Windows usa
cp1252 — a saída cai automaticamente para `#`, `.` e `>`. Antes de 26/09 ela
simplesmente quebrava com `UnicodeEncodeError` no meio da tabela.

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
├── configs/                        um YAML por modelo
│   ├── level1_object.yaml          galáxia / nebulosa / outro
│   ├── level2_galaxy.yaml          morfologia de galáxia
│   ├── level2_galaxy_soft.yaml     ↑ variante com soft labels (experimento)
│   ├── level3_nebula.yaml          tipo de nebulosa
│   ├── level*_*_resnet50.yaml      ↑ variantes de backbone (comparação)
│   ├── level*_*_efficientnet_b0.yaml
│   └── multitask.yaml              um backbone, três cabeças (comparação)
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
│   ├── models/
│   │   ├── backbone.py             resnet18 | resnet50 | efficientnet_b0
│   │   ├── classifier.py           backbone + cabeça linear (um nível)
│   │   └── multitask.py            ⭐ um backbone, três cabeças + loss mascarada
│   ├── training/
│   │   ├── loops.py                treino, early stopping, sanidade numérica
│   │   └── soft_labels.py          KL-divergência contra votos humanos
│   ├── evaluation/
│   │   ├── metrics.py              o pacote completo de métricas
│   │   ├── confusion.py            matrizes e curvas
│   │   ├── gradcam.py              onde o modelo olhou
│   │   └── cascade.py              ⭐ avaliação ponta a ponta
│   ├── ood/                        detector de fora-de-domínio
│   │   ├── msp.py                  máx. softmax (baseline, Hendrycks 2017)
│   │   ├── energy.py               log-sum-exp dos logits (Liu 2020)
│   │   └── mahalanobis.py          ⭐ distância nas features (Lee 2018)
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
│   ├── compare_ood.py              ⭐ MSP × energia × Mahalanobis
│   ├── train_multitask.py          ⭐ cascata × multi-tarefa
│   ├── benchmark_galaxy10.py       ⭐ Galaxy10 DECaLS: benchmark + troca de survey
│   │   ── uso ──
│   ├── predict.py                  classificação pela linha de comando
│   ├── export_dataset.py           publica o catálogo em docs/dataset/
│   ├── check_docs.py               ⭐ confere os números da documentação
│   ├── serve_api.ps1               sobe a API
│   └── setup_env.ps1 / .sh         prepara o ambiente
│
├── tests/                          204 testes, sem GPU, ~20 segundos
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

**Custo medido:** 7,2 pontos de acurácia entre o nível 1 (0,966) e o rótulo
final (0,894). Desses, 3,4% são amostras que chegaram ao submodelo errado.

### O aviso de domínio

```
   TREINO (DSS2/SDSS)          USO REAL (Hubble)
     256×256, fraca              4000×4000, vibrante
     fundo escuro                falsa-cor saturada
     objeto pequeno              objeto preenche o quadro
                    └──── domain shift ────┘
```

O detector compara a confiança máxima do nível 1 com um limiar calibrado
(0,8248, para aceitar 95% das imagens legítimas). Abaixo dele, a resposta vem
com `out_of_domain: true`.

**Testado com a mesma nebulosa (M16) em quatro instrumentos**, e o resultado é
desconfortável: no recorte DSS2 o sistema acerta com 97,8%, mas nas versões
Mellinger e allWISE ele erra — chama de `other` — com confiança de 96% e 92%, ou
seja **sem disparar o aviso**. É a falha do MSP no caso de uso real, e é o que a
distância de Mahalanobis resolve (AUROC 0,993 nos dois conjuntos). A tabela
completa está em
[`docs/results.md`](docs/results.md#validação-com-imagem-real--o-mesmo-objeto-em-quatro-instrumentos).

### Grad-CAM

`evaluation/gradcam.py` mostra onde o modelo olhou. Não é enfeite: **foi assim
que descobrimos que o modelo reconhecia o plano galáctico em vez do
remanescente de supernova.** Ver [`docs/results.md`](docs/results.md).

---

## Reproduzindo do zero

```powershell
# 1. Nebulosas — automático, ~40 min (8 catálogos, deduplicados)
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
#    O --cap-object 2500 NÃO é opcional para reproduzir os números publicados:
#    ele dá 2.500 galáxias no nível 1 (1.750 de treino), equilibrando com as
#    nebulosas. O padrão do script é 2000, e produziria outros resultados.
python scripts\make_splits.py --all --from-index galaxies_index.csv --cap-object 2500

# 6. Treino e avaliação — ~100 min nesta GPU (os três níveis)
powershell -ExecutionPolicy Bypass -File scripts\train_all.ps1

# 7. Análises
python scripts\evaluate_cascade.py
python scripts\analyze_occlusion.py --config configs\level3_nebula.yaml
python scripts\analyze_shortcut.py
python scripts\build_ood_set.py
python scripts\compare_ood.py                 # MSP x energia x Mahalanobis

# 8. Confira que a documentação continua verdadeira
python scripts\check_docs.py
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
| **Três modelos em cascata**, não multi-tarefa | **Medido, não suposto:** o backbone compartilhado custa 1,5 pt de acurácia da folha, toda a perda concentrada em `other` (−4,1 pt) — os subtipos empatam. Em troca economiza 39% dos parâmetros e uma passada de inferência. A cascata vence porque o objetivo é precisão, não peso. [Medição](docs/results.md#cascata--multi-tarefa). |
| **Splits globais**, decididos uma vez | Dividir cada nível independentemente vazou 218 das 300 galáxias do teste para o treino do nível 2. |
| **Transfer learning**, não CNN do zero | Com ~1,9 mil imagens de nebulosa, uma rede do zero decora o treino. |
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

1. **O nível 3 decide pelo contexto**, não pela nebulosa — descontando o chute,
   83% do desempenho sobrevive ao desfoque do objeto. Seus números não medem
   reconhecimento.
2. **`irregular` é limitado pelo rótulo** (precisão 0,729). O Galaxy Zoo 2 não
   tem pergunta direta para essa classe.
3. **`supernova_remnant` tem sinal óptico fraco** — F1 0,673, e desaba nas duas
   condições da ablação.
4. **O desempenho depende do instrumento, e isso foi medido.** Nosso nível 2,
   treinado em SDSS, perde **12,4 pontos** de acurácia balanceada ao ser avaliado
   nas mesmas classes em imagens DECaLS — contra um modelo treinado nativamente
   em DECaLS no mesmo teste. Morfologia de galáxia deveria ser independente do
   telescópio; não é. As nebulosas seguem presas ao DSS2 por não haver
   levantamento melhor no plano galáctico.
5. **Três sementes.** Nível 3 tem desvio de ±0,012 na acurácia: diferenças
   menores que ~2 pontos ali são ruído.
6. **Vazamento residual no nível 3.** A deduplicação por posição deixava passar
   22 objetos; 6 punham a mesma imagem no treino e no teste. Corrigido no código,
   mas os modelos aqui foram treinados antes — efeito estimado ≤1,5 pt, abaixo do
   ruído entre sementes.
7. **O grupo de controle do atalho estava contaminado.** 15% dos campos
   estelares continham uma nebulosa catalogada, e a taxa do atalho cai de 20%
   para **15,4%** com o controle limpo. Corrigido; a conclusão sobre o atalho se
   sustenta pela ablação, que tem controle interno.
8. **O detector de domínio dá 4,9% de falso alarme.** É o preço do ponto de
   operação escolhido (aceitar 95% das imagens legítimas), não um defeito — mas
   significa que ~1 em 20 imagens boas recebe o aviso.

---

## Roteiro

- [x] **Infraestrutura**: estrutura, config, API mock, testes, CI
- [x] **Dados**: Galaxy Zoo, 8 catálogos de nebulosa, classe `other`, inspeção
- [x] **Baselines**: os três níveis treinados e avaliados
- [x] **Cascata completa**: avaliação ponta a ponta, erro em cascata medido
- [x] **Detector de domínio**: calibrado e medido em 3 levantamentos
- [x] **Análise de viés**: Grad-CAM, geografia, grupo de controle
- [x] **Experimento com soft labels** — resultado **negativo**: −2,1 pts de macro-F1, o dano concentrado em `irregular` ([medição](docs/results.md#soft-labels--um-resultado-negativo))
- [x] **Múltiplas sementes** com média e desvio
- [x] **Comparação de backbones**: EfficientNet-B0 ganha 4,8 pts no nível 3 com 1/3 dos parâmetros
- [x] **Test-time augmentation**: +0,6 pt de acurácia no nível 1
- [x] **Ablação por oclusão**: isola a contribuição do objeto e do contexto
- [x] **Benchmark contra trabalhos publicados**: **0,8764** de acurácia na
      taxonomia oficial de 10 classes do Galaxy10 DECaLS
      ([medição](docs/results.md#galaxy10-decals--benchmark-externo-e-troca-de-survey))
- [x] **Trocar o survey das galáxias** para DECaLS: treinado em DECaLS dá 0,8932
      de acurácia balanceada, e o nosso modelo treinado em SDSS fica **12,4 pts
      abaixo** no mesmo teste — o preço de trocar de instrumento
      ([medição](docs/results.md#galaxy10-decals--benchmark-externo-e-troca-de-survey))
- [x] **Cascata × multi-tarefa** (backbone compartilhado): a cascata ganha 1,5 pt
      de folha; o subtipo **empata** e a perda toda está em `other`
      ([medição](docs/results.md#cascata--multi-tarefa))
- [x] **Métodos melhores de OOD**: energia e Mahalanobis implementados e
      comparados — Mahalanobis leva a AUROC de 0,727 para **0,976**
      ([medição](docs/results.md#três-métodos-de-fora-de-domínio))

---

## Desenvolvimento

```powershell
pytest                              # 204 testes, sem GPU, ~20 s
ruff check src tests scripts        # lint
python scripts\check_docs.py        # os números dos docs batem com os dados?
```

O `check_docs.py` existe porque a auditoria final encontrou seis números
errados **na documentação** — incluindo o total do dataset citado como 66.409
quando a soma das partes no mesmo parágrafo dava 42.437. Nenhum quebrava
código; todos quebravam a confiança de quem lê. Ele compara contagens de
arquivo, métricas gravadas em JSON e somas de tabela contra o que os `.md`
afirmam, e sai com código 1 se algo divergir.

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
- Lee et al. (2018), *A Simple Unified Framework for Detecting
  Out-of-Distribution Samples and Adversarial Attacks*, NeurIPS
- Leung & Bovy (2019), *astroNN* — origem do Galaxy10 DECaLS
- Walmsley et al. (2022), *Galaxy Zoo DECaLS*, *MNRAS* 509, 3966

---

## Licença

MIT — ver [LICENSE](LICENSE).

O código é MIT. **As imagens obtidas de levantamentos astronômicos e catálogos
seguem os termos de cada fonte** — verifique-os antes de redistribuir dados.
