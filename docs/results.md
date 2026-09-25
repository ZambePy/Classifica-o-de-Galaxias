# Resultados

Todos os números medidos no **conjunto de teste**, que só foi tocado ao final.
Hardware: GTX 1660 Ti (6 GB), PyTorch 2.14, CUDA 12.6.

Cada seção traz o comando que reproduz o número. Os artefatos ficam em
`ASTRO_DATA_ROOT/runs/<experimento>/`.

---

## Índice

- [Resumo](#resumo)
- [Nível 1 — tipo de objeto](#nível-1--tipo-de-objeto)
- [Nível 2 — morfologia de galáxia](#nível-2--morfologia-de-galáxia)
- [Nível 3 — tipo de nebulosa](#nível-3--tipo-de-nebulosa)
- [Cascata completa](#cascata-completa)
- [Detector de domínio](#detector-de-domínio)
- [Experimento: soft labels](#experimento-soft-labels)
- [Achado: o viés do plano galáctico](#achado-o-viés-do-plano-galáctico)
- [Histórico das rodadas](#histórico-das-rodadas)
- [Limitações conhecidas](#limitações-conhecidas)

---

## Resumo

| modelo | n (teste) | acurácia | macro-F1 | MCC | ROC AUC |
|---|---:|---:|---:|---:|---:|
| **nível 1** · objeto | 753 | 0,9841 | 0,9824 | 0,9758 | 0,9949 |
| **nível 2** · galáxia | 5.657 | 0,9420 | 0,8837 | 0,8936 | 0,9858 |
| **nível 3** · nebulosa | 251 | 0,8367 | 0,8297 | 0,7520 | 0,9635 |
| **cascata** ponta a ponta | 753 | **0,9163** | — | — | — |

O número que descreve o sistema é o da cascata: **91,6% das imagens recebem o
rótulo final correto**, atravessando os dois níveis.

---

## Nível 1 — tipo de objeto

`galaxy` · `nebula` · `other` — 3.510 imagens de treino

```powershell
python scripts\train.py --config configs\level1_object.yaml
python scripts\evaluate.py --config configs\level1_object.yaml
```

| | acurácia | acur. balanceada | kappa | MCC | ROC AUC |
|---|---:|---:|---:|---:|---:|
| | 0,9841 | 0,9823 | 0,9758 | 0,9758 | 0,9949 |

| classe | precisão | recall | F1 | especificidade | n | ROC AUC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| galaxy | **1,000** | 0,997 | 0,998 | 1,000 | 300 | 1,000 | 1,000 |
| nebula | 0,972 | 0,980 | 0,976 | 0,986 | 251 | 0,993 | 0,991 |
| other | 0,975 | 0,970 | 0,973 | 0,991 | 202 | 0,992 | 0,983 |

**Matriz de confusão** (linha = verdadeiro):

| | galaxy | nebula | other |
|---|---:|---:|---:|
| **galaxy** | 299 | 1 | 0 |
| **nebula** | 0 | 246 | 5 |
| **other** | 0 | 6 | 196 |

**299 de 300 galáxias corretas, e precisão de 1,000** — nada que não seja
galáxia foi chamado de galáxia. Toda a confusão residual está entre `nebula` e
`other`, e tem explicação: ver [o viés do plano galáctico](#achado-o-viés-do-plano-galáctico).

Treino: 30 épocas, ~6 min.

---

## Nível 2 — morfologia de galáxia

`spiral` · `elliptical` · `irregular` — 26.401 imagens de treino

```powershell
python scripts\train.py --config configs\level2_galaxy.yaml
python scripts\evaluate.py --config configs\level2_galaxy.yaml
```

| | acurácia | acur. balanceada | kappa | MCC | ROC AUC |
|---|---:|---:|---:|---:|---:|
| | 0,9420 | 0,9089 | 0,8932 | 0,8936 | 0,9858 |

| classe | precisão | recall | F1 | especificidade | n | ROC AUC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| spiral | 0,964 | 0,944 | 0,954 | 0,954 | 3.210 | 0,989 | 0,990 |
| elliptical | 0,966 | 0,959 | 0,962 | 0,980 | 2.105 | 0,994 | 0,991 |
| irregular | **0,664** | 0,825 | 0,735 | 0,973 | 342 | 0,975 | 0,814 |

**Matriz de confusão:**

| | spiral | elliptical | irregular |
|---|---:|---:|---:|
| **spiral** | 3.029 | 64 | 117 |
| **elliptical** | 61 | 2.018 | 26 |
| **irregular** | 52 | 8 | 282 |

`irregular` é a classe frágil, e a causa está no **rótulo, não no modelo**: o
Galaxy Zoo 2 não tem pergunta direta para "irregular". A aproximação por
`Class6.1` ("tem algo estranho") mistura galáxias genuinamente irregulares com
fusões, anéis e artefatos. 117 espirais acabam classificadas assim.

Treino: 40 épocas, 66 min.

---

## Nível 3 — tipo de nebulosa

`emission` · `reflection` · `planetary` · `supernova_remnant` — 1.167 de treino

```powershell
python scripts\train.py --config configs\level3_nebula.yaml
python scripts\evaluate.py --config configs\level3_nebula.yaml
```

| | acurácia | acur. balanceada | kappa | MCC | ROC AUC |
|---|---:|---:|---:|---:|---:|
| | 0,8367 | 0,8334 | 0,7518 | 0,7520 | 0,9635 |

| classe | precisão | recall | F1 | especificidade | n | ROC AUC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| emission | 0,866 | 0,846 | 0,856 | 0,860 | 130 | 0,932 | 0,947 |
| reflection | 0,892 | 0,868 | 0,880 | 0,981 | 38 | 0,984 | 0,939 |
| planetary | 0,825 | 0,846 | 0,835 | 0,967 | 39 | 0,982 | 0,917 |
| supernova_remnant | 0,723 | 0,773 | 0,747 | 0,937 | 44 | 0,956 | 0,756 |

É o nível mais fraco, e o motivo é quantidade de dados: **1.167 imagens de
treino contra 26.401 do nível 2**. Não existe "Galaxy Zoo das nebulosas" — o
dataset foi construído a partir de catálogos (ver [`datasets.md`](datasets.md)).

Treino: 26 épocas, 2 min.

---

## Cascata completa

O sistema como ele roda em produção: o nível 1 decide, e o resultado escolhe
qual modelo de nível 2 é chamado.

```powershell
python scripts\evaluate_cascade.py
```

| métrica | valor |
|---|---:|
| acurácia do nível 1 | 0,9841 |
| **acurácia da folha (rótulo final)** | **0,9163** |
| erro em cascata | 0,0159 |
| acurácia do subtipo, dado nível 1 correto | 0,9064 |

| ramo | n | nível 1 | folha | subtipo \| nível 1 ok |
|---|---:|---:|---:|---:|
| galaxy | 300 | 0,997 | 0,957 | 0,960 |
| nebula | 251 | 0,980 | 0,825 | 0,841 |
| other | 202 | 0,970 | 0,970 | — |

**Os 6,8 pontos perdidos entre o nível 1 (0,984) e a folha (0,916) são o custo
da arquitetura em cascata.** Parte é erro do modelo de subtipo; 1,6% são
amostras que o nível 1 errou e por isso chegaram ao submodelo errado.

O ramo das nebulosas é o mais caro: 0,825 contra 0,957 das galáxias.

> **Nota metodológica.** Esta métrica só é confiável porque os splits são
> consistentes entre níveis. Numa versão anterior cada nível era dividido de
> forma independente, e 218 das 300 galáxias do teste do nível 1 estavam no
> **treino** do nível 2. O número saía inflado e parecia plausível. A
> correção está em `build_assignment()`, com teste de regressão.

---

## Detector de domínio

Os modelos foram treinados em recortes de levantamentos (DSS2/SDSS). Uma
astrofoto processada é outro domínio. O detector usa MSP — *Maximum Softmax
Probability* (Hendrycks & Gimpel, ICLR 2017) — com limiar calibrado na
validação para aceitar 95% das imagens legítimas.

**Limiar calibrado: 0,9298**

### Detecção de erro

A confiança avisa quando *esta* predição está errada?

```powershell
python scripts\evaluate_ood.py
```

| métrica | valor |
|---|---:|
| AUROC | 0,7645 |
| confiança média nos acertos | 0,9613 (n=741) |
| confiança média nos erros | 0,8187 (n=12) |
| acertos acima do limiar | 96,4% |
| erros pegos pelo limiar | 66,7% |

### Detecção de domínio

Os **mesmos objetos** do conjunto de teste, mesmas coordenadas e mesmo
enquadramento, obtidos de outros levantamentos. O desenho isola a variável:
não é "outro objeto", é o mesmo objeto visto por outro instrumento.

```powershell
python scripts\build_ood_set.py
python scripts\evaluate_ood.py --ood-dir <pasta>
```

| levantamento | o que é | AUROC | fora do domínio aceito |
|---|---|---:|---:|
| **allWISE** | infravermelho | **0,953** | 11,2% |
| **Mellinger** | astrofotografia amadora processada | **0,887** | 36,2% |
| **PanSTARRS** | telescópio óptico moderno | 0,695 | 46,2% |

**A ordem valida o detector:** quanto mais distante o domínio, mais fácil
detectar. O infravermelho — onde o objeto nem se parece com a imagem óptica —
é pego em 89% dos casos. O PanSTARRS, óptico e portanto o mais próximo do
DSS2, passa quase metade das vezes.

### Validação com uma imagem real

Foto dos **Pilares da Criação** (Hubble, M16), baixada da internet:

```
object: "other"  ·  confidence: 0.895  ·  out_of_domain: TRUE
```

Classificou errado, **mas avisou**. Para comparação, a mesma nebulosa em
recorte DSS2:

```
Nebulosa de emissão  ·  0.977 → 0.926   ✓ correto
```

---

## Experimento: soft labels

O Galaxy Zoo não diz "esta galáxia é espiral" — diz "73% dos voluntários viram
um disco". A pergunta: treinar com a distribuição de votos, em vez do rótulo
majoritário, melhora o classificador?

**Desenho pareado:** mesma arquitetura, dados, semente e divisão. Só muda o
alvo. A validação mede contra o rótulo duro nos dois casos.

```powershell
python scripts\train.py --config configs\level2_galaxy_soft.yaml
```

| | rótulo duro | soft labels | |
|---|---:|---:|---|
| acurácia | 0,9420 | **0,9450** | ↑ |
| acurácia balanceada | **0,9089** | 0,8710 | ↓ |
| macro-F1 | **0,8837** | 0,8787 | ↓ |
| MCC | 0,8936 | **0,8970** | ↑ |
| ROC AUC | 0,9858 | **0,9893** | ↑ |

E na classe alvo do experimento:

| `irregular` | rótulo duro | soft labels | |
|---|---:|---:|---|
| precisão | 0,664 | **0,747** | **+8,3 pts** |
| recall | **0,825** | 0,690 | −13,5 pts |
| F1 | **0,735** | 0,717 | ↓ |

**Conclusão.** O método fez o que prometia e não fez o que se esperava dele. A
precisão do `irregular` subiu 8 pontos — o modelo deixou de chutar essa classe
onde não tinha certeza, que era o sintoma. Mas perdeu recall, e o F1 caiu de
leve.

O que melhorou foi a **ordenação**, não a decisão: ROC AUC e MCC subiram. O
modelo separa melhor as classes, mas o ponto de operação padrão (`argmax`)
ficou mal posicionado. Ajustar o limiar de decisão dessa classe provavelmente
captura os dois ganhos — experimento para outra rodada.

**Para produção, use o checkpoint de rótulo duro:** o recall do `irregular`
importa mais num dashboard (perder uma galáxia irregular é pior que classificar
uma espiral a mais como irregular).

22,8% das galáxias de treino são ambíguas (voto entre 0,35 e 0,65).

---

## Achado: o viés do plano galáctico

Nebulosas galácticas ficam no plano da Via Láctea. Olhar nessa direção
significa ver um campo denso e alaranjado — uma assinatura visual fácil, que
correlaciona com a classe **por acidente de geografia**. Um modelo pode
aprender isso e parecer ótimo nas métricas sem nunca ter olhado o objeto.

```powershell
python scripts\analyze_shortcut.py
```

### Evidência 1 — grupo de controle

`star_field` são recortes sorteados no plano galáctico, **sem nebulosa alguma**.

| origem | n | acerto | erros |
|---|---:|---:|---|
| empty_field | 45 | 1,000 | — |
| non_astronomical | 90 | 1,000 | — |
| globular_cluster | 22 | 1,000 | — |
| emission | 130 | 0,985 | 2 → other |
| supernova_remnant | 44 | 0,977 | 1 → other |
| planetary | 39 | 0,974 | 1 → other |
| reflection | 38 | 0,974 | 1 → other |
| **star_field** | 45 | **0,867** | **6 → nebulosa** |

**13,3% dos campos de céu vazios são classificados como nebulosa.**

### Evidência 2 — a geografia sozinha

Árvore de decisão usando **só a latitude galáctica**, sem ver imagem nenhuma:

| classe | AUROC só com \|b\| |
|---|---:|
| supernova_remnant | 0,768 |
| planetary | 0,727 |
| reflection | 0,719 |
| emission | 0,526 |

### Evidência 3 — distribuição no céu

| classe | n | \|b\| mediana | em \|b\| < 10° |
|---|---:|---:|---:|
| supernova_remnant | 292 | 0,6° | **99%** |
| emission | 867 | 1,9° | 85% |
| planetary | 258 | 7,2° | 59% |
| reflection | 250 | 11,6° | 47% |

### O que mudou com mais dados

Ampliar o dataset de nebulosas (1.032 → 1.667) resolveu boa parte:

| | antes | depois |
|---|---:|---:|
| supernova_remnant | 0,786 | **0,977** |
| globular_cluster | 0,867 | **1,000** |
| **star_field → nebulosa** | 6/47 | 6/45 |

**O controle não melhorou.** A leitura honesta: isso não é falta de dado. No
DSS2, um campo estelar muito denso genuinamente se parece com nebulosidade
difusa. É limite de resolução do levantamento, não do classificador.

Atacar esse resto exigiria um levantamento com mais detalhe no plano
galáctico, ou uma banda em Hα — que separa emissão de estrelas por física em
vez de por aparência.

---

## Histórico das rodadas

| | vazamento | splits corrigidos | dataset ampliado |
|---|---:|---:|---:|
| nível 1 macro-F1 | 0,958 | 0,968 | **0,982** |
| nível 1 MCC | 0,948 | 0,960 | **0,976** |
| nível 2 macro-F1 | 0,866 | 0,890 | 0,884 |
| nível 3 macro-F1 | 0,831 | 0,806 | **0,830** |
| cascata folha | — | 0,894 | **0,916** |
| erro em cascata | — | 0,026 | **0,016** |

As colunas não são estritamente comparáveis: a primeira tinha vazamento entre
níveis, e da segunda para a terceira o conjunto de teste mudou (o dataset de
nebulosas cresceu 62%). A terceira é a única metodologicamente correta.

### Bugs encontrados, todos silenciosos

Vale registrar: os três produziam números plausíveis.

| bug | sintoma | impacto |
|---|---|---|
| NaN em precisão mista | `train_loss=nan` entre métricas normais | modelo previa sempre a mesma classe |
| vazamento entre splits | acurácia de folha 0,38 com níveis de 0,97 | métrica da cascata inutilizável |
| colisão de nomes | log dizia "1667 salvos" | 301 imagens (18%) sobrescritas |

Todos têm teste de regressão hoje.

---

## Limitações conhecidas

1. **`irregular` é limitado pelo rótulo.** O Galaxy Zoo 2 não tem pergunta
   direta para essa classe; a aproximação mistura categorias distintas.
2. **`supernova_remnant` tem sinal óptico fraco.** A maioria foi descoberta em
   rádio. O F1 de 0,747 se apoia parcialmente em contexto.
3. **Domínio restrito ao DSS2/SDSS.** Astrofotos processadas são outro domínio
   — daí o detector existir.
4. **`star_field` × `nebula`** permanece ambíguo por limite de resolução.
5. **Conjunto de teste do nível 3 é pequeno** (251 imagens). Diferenças abaixo
   de ~3 pontos estão dentro do ruído.
6. **Uma semente só.** Não há barras de erro; para publicar, rode 3–5 sementes
   e reporte média e desvio.
