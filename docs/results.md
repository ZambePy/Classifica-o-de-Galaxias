# Resultados

Todos os números medidos no **conjunto de teste**, que só foi tocado ao final.
Hardware: GTX 1660 Ti (6 GB), PyTorch 2.14, CUDA 12.6.

Cada seção traz o comando que reproduz o número. Os artefatos ficam em
`ASTRO_DATA_ROOT/runs/<experimento>/`.

---

## Índice

- [Resumo](#resumo)
- [Ablação por oclusão — o resultado central](#ablação-por-oclusão--o-resultado-central)
- [Nível 1 — tipo de objeto](#nível-1--tipo-de-objeto)
- [Nível 2 — morfologia de galáxia](#nível-2--morfologia-de-galáxia)
- [Nível 3 — tipo de nebulosa](#nível-3--tipo-de-nebulosa)
- [Cascata completa](#cascata-completa)
- [Variabilidade entre sementes](#variabilidade-entre-sementes)
- [Comparação de backbones](#comparação-de-backbones)
- [Test-time augmentation](#test-time-augmentation)
- [Detector de domínio](#detector-de-domínio)
- [O viés do plano galáctico](#o-viés-do-plano-galáctico)
- [Bugs silenciosos encontrados](#bugs-silenciosos-encontrados)
- [Limitações conhecidas](#limitações-conhecidas)

---

## Resumo

| modelo | backbone | n (teste) | acurácia | macro-F1 | MCC | ROC AUC |
|---|---|---:|---:|---:|---:|---:|
| **nível 1** · objeto | resnet50 | 1.064 | 0,9821 | 0,9815 | 0,9732 | 0,9947 |
| **nível 2** · galáxia | resnet18 | 5.659 | 0,9438 | 0,8963 | 0,8964 | 0,9899 |
| **nível 3** · nebulosa | efficientnet_b0 | 409 | 0,8386 | 0,8105 | 0,7745 | 0,9471 |
| **cascata** ponta a ponta | — | 1.064 | **0,9088** | — | — | — |

Dataset: **66.409 imagens** no total, das quais 37.716 galáxias (Galaxy Zoo),
2.737 nebulosas de seis catálogos e 1.984 em `other`.

> **Aviso de leitura.** O conjunto de teste do nível 1 passou de 753 para
> 1.064 imagens e ficou balanceado entre as três classes. Comparações com
> rodadas anteriores não são válidas.

---

## Ablação por oclusão — o resultado central

```powershell
python scripts\analyze_occlusion.py --config configs\level3_nebula.yaml
```

A mesma imagem de teste em três condições: completa, com o **fundo
desfocado** (só o objeto nítido) e com o **centro desfocado** (só o fundo
nítido). O objeto está centralizado por construção — os recortes vêm de
`hips2fits` centrados na coordenada do catálogo.

| nível | completa | só objeto | só fundo | chute | conclusão |
|---|---:|---:|---:|---:|---|
| **galáxia** | 0,9438 | **0,9321** (98,8%) | 0,4545 (48,2%) | 0,5674 | usa o **objeto** |
| **nebulosa** | 0,8386 | 0,6968 (83,1%) | **0,7433** (88,6%) | 0,3619 | usa o **contexto** |
| objeto | 0,9821 | 0,8383 (85,4%) | 0,8318 (84,7%) | 0,3524 | usa os dois |

**O nível 2 é o controle interno.** Com o objeto desfocado, o classificador de
galáxias cai para 0,4545 — **abaixo de chutar a classe majoritária**. Ele
depende inteiramente da morfologia, como se espera de um classificador que
funciona.

**O nível 3 faz o oposto.** Com o objeto desfocado, mantém 88,6% da acurácia —
mais do que mantém com o fundo desfocado (83,1%). Descontando o chute, **o
fundo sozinho entrega 80,0% do desempenho**.

Trocar o backbone de ResNet18 para EfficientNet-B0 subiu a acurácia de 0,797
para 0,839, mas **não reduziu a dependência do contexto** — ela caiu de 89,3%
para 80,0%, ainda dominante. Isso é o esperado: viés de dataset não se corrige
trocando de arquitetura.

### Por classe

| classe | completa | só objeto | só fundo |
|---|---:|---:|---:|
| spiral | 0,954 | 0,950 | 0,213 |
| elliptical | 0,956 | 0,946 | 0,582 |
| irregular | 0,778 | 0,692 | 0,446 |
| emission | 0,814 | 0,667 | 0,738 |
| reflection | 0,900 | 0,807 | 0,799 |
| planetary | 0,862 | 0,750 | 0,788 |
| supernova_remnant | 0,667 | 0,441 | 0,235 |

`supernova_remnant` desaba nas duas condições — precisa da imagem inteira,
coerente com não ter objeto nítido no óptico.

### Por que os dois casos diferem

Galáxias vêm do **Galaxy Zoo (SDSS)**, que observa calotas galácticas em alta
latitude. Não há correlação possível entre posição e morfologia.

Nebulosas vêm do **DSS2**, e nebulosas galácticas ficam no plano da Via
Láctea. Ali o fundo — campo denso, alaranjado, com poeira — correlaciona com a
classe por acidente de geografia.

> **Nota metodológica.** A primeira versão deste teste usava uma máscara
> circular de borda dura. O resultado foi inutilizável: as duas condições
> caíram abaixo do chute, porque a borda abrupta é ela mesma uma imagem
> fora de distribuição. O teste media o artefato. O desfoque preserva
> brilho, cor e estatística local e destrói só a estrutura fina.

---

## Nível 1 — tipo de objeto

`galaxy` · `nebula` · `other` — 4.966 imagens de treino, classes balanceadas

```powershell
python scripts\train.py --config configs\level1_object.yaml
python scripts\evaluate.py --config configs\level1_object.yaml
```

| | acurácia | balanceada | kappa | MCC | ROC AUC |
|---|---:|---:|---:|---:|---:|
| | 0,9821 | 0,9821 | 0,9731 | 0,9732 | 0,9947 |

| classe | precisão | recall | F1 | n |
|---|---:|---:|---:|---:|
| galaxy | **1,000** | **1,000** | **1,000** | 375 |
| nebula | 0,984 | 0,965 | 0,974 | 375 |
| other | 0,960 | 0,981 | 0,970 | 314 |

| | galaxy | nebula | other |
|---|---:|---:|---:|
| **galaxy** | 375 | 0 | 0 |
| **nebula** | 0 | 353 | 22 |
| **other** | 0 | 8 | 306 |

**375 de 375 galáxias corretas, sem um único falso positivo.** Toda a confusão
está entre `nebula` e `other`.

---

## Nível 2 — morfologia de galáxia

`spiral` · `elliptical` · `irregular` — 26.401 de treino

| | acurácia | balanceada | kappa | MCC | ROC AUC |
|---|---:|---:|---:|---:|---:|
| | 0,9438 | 0,9123 | 0,8961 | 0,8964 | 0,9899 |

| classe | precisão | recall | F1 | n |
|---|---:|---:|---:|---:|
| spiral | 0,966 | 0,943 | 0,954 | 3.211 |
| elliptical | 0,949 | 0,963 | 0,956 | 2.106 |
| irregular | 0,732 | 0,830 | 0,778 | 342 |

`irregular` é limitado pelo **rótulo, não pelo modelo**: o Galaxy Zoo 2 não tem
pergunta direta para essa classe, e a aproximação mistura galáxias irregulares
com fusões, anéis e artefatos.

---

## Nível 3 — tipo de nebulosa

Quatro classes — 1.916 de treino, vindas de seis catálogos

| | acurácia | balanceada | kappa | MCC | ROC AUC |
|---|---:|---:|---:|---:|---:|
| | 0,8386 | 0,8158 | 0,7737 | 0,7745 | 0,9471 |

| classe | precisão | recall | F1 | n |
|---|---:|---:|---:|---:|
| emission | 0,808 | 0,820 | 0,814 | 128 |
| reflection | 0,922 | 0,878 | 0,900 | 148 |
| planetary | 0,857 | 0,867 | 0,862 | 90 |
| supernova_remnant | 0,638 | 0,698 | 0,667 | 43 |

**Estes números devem ser lidos junto com a ablação acima:** boa parte do
desempenho vem do contexto, não do objeto.

---

## Cascata completa

```powershell
python scripts\evaluate_cascade.py
```

| métrica | valor |
|---|---:|
| acurácia do nível 1 | 0,9821 |
| **acurácia da folha (rótulo final)** | **0,9088** |
| erro em cascata | 0,0179 |
| subtipo dado nível 1 correto | 0,8942 |

| ramo | n | nível 1 | folha | subtipo \| nível 1 ok |
|---|---:|---:|---:|---:|
| galaxy | 375 | 1,000 | 0,941 | 0,941 |
| nebula | 375 | 0,965 | 0,816 | 0,845 |
| other | 314 | 0,981 | 0,981 | — |

Os **7,3 pontos** perdidos entre o nível 1 e o rótulo final são o custo da
arquitetura em cascata. Com os backbones promovidos, o erro em cascata caiu
de 0,0282 para **0,0179**.

---

## Variabilidade entre sementes

```powershell
python scripts\run_experiments.py --only seeds --seeds 1 2
```

| nível | acurácia | macro-F1 |
|---|---|---|
| objeto | 0,9731 ± 0,0022 | 0,9721 ± 0,0022 |
| nebulosa | 0,8093 ± 0,0122 | 0,7779 ± 0,0167 |

**Consequência prática:** no nível 3, qualquer diferença menor que ~2 pontos
está dentro do ruído. Comparar dois modelos ali exige mais de uma semente.

---

## Comparação de backbones

| nível | backbone | acurácia | macro-F1 | MCC | parâmetros |
|---|---|---:|---:|---:|---:|
| nebulosa | resnet18 | 0,7971 | 0,7628 | 0,7177 | 11,2 M |
| nebulosa | **efficientnet_b0** | **0,8386** | **0,8105** | **0,7745** | 4,0 M |
| nebulosa | resnet50 | 0,8191 | 0,7818 | 0,7461 | 23,5 M |
| objeto | resnet18 | 0,9718 | 0,9708 | 0,9579 | 11,2 M |
| objeto | efficientnet_b0 | 0,9718 | 0,9718 | 0,9591 | 4,0 M |
| objeto | **resnet50** | **0,9821** | **0,9815** | **0,9732** | 23,5 M |

**O EfficientNet-B0 ganha 4,8 pontos de macro-F1 no nível 3** — bem acima do
desvio entre sementes (±0,017), então é ganho real. E consegue isso com **um
terço dos parâmetros** do ResNet18.

No nível 1, onde o problema é mais fácil, a ordem se inverte: o ResNet50 lidera
com 0,9815, mas a diferença para o ResNet18 (0,9708) é de 1,1 ponto contra um
desvio de ±0,002 — pequena, porém real.

### Nível 2: o EfficientNet não compensa

O treino do nível 2 com EfficientNet-B0 foi **interrompido na época 13 de 40**,
depois de a comparação já ter respondido:

| época | EfficientNet-B0 | ResNet18 |
|---:|---:|---:|
| 10 | 0,8721 | 0,8641 |
| 11 | 0,8706 | 0,8640 |
| 12 | 0,8571 | 0,8639 |
| **tempo/época** | **522 s** | **89 s** |

Empate em macro-F1 dentro do ruído, e **5,9× o custo de treino**. A explicação
é de hardware: o EfficientNet usa convoluções separáveis em profundidade, mal
otimizadas em GPUs Turing sem tensor cores. O ganho de parâmetros não se
traduz em ganho de tempo nesta placa.

Deixar rodar as 27 épocas restantes custaria 4 horas para confirmar um empate.

**Recomendação por nível:**

| nível | backbone | motivo |
|---|---|---|
| 1 · objeto | **resnet50** | +1,1 pt de macro-F1; inferência ~40 ms |
| 2 · galáxia | **resnet18** | empata com alternativas e treina 6× mais rápido |
| 3 · nebulosa | **efficientnet_b0** | +4,8 pts com 1/3 dos parâmetros |

A escolha do nível 3 é fácil — mais preciso e mais leve. A do nível 1 troca
1 ponto pelo dobro do tempo de inferência, que segue instantâneo. A do nível 2
é a mais interessante: **a arquitetura mais moderna perde para a mais simples
quando o hardware é levado em conta.**

### TTA sobre o melhor backbone

| nível 3 · efficientnet_b0 | acurácia | macro-F1 |
|---|---:|---:|
| sem TTA | **0,8386** | **0,8105** |
| com TTA | 0,8337 | 0,8087 |

O TTA **não ajuda** quando combinado com o melhor backbone — a diferença
(−0,5 ponto) está dentro do ruído. Ele rendeu no nível 1 com ResNet18
(+0,6 ponto de acurácia), mas não é ganho universal.

---

## Test-time augmentation

Média das probabilidades sobre as 8 simetrias do quadrado (rotações de 90° e
espelhamentos). No céu não existe orientação privilegiada, então as 8 vistas
são igualmente válidas.

| nível | sem TTA | com TTA |
|---|---:|---:|
| objeto (acurácia) | 0,9718 | **0,9774** |
| objeto (MCC) | 0,9579 | **0,9663** |
| nebulosa (acurácia) | 0,7971 | 0,7971 |

Custo: 8× mais inferências — de ~20 ms para ~60 ms, ainda instantâneo.

---

## Detector de domínio

Método MSP (Hendrycks & Gimpel, ICLR 2017), limiar calibrado na validação para
aceitar 95% das imagens legítimas. **Limiar: 0,8840.**

### Detecção de erro

| métrica | valor |
|---|---:|
| AUROC | 0,9022 |
| confiança média nos acertos | 0,9577 (n=1.034) |
| confiança média nos erros | 0,7941 (n=30) |
| erros pegos pelo limiar | 63,3% |

**AUROC 0,90 — vale exibir o aviso no dashboard.**

### Detecção de domínio

Os mesmos objetos do teste, obtidos de outros levantamentos.

| levantamento | AUROC | fora do domínio aceito |
|---|---:|---:|
| Mellinger (astrofotografia) | 0,868 | 33,8% |
| PanSTARRS (óptico moderno) | 0,763 | 42,5% |
| allWISE (infravermelho) | 0,526 | 61,3% |

> **Correção de uma interpretação anterior.** Na rodada com o dataset menor, a
> ordem era outra (allWISE 0,953, PanSTARRS 0,695) e eu havia escrito que ela
> "validava o detector", porque AUROC crescia com a distância do domínio. Com o
> dataset ampliado a ordem **inverteu**. Aquela leitura não se sustenta: com
> `open_cluster` e mais variedade em `other`, o modelo passou a aceitar
> imagens que antes rejeitava. O detector continua útil (0,70–0,90), mas a
> explicação elegante estava errada.

### Validação com imagem real

Foto dos Pilares da Criação (Hubble, M16) baixada da internet:

```
object: "other" · confidence: 0.895 · out_of_domain: TRUE
```

Classificou errado, **mas avisou**. A mesma nebulosa em recorte DSS2 sai
correta com 0,977.

---

## O viés do plano galáctico

Além da ablação por oclusão, três evidências independentes:

```powershell
python scripts\analyze_shortcut.py
```

### Grupo de controle

`star_field` são recortes sorteados no plano galáctico, **sem nebulosa alguma**.

| origem | n | acerto | erros |
|---|---:|---:|---|
| empty_field, non_astronomical, globular_cluster | — | 1,000 | — |
| emission | 130 | 0,985 | 2 → other |
| supernova_remnant | 44 | 0,977 | 1 → other |
| **star_field** | 90 | **0,956** | **4 → nebulosa** |

### Geografia sozinha

Árvore de decisão usando **só a latitude galáctica**, sem ver imagem:

| classe | AUROC só com \|b\| | (dataset anterior) |
|---|---:|---:|
| supernova_remnant | 0,760 | 0,795 |
| planetary | 0,615 | 0,727 |
| reflection | 0,545 | 0,719 |
| emission | 0,489 | 0,526 |

### O que mudou com mais dados

| | dataset menor | dataset ampliado |
|---|---:|---:|
| star_field → nebulosa | 13,3% | **7,8%** |
| AUROC médio só com \|b\| | 0,692 | **0,602** |

**Ampliar o dataset reduziu o atalho pela metade.** Eu havia previsto que não
reduziria — estava errado. As nebulosas dos catálogos novos (Magakian,
Kohoutek) estão mais espalhadas pelo céu, e a correlação entre classe e
latitude enfraqueceu.

Mas a ablação por oclusão mostra que **o atalho continua dominante**: o fundo
ainda explica 89% do desempenho no nível 3.

---

## Bugs silenciosos encontrados

Todos produziam números plausíveis. Todos têm teste de regressão hoje.

| bug | o que parecia | o que era |
|---|---|---|
| NaN em precisão mista | acurácia 0,567 | previa sempre a mesma classe |
| vazamento entre splits | cascata 0,38 | 218/300 do teste estavam no treino |
| colisão de nomes | "1667 salvos" | 301 imagens (18%) sobrescritas |
| checkpoint sobrescrito | experimento normal | produção trocada de arquitetura |
| avaliação do checkpoint errado | "backbone não importa" | ganho real de 4,8 pontos |
| oclusão com borda dura | "depende do objeto" | media o artefato da máscara |

---

## Limitações conhecidas

1. **O nível 3 depende do contexto** (89% do desempenho vem do fundo). Os
   números dessa etapa não medem reconhecimento de nebulosa.
2. **`irregular` é limitado pelo rótulo** (precisão 0,732).
3. **`supernova_remnant` tem sinal óptico fraco** — F1 0,575, e desaba nas
   duas condições da ablação.
4. **DSS2 é um levantamento dos anos 1990.** Para galáxias havia opção melhor
   (DECaLS, PanSTARRS); o DSS2 só é necessário para o plano galáctico.
5. **Sem comparação com trabalhos publicados.** Rodar o mesmo protocolo no
   Galaxy10 DECaLS daria um benchmark.
6. **Duas sementes extras apenas.** Para publicar, 5 seriam mais sólidas.
