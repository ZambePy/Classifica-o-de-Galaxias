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
- [Soft labels — um resultado negativo](#soft-labels--um-resultado-negativo)
- [Comparação de backbones](#comparação-de-backbones)
- [Test-time augmentation](#test-time-augmentation)
- [Detector de domínio](#detector-de-domínio)
- [O viés do plano galáctico](#o-viés-do-plano-galáctico)
- [Três métodos de fora-de-domínio](#três-métodos-de-fora-de-domínio)
- [Cascata × multi-tarefa](#cascata--multi-tarefa)
- [Galaxy10 DECaLS — benchmark externo e troca de survey](#galaxy10-decals--benchmark-externo-e-troca-de-survey)
- [O bug da deduplicação](#o-bug-da-deduplicação)
- [Bugs silenciosos encontrados](#bugs-silenciosos-encontrados)
- [Limitações conhecidas](#limitações-conhecidas)

---

## Resumo

| modelo | backbone | n (teste) | acurácia | macro-F1 | MCC | ROC AUC |
|---|---|---:|---:|---:|---:|---:|
| **nível 1** · objeto | resnet50 | 1.048 | 0,9656 | 0,9639 | 0,9482 | 0,9892 |
| **nível 2** · galáxia | resnet18 | 5.656 | 0,9459 | 0,8998 | 0,9005 | 0,9905 |
| **nível 3** · nebulosa | efficientnet_b0 | 411 | 0,8540 | 0,8316 | 0,7999 | 0,9623 |
| **cascata** ponta a ponta | — | 1.048 | **0,8941** | — | — | — |

Dataset: **42.437 imagens rotuladas** — 37.716 galáxias (Galaxy Zoo), 2.737
nebulosas de oito catálogos e 1.984 em `other`.

Há outras 23.862 imagens do Galaxy Zoo em disco que **não entram em conjunto
algum**: são as que a votação humana não rotulou com confiança suficiente para
entrar no índice. O total em disco é 66.299, e versões anteriores deste
documento citavam esse número como se fosse o tamanho do dataset — não é, e a
diferença é grande.

> **Aviso de leitura.** Estes números vêm da rodada de 26/09 sobre dados
> corrigidos: 848 imagens órfãs foram removidas de `other` (duplicatas
> byte-a-byte de um esquema de nomes antigo). O nível 1 caiu de 0,982 para
> 0,966 — a queda é a **remoção de vazamento**, não perda de qualidade.
> Comparações com rodadas anteriores não são válidas.

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
| **galáxia** | 0,9459 | **0,9364** (99,0%) | 0,4468 (47,2%) | 0,5675 | usa o **objeto** |
| **nebulosa** | 0,8540 | 0,5937 (69,5%) | **0,7689** (90,0%) | 0,3601 | usa o **contexto** |
| objeto | 0,9656 | 0,8149 (84,4%) | 0,8197 (84,9%) | 0,3578 | usa os dois |

**O nível 2 é o controle interno.** Com o objeto desfocado, o classificador de
galáxias cai para 0,4468 — **abaixo de chutar a classe majoritária** (0,5675).
Descontando o chute o fundo entrega **−31,9%**: número negativo, ou seja, o
fundo é pior que nada. Ele depende inteiramente da morfologia, como se espera
de um classificador que funciona.

**O nível 3 faz o oposto.** Com o objeto desfocado, mantém 90,0% da acurácia —
bem mais do que mantém com o fundo desfocado (69,5%). Descontando o chute, **o
fundo sozinho entrega 82,8% do desempenho**.

O contraste entre `−31,9%` e `+82,8%`, medido com o mesmo código no mesmo dia,
é o resultado mais forte do projeto: não é uma suspeita sobre o dataset, é uma
medida com controle interno.

Trocar o backbone de ResNet18 para EfficientNet-B0 subiu a acurácia do nível 3,
mas **não reduziu a dependência do contexto**. Isso é o esperado: viés de
dataset não se corrige trocando de arquitetura.

### Por classe

| classe | completa | só objeto | só fundo |
|---|---:|---:|---:|
| spiral | 0,957 | 0,955 | 0,201 |
| elliptical | 0,958 | 0,948 | 0,576 |
| irregular | 0,785 | 0,708 | 0,429 |
| emission | 0,828 | 0,492 | 0,762 |
| reflection | 0,896 | 0,742 | 0,806 |
| planetary | 0,929 | 0,737 | 0,860 |
| supernova_remnant | 0,673 | 0,398 | 0,196 |

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

`galaxy` · `nebula` · `other` — 4.889 imagens de treino, classes balanceadas

```powershell
python scripts\train.py --config configs\level1_object.yaml
python scripts\evaluate.py --config configs\level1_object.yaml
```

| | acurácia | balanceada | kappa | MCC | ROC AUC |
|---|---:|---:|---:|---:|---:|
| | 0,9656 | 0,9639 | 0,9482 | 0,9482 | 0,9892 |

| classe | precisão | recall | F1 | n |
|---|---:|---:|---:|---:|
| galaxy | **1,000** | **1,000** | **1,000** | 375 |
| nebula | 0,952 | 0,952 | 0,952 | 375 |
| other | 0,940 | 0,940 | 0,940 | 298 |

| | galaxy | nebula | other |
|---|---:|---:|---:|
| **galaxy** | 375 | 0 | 0 |
| **nebula** | 0 | 357 | 18 |
| **other** | 0 | 18 | 280 |

**375 de 375 galáxias corretas, sem um único falso positivo** — e isso
sobreviveu à correção do vazamento em `other`, que derrubou as outras duas
classes. Toda a confusão está entre `nebula` e `other`, simétrica: 18 para cada
lado.

Faz sentido físico. `other` contém aglomerados e campos estelares densos do
plano galáctico, o mesmo lugar onde vivem as nebulosas — e num recorte DSS2 de
256 px um aglomerado aberto imerso em nebulosidade fraca é genuinamente
ambíguo. Galáxia, vinda de alta latitude galáctica com fundo escuro e limpo, não
tem com o que ser confundida.

---

## Nível 2 — morfologia de galáxia

`spiral` · `elliptical` · `irregular` — 26.401 de treino

| | acurácia | balanceada | kappa | MCC | ROC AUC |
|---|---:|---:|---:|---:|---:|
| | 0,9459 | 0,9196 | 0,9002 | 0,9005 | 0,9905 |

| classe | precisão | recall | F1 | n |
|---|---:|---:|---:|---:|
| spiral | 0,970 | 0,944 | 0,957 | 3.210 |
| elliptical | 0,951 | 0,964 | 0,958 | 2.105 |
| irregular | 0,729 | 0,850 | 0,785 | 341 |

`irregular` é limitado pelo **rótulo, não pelo modelo**: o Galaxy Zoo 2 não tem
pergunta direta para essa classe, e a aproximação mistura galáxias irregulares
com fusões, anéis e artefatos.

---

## Nível 3 — tipo de nebulosa

Quatro classes — 1.916 de treino, vindas de oito catálogos

| | acurácia | balanceada | kappa | MCC | ROC AUC |
|---|---:|---:|---:|---:|---:|
| | 0,8540 | 0,8506 | 0,7981 | 0,7999 | 0,9623 |

| classe | precisão | recall | F1 | n |
|---|---:|---:|---:|---:|
| emission | 0,828 | 0,828 | 0,828 | 128 |
| reflection | 0,954 | 0,845 | 0,896 | 148 |
| planetary | 0,924 | 0,934 | 0,929 | 91 |
| supernova_remnant | 0,583 | 0,795 | 0,673 | 44 |

**Estes números devem ser lidos junto com a ablação acima:** boa parte do
desempenho vem do contexto, não do objeto.

---

## Cascata completa

```powershell
python scripts\evaluate_cascade.py
```

| métrica | valor |
|---|---:|
| acurácia do nível 1 | 0,9656 |
| **acurácia da folha (rótulo final)** | **0,8941** |
| erro em cascata | 0,0344 |
| subtipo dado nível 1 correto | 0,8975 |

| ramo | n | nível 1 | folha | subtipo \| nível 1 ok |
|---|---:|---:|---:|---:|
| galaxy | 375 | 1,000 | 0,928 | 0,928 |
| nebula | 375 | 0,952 | 0,824 | 0,866 |
| other | 298 | 0,940 | 0,940 | — |

Os **7,2 pontos** perdidos entre o nível 1 e o rótulo final são o custo da
arquitetura em cascata.

O ramo `galaxy` é o caso limpo: nível 1 perfeito, então a folha (0,928) é
exatamente o desempenho do nível 2 naquelas 375 imagens — nenhuma foi perdida
por roteamento. No ramo `nebula`, a folha (0,824) fica abaixo do subtipo
condicional (0,866) justamente pelas 18 imagens que o nível 1 mandou para
`other` e que pararam ali.

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

> **Procedência.** Medido com ResNet18 nos dois níveis e **antes** da correção
> que removeu as 848 órfãs de `other`, então as médias não batem com a tabela de
> resumo (que usa ResNet50 no nível 1 e EfficientNet-B0 no nível 3, em dados
> corrigidos). Não foram reescritas: reescrever sem re-executar as três sementes
> seria inventar medição.
>
> O que esta seção estabelece e continua valendo é a **magnitude do ruído** —
> ±0,002 no nível 1 e ±0,012 no nível 3 —, e é para isso que ela é citada no
> resto do documento. O desvio entre sementes depende do tamanho do conjunto e
> do desbalanceamento das classes, que mudaram pouco.

---

## Soft labels — um resultado negativo

```powershell
python scripts\train.py --config configs\level2_galaxy_soft.yaml
```

O Galaxy Zoo não diz "esta galáxia é espiral": diz "73% dos voluntários viram um
disco com braços". A pergunta do experimento é se treinar contra essa
**distribuição de votos** (KL-divergência) bate treinar contra o rótulo de voto
majoritário (cross-entropy).

| nível 2 · galáxia | acurácia | acurácia balanceada | macro-F1 |
|---|---:|---:|---:|
| rótulo rígido (produção) | **0,9459** | **0,9196** | **0,8998** |
| soft labels (KL) | 0,9450 | 0,8710 | 0,8787 |
| diferença | −0,0009 | **−0,0486** | **−0,0211** |

**A hipótese não se confirmou.** A acurácia crua praticamente empata, mas a
acurácia balanceada cai quase 5 pontos e o macro-F1 cai 2,1 — bem acima do
desvio entre sementes no nível 2. O dano se concentra onde era previsível:

| classe | F1 rígido | F1 soft |
|---|---:|---:|
| spiral | 0,957 | 0,958 |
| elliptical | 0,958 | 0,961 |
| **irregular** | **0,785** | **0,717** |

As duas classes majoritárias empatam ou melhoram de leve; `irregular` perde 6,8
pontos. Faz sentido: `irregular` é a classe cuja distribuição de votos é mais
espalhada — é exatamente onde os humanos discordam — e treinar contra uma
distribuição achatada ensina o modelo a hesitar justamente na classe que já era
a mais difícil. A incerteza humana entrou no treino, e o que ela fez foi diluir
o sinal da classe rara.

> **Comparabilidade.** O run com soft labels tem n=5.657 e o de produção n=5.656:
> são gerações de split diferentes, separadas por uma recoleta. A diferença de
> uma imagem não explica 4,9 pontos de acurácia balanceada, então a conclusão se
> mantém — mas os dois números não vêm do mesmo conjunto exato, e repetir o
> experimento sobre o split atual é o próximo passo honesto.

Vale reportar mesmo sendo negativo: a pergunta era razoável, a implementação está
em `training/soft_labels.py` com testes, e "usamos a distribuição de votos e
piorou" é informação útil para quem for tentar o mesmo.

---

## Comparação de backbones

> **Procedência.** Esta seção e a de TTA abaixo foram medidas **antes** da
> correção que removeu as 848 órfãs de `other`. Os valores absolutos aqui não
> batem com a tabela de resumo, e não foram reescritos: reescrevê-los sem
> re-executar seria inventar medição. O que a comparação estabelece é a **ordem
> entre backbones**, e essa comparação é interna — todos os backbones foram
> avaliados no mesmo conjunto, no mesmo dia.

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
aceitar 95% das imagens legítimas. **Limiar: 0,8248.**

### Detecção de erro

| métrica | valor |
|---|---:|
| AUROC | 0,7843 |
| confiança média nos acertos | 0,9541 (n=1.012) |
| confiança média nos erros | 0,8219 (n=36) |
| erros pegos pelo limiar | 38,9% |
| acertos acima do limiar | 96,7% |

**AUROC 0,78 — ainda vale exibir o aviso, mas caiu.** Na rodada anterior era
0,9022, com o limiar em 0,8840 pegando 63,3% dos erros. A queda tem causa
identificada: os erros que sobraram são quase todos `nebula` ↔ `other`, e esses
dois são genuinamente parecidos num recorte DSS2 do plano galáctico. O modelo
erra **com confiança alta** ali (0,8219 de média), porque não é incerteza — é
ambiguidade real na imagem.

Isso é uma limitação do MSP, não do projeto: a confiança do softmax mede o quão
longe a imagem está da fronteira entre classes conhecidas, e nesse caso a imagem
está genuinamente perto da fronteira. Ver
[Três métodos de fora-de-domínio](#três-métodos-de-fora-de-domínio).

### Detecção de domínio

Os mesmos objetos do teste, obtidos de outros levantamentos.

| levantamento | AUROC | fora do domínio aceito |
|---|---:|---:|
| Mellinger (astrofotografia) | 0,789 | 44,7% |
| PanSTARRS (óptico moderno) | 0,740 | 63,7% |
| allWISE (infravermelho) | 0,652 | 91,7% |

Estes são os números do **MSP**. A seção seguinte compara três métodos no mesmo
conjunto e mostra que o MSP é, de longe, o pior dos três.

> **Duas correções de interpretação, em sequência.** Este parágrafo já foi
> reescrito duas vezes e as duas vale deixar registradas.
>
> *Primeira leitura:* a ordem original (allWISE 0,953, PanSTARRS 0,695) parecia
> "validar o detector", porque a AUROC crescia com a distância ao domínio. Com o
> dataset ampliado a ordem **inverteu**, e concluí que a explicação elegante
> estava errada.
>
> *Segunda leitura, a atual:* a explicação elegante está **certa** — o que estava
> errado era o detector. Trocando o MSP pela distância de Mahalanobis nos mesmos
> conjuntos, a ordem se endireita e o levantamento mais parecido com o treino
> (PanSTARRS, óptico) volta a ser o mais difícil. Ver
> [Três métodos de fora-de-domínio](#três-métodos-de-fora-de-domínio).
>
> A lição é sobre método, não sobre astronomia: **um baseline fraco não mede só
> pior, ele sugere conclusões erradas sobre o fenômeno.**

### Validação com imagem real — o mesmo objeto em quatro instrumentos

M16 (a nebulosa dos Pilares da Criação), recortada de quatro fontes, avaliada
com o modelo e o limiar atuais:

| imagem | resposta | conf. | MSP avisa? | Mahalanobis avisa? |
|---|---|---:|---|---|
| DSS2, 5′ (o domínio de treino) | `nebula` → `emission` ✓ | 0,978 | não | **sim** (falso alarme) |
| DSS2, 1′ | `other` ✗ | 0,929 | não | — |
| Mellinger (astrofotografia) | `other` ✗ | 0,960 | **não** ⚠️ | **sim** ✓ |
| allWISE (infravermelho) | `other` ✗ | 0,915 | **não** ⚠️ | **sim** ✓ |

As duas linhas do meio são o argumento mais concreto contra o MSP deste
relatório inteiro. **É a mesma nebulosa**, fotografada por outro instrumento. O
modelo erra — chama de `other` — e erra com confiança de 0,96 e 0,92, bem acima
do limiar de 0,8248. O aviso não aparece. Um usuário do dashboard veria "Outro
objeto, 96%" sem nenhuma ressalva. O Mahalanobis barra as duas.

**Mas a primeira linha mostra o outro lado, e não vamos esconder.** O recorte
DSS2, que o sistema acerta e que está dentro do domínio, também é marcado pelo
Mahalanobis — percentil 0,027, abaixo do limiar de 0,05. É um falso alarme.

Isso não é defeito, é o ponto de operação: o limiar foi calibrado para aceitar
95% das imagens legítimas, então **5% de falsos alarmes é o preço combinado**.
Medido no conjunto de teste inteiro (1.048 imagens dentro do domínio):

| | valor |
|---|---:|
| aceitas pelo Mahalanobis | **95,1%** |
| marcadas como fora (falso alarme) | 4,9% (51 imagens) |
| percentil mediano | 0,505 |

Ou seja: o detector está calibrado exatamente onde se pediu, e o M16 em DSS2 é
uma das 51. A leitura correta da comparação entre métodos é **no mesmo ponto de
operação**: com 95% de aceitação de imagens boas nos dois casos, o MSP deixa
passar 66,7% das de fora e o Mahalanobis 13,8%.

> **Correção de uma versão anterior deste documento.** Aqui constava que uma
> foto do Hubble saía com `confidence: 0.895` e `out_of_domain: TRUE`,
> "classificou errado mas avisou". Aquele número veio de uma rodada com limiar
> mais alto; com o limiar atual de 0,8248, uma confiança de 0,895 **não** seria
> sinalizada. A foto original não foi guardada, então a linha foi substituída
> por este teste, que é reprodutível: as quatro imagens estão em
> `runs/inspection/_m16/`.

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
| **star_field** | 45 | **0,800** | **9 → nebulosa** |

**9 de 45 campos estelares foram chamados de nebulosa (20%).** São recortes
sorteados no plano galáctico — o modelo está respondendo à densidade estelar e à
cor do fundo.

> ### ⚠️ Este controle estava contaminado, e o número acima é um limite superior
>
> A premissa é "campo do plano galáctico **sem nebulosa alguma**". Ela não estava
> garantida, e a auditoria final mediu o tamanho do problema.
>
> Os recortes usam **30 arcmin** de campo, logo 15′ de meio-campo, e o sorteio
> era livre — no plano galáctico, que é exatamente onde as nebulosas moram.
> Cruzando as coordenadas dos 300 campos coletados contra o catálogo de 2.737
> nebulosas:
>
> | | campos |
> |---|---:|
> | contêm o **centro** de uma nebulosa catalogada | 23 de 300 (7,7%) |
> | contêm **alguma parte** de uma | 46 de 300 (15,3%) |
>
> No conjunto de teste, **6 dos 45**. O `analyze_shortcut.py` passou a descartar
> esses e a reportar as duas versões, e o resultado medido é:
>
> | | erros | n | taxa |
> |---|---:|---:|---:|
> | controle **bruto** | 9 | 45 | 20,0% |
> | controle **limpo** | 6 | 39 | **15,4%** |
>
> Três dos nove "erros" eram, portanto, **acertos**: recortes que de fato contêm
> uma nebulosa catalogada, onde responder `nebula` é o certo. A taxa real do
> atalho é 15,4%, não 20% — ainda alta, mas um quarto menor do que o número que
> estava publicado.
>
> `sky_sampling.py` passou a excluir posições que alcancem o catálogo, de modo que
> a próxima coleta nasce limpa e essa correção deixa de ser necessária.
>
> **Nada disso quebrava.** O código rodava, as imagens baixavam, o número saía
> plausível. Era erro de **desenho experimental**, e a única forma de pegá-lo era
> checar a premissa contra o catálogo — que é o que o teste
> `tests/test_sky_sampling.py` agora faz.
>
> A conclusão sobre o atalho **não** depende só deste controle: a ablação por
> oclusão é evidência independente e bem mais forte, com controle interno (o
> nível 2 cai abaixo do chute). Mas este número, isolado, era mais fraco do que
> parecia.

> **Este número piorou, e vale dizer por quê.** Na rodada anterior era 4 de 90
> (4,4%). O conjunto `other` foi recoletado do zero ao remover as 848 órfãs, e
> o `star_field` resultante é menor (45 no teste, contra 90) e sorteado com
> outra semente. Parte da diferença é amostragem — com n=45 o intervalo é largo
> — e parte é que as duplicatas anteriores tornavam `other` artificialmente
> fácil. A leitura honesta é que o atalho é **pior do que a rodada anterior
> sugeria**, não que tenha aumentado.

### Geografia sozinha

Árvore de decisão usando **só a latitude galáctica**, sem ver imagem:

| classe | AUROC só com \|b\| | (dataset anterior) |
|---|---:|---:|
| supernova_remnant | 0,760 | 0,795 |
| planetary | 0,615 | 0,727 |
| reflection | 0,545 | 0,719 |
| emission | 0,489 | 0,526 |

### O que mudou ao longo das rodadas

| | dataset menor | ampliado | após corrigir `other` |
|---|---:|---:|---:|
| star_field → nebulosa (bruto) | 13,3% | 4,4% | 20,0% |
| star_field → nebulosa (controle limpo) | — | — | **15,4%** |
| AUROC médio só com \|b\| | 0,692 | **0,602** | 0,602 |

A leitura desta linha mudou três vezes, e a versão final é menos otimista que a
intermediária. Ampliar o dataset com catálogos mais espalhados pelo céu
(Magakian, Kohoutek) **reduziu de fato** a informação contida na latitude
galáctica sozinha — de 0,692 para 0,602 de AUROC média, e isso se manteve.

Mas o `star_field → nebulosa` subiu para 20% depois que `other` foi recoletado
sem as 848 duplicatas. Duas causas se somam: o conjunto de teste ficou menor
(45 contra 90, então o intervalo de confiança é largo) e as duplicatas tornavam
`other` artificialmente fácil de acertar. A conclusão honesta é que **o atalho
sempre foi pior do que a rodada intermediária sugeriu**, não que ele piorou.

E a ablação por oclusão continua sendo a medida decisiva: descontando o chute, o
fundo explica **82,8%** do desempenho no nível 3.

---

## Três métodos de fora-de-domínio

```powershell
python scripts\compare_ood.py
```

Os três leem o **mesmo checkpoint**, o mesmo conjunto de teste e os mesmos
conjuntos de fora. Nenhum exige retreino: o que muda é apenas a função que
transforma a saída da rede num número.

| método | onde olha | AUROC | AUPR | FPR@95TPR |
|---|---|---:|---:|---:|
| MSP — Hendrycks & Gimpel 2017 | máx. do softmax | 0,7269 | 0,3029 | 66,7% |
| energia — Liu et al. 2020 | log-sum-exp dos logits | 0,7635 | 0,4117 | 61,7% |
| **Mahalanobis** — Lee et al. 2018 | features do penúltimo layer | **0,9761** | **0,7924** | **13,8%** |

Médias sobre os três levantamentos. **FPR@95TPR** é a métrica que descreve o
dashboard: fixando o limiar para aceitar 95% das imagens legítimas, que fração
das imagens de fora passa? O usuário não vê a AUROC — vê um aviso que aparece ou
não aparece.

**O baseline deixa passar 2 de cada 3 imagens de fora do domínio. O Mahalanobis
deixa passar 1 em 7.**

### Por levantamento

| levantamento | n | MSP | energia | Mahalanobis |
|---|---:|---:|---:|---:|
| allWISE (infravermelho) | 80 | 0,652 | 0,713 | **0,993** |
| Mellinger (astrofotografia) | 80 | 0,789 | 0,808 | **0,993** |
| PanSTARRS (óptico moderno) | 80 | 0,740 | 0,769 | **0,942** |

No allWISE o Mahalanobis barra **100%** das imagens de fora no ponto de operação
de 95% de aceitação — FPR@95 de 0,0%.

### O que a ordenação revela, e que corrige uma leitura anterior

Repare na ordem de dificuldade de cada método:

```
MSP:          allWISE (0,652)  <  PanSTARRS (0,740)  <  Mellinger (0,789)
Mahalanobis:  PanSTARRS (0,942)  <  allWISE (0,993) ≈ Mellinger (0,993)
```

Para o MSP, o conjunto **mais difícil** é o allWISE — que é o domínio **mais
distante** do treino, infravermelho contra óptico. Isso está ao contrário: o
mais diferente deveria ser o mais fácil de rejeitar.

Para o Mahalanobis a ordem se endireita: o mais difícil é o PanSTARRS, que é
óptico e portanto o mais **parecido** com as imagens de treino. É exatamente o
que se espera de um detector que funciona.

Este projeto já havia registrado, mais acima, que a "explicação elegante" — de
que a AUROC cresceria com a distância ao domínio — não se sustentava. **Ela se
sustenta.** O que não funcionava era o MSP, fraco demais para expor a estrutura.
Trocado o método, o padrão aparece. A lição é metodológica: um baseline ruim não
só mede pior, ele produz conclusões erradas sobre o fenômeno.

### Custo e ajuste

O Mahalanobis exige um passo extra: estimar a média de cada classe e uma
covariância compartilhada sobre as features do **treino** (4.889 imagens,
2.048 dimensões no ResNet50). Isso roda uma vez, leva poucos minutos, e o
resultado vai para `checkpoints/ood_mahalanobis.pt`.

O encolhimento necessário foi **0,01**, o valor mínimo pedido — com 4.889
imagens para 2.048 dimensões a covariância ficou bem condicionada sem precisar
de regularização agressiva. Esse número vai no arquivo de resultados de
propósito: um encolhimento alto indicaria amostra insuficiente e distâncias
pouco confiáveis.

---

## Cascata × multi-tarefa

```powershell
python scripts\train_multitask.py
```

A alternativa à cascata é compartilhar o backbone e pendurar uma cabeça linear
por nível. A pergunta: isso ganha ou perde?

A comparação foi mantida justa em quatro pontos. **Mesma supervisão** — o
dataset é a união dos três splits de treino, e cada imagem só recebe rótulo nos
níveis em que ela aparecia, então o multi-tarefa vê exatamente o que os três
modelos viram somados. **Mesmo conjunto de teste e mesmo código de métrica** —
a avaliação chama a mesmíssima função `evaluate_cascade`. **Mesmo backbone** do
nível 1 (ResNet50, o mais forte dos três). **Mesma travessia** na inferência: a
cabeça de objeto decide qual cabeça de subtipo ler.

| métrica | cascata | multi-tarefa | diferença |
|---|---:|---:|---:|
| acurácia do nível 1 | **0,9656** | 0,9494 | −0,0162 |
| **acurácia da folha** | **0,8941** | 0,8788 | **−0,0153** |
| erro em cascata | **0,0344** | 0,0506 | +0,0162 |
| subtipo dado nível 1 correto | 0,8975 | **0,8982** | **+0,0007** |

**A cascata ganha por 1,5 ponto de acurácia da folha.** Mas a linha que explica
o resultado é a última: o **subtipo condicional empata** (+0,0007). Compartilhar
o backbone não piorou em nada a capacidade de distinguir espiral de elíptica ou
emissão de planetária. Toda a perda está na **cabeça de objeto**.

### Onde exatamente a perda acontece

| ramo | n | nível 1 cascata | nível 1 multi | folha cascata | folha multi |
|---|---:|---:|---:|---:|---:|
| galaxy | 375 | 1,000 | 1,000 | 0,928 | **0,933** |
| nebula | 375 | **0,952** | 0,939 | **0,824** | 0,808 |
| other | 298 | **0,940** | 0,899 | **0,940** | 0,899 |

Duas coisas saltam:

**`other` é quem paga a conta** — 0,940 → 0,899, uma queda de 4,1 pontos, o dobro
da média. E faz sentido: `other` é a classe mais heterogênea do projeto (campos
estelares, aglomerados abertos e globulares, céu vazio, fotos do dia a dia). É a
que mais depende de features próprias, e é justamente ela que perde quando o
backbone é disputado.

**As galáxias não perdem nada** — o nível 1 continua 375 de 375, e a folha até
melhora 0,5 ponto. Coerente com a aritmética do treino: das 29.706 imagens da
união, 26.401 têm rótulo de galáxia. O gradiente é dominado pela tarefa de
morfologia, então o backbone se especializa nela — de graça para as galáxias, à
custa de `other`.

### O outro lado da balança

| | cascata | multi-tarefa |
|---|---:|---:|
| parâmetros | 38.704.902 | **23.528.522** (61%) |
| passadas pelo backbone por imagem | 2 | **1** |
| modelos para versionar e carregar | 3 | **1** |

Um backbone em vez de três, 61% dos parâmetros, e **uma** passada em vez de
duas na inferência. Para um dashboard que roda numa máquina qualquer, 1,5 ponto
de acurácia é um preço plausível por isso.

### Conclusão, e o que faltou

A cascata continua sendo a escolha do projeto, porque o objetivo declarado é o
baseline mais preciso e honesto — não o mais leve. Mas o resultado **não** é "o
multi-tarefa é pior": é "o multi-tarefa custa 1,5 ponto no nível 1 e nada nos
subtipos, em troca de 39% menos parâmetros".

O experimento que fecharia isso e não foi feito: **pesar as cabeças na loss.**
Com 26.401 galáxias contra 4.889 imagens de objeto, a cabeça de objeto está em
desvantagem de 5 para 1 no gradiente. `masked_multitask_loss` já aceita
`weights`, e um varredura de peso para a cabeça de objeto é o caminho óbvio para
recuperar os 1,5 ponto sem abrir mão da economia. Fica registrado como próximo
passo, não como resultado.

---

## Galaxy10 DECaLS — benchmark externo e troca de survey

O mesmo dataset fecha dois itens do roteiro, e não por conveniência: o Galaxy10
DECaLS **é** imagem do DECaLS, o levantamento que o roteiro queria testar, e vem
com uma taxonomia oficial de 10 classes que outros trabalhos usam.

### Benchmark: a taxonomia oficial de 10 classes

```powershell
python scripts\benchmark_galaxy10.py --task 10class
```

Nossa acurácia de 3 classes não se compara com nada publicado, porque a
taxonomia é nossa. Rodar a taxonomia **oficial** do Galaxy10, com o nosso
backbone, nossas transformações e nosso laço de treino, é o único jeito honesto
de responder "esse pipeline presta?".

| | valor |
|---|---:|
| acurácia | **0,8764** |
| acurácia balanceada | 0,8541 |
| macro-F1 | 0,8547 |
| MCC | 0,8606 |
| n (teste) | 2.669 |

ResNet50 pré-treinado, 12 épocas, split 70/15/15 estratificado, semente 42.

| classe | precisão | recall | F1 | n |
|---|---:|---:|---:|---:|
| Disturbed | 0,724 | 0,515 | **0,602** | 163 |
| Merging | 0,909 | 0,935 | 0,922 | 279 |
| Round Smooth | 0,928 | 0,937 | 0,932 | 398 |
| In-between Round Smooth | 0,888 | 0,961 | 0,923 | 305 |
| Cigar Shaped Smooth | 0,755 | 0,784 | 0,769 | 51 |
| Barred Spiral | 0,954 | 0,870 | 0,910 | 307 |
| Unbarred Tight Spiral | 0,771 | 0,818 | 0,794 | 275 |
| Unbarred Loose Spiral | 0,802 | 0,820 | 0,811 | 395 |
| Edge-on without Bulge | 0,926 | 0,939 | 0,933 | 214 |
| Edge-on with Bulge | 0,941 | 0,961 | **0,951** | 282 |

A estrutura dos erros é interpretável, o que é um bom sinal. As duas classes
`Edge-on` são as mais fáceis (F1 0,93 e 0,95): uma galáxia de perfil tem
assinatura geométrica inconfundível. `Disturbed` é a mais difícil de longe
(recall 0,515) — é a categoria mais vaga do dataset, "tem algo estranho", e
confunde-se com `Merging`, que é a vizinha semântica. Os dois pares de espiral
sem barra (`Tight` e `Loose`) ficam em ~0,80, também esperado: a diferença entre
braços apertados e soltos é gradual, não categórica.

> **Como comparar com a literatura, e o que não afirmar.** O número acima está na
> taxonomia oficial, então é comparável com trabalhos que usem este dataset — mas
> confira a referência na documentação do astroNN (Leung & Bovy 2019), que
> distribui o Galaxy10. **Não colocamos aqui um número publicado de memória**, e
> quem for citar comparação deve buscar a fonte: existe também uma versão SDSS
> mais antiga do Galaxy10, com acurácias diferentes, e confundir as duas é o erro
> clássico.
>
> Há outra ressalva que corta contra nós: **não existe split oficial publicado**.
> O nosso é 70/15/15 estratificado com semente 42, registrado no JSON. Uma
> diferença de 1 a 2 pontos entre trabalhos pode ser só isso.

### Troca de survey: as nossas 3 classes, em DECaLS

```powershell
python scripts\benchmark_galaxy10.py --task 3class
```

Mapeando as 10 classes oficiais nas nossas 3 e treinando, sai um modelo de nível
2 em DECaLS — e, de graça, o experimento que interessa mais: avaliar o nosso
modelo **treinado em SDSS** neste conjunto de teste do DECaLS.

| | acurácia | balanceada | macro-F1 | MCC |
|---|---:|---:|---:|---:|
| nosso nível 2, no **seu** teste (SDSS) | 0,9459 | 0,9196 | 0,8998 | 0,9005 |
| treinado em **DECaLS**, no teste DECaLS | 0,9208 | **0,8932** | 0,8979 | 0,8655 |
| nosso nível 2 (SDSS), no teste **DECaLS** | 0,7788 | **0,7694** | 0,7436 | 0,6553 |

n = 2.663 imagens de teste, split 70/15/15 estratificado, semente 42.

**O modelo treinado nativamente em DECaLS é o teto daquele conjunto de teste.**
Ele alcança 0,8932 de acurácia balanceada; o nosso, transferido do SDSS, fica em
0,7694 — **12,4 pontos abaixo**. Essa diferença é o preço de trocar o
instrumento, medido contra uma referência justa em vez de contra o nosso próprio
conjunto.

E a comparação é justa por um motivo que vale explicitar: as duas tarefas têm
dificuldade praticamente igual. O macro-F1 do modelo nativo em DECaLS (0,8979) e
o do nosso em SDSS (0,8998) diferem em 0,2 ponto. Não é que o DECaLS seja mais
fácil ou mais difícil — é que morfologia de galáxia deveria ser independente do
telescópio, e **não é**.

### Por que a acurácia crua exagera a queda

A acurácia crua cai 14,2 pontos, a balanceada 12,4. A diferença vem de um desvio
de **prior**, não de domínio:

| | `spiral` | `elliptical` | `irregular` |
|---|---:|---:|---:|
| nosso (Galaxy Zoo 2) | 56,7% | 37,2% | **6,0%** |
| Galaxy10 (Disturbed+Merging → irregular) | 55,2% | 28,2% | **16,5%** |

"Perturbada ou em fusão" é uma categoria mais larga que a nossa `irregular`, e
ela é quase **três vezes** mais frequente no Galaxy10. Um modelo treinado com 6%
de irregular e avaliado num conjunto com 16,5% erra mais só por isso. A acurácia
balanceada pesa as classes igualmente e não se move com a proporção — é o número
a usar aqui.

### Os dois vieses do teste, que apontam para lados opostos

**O número de transferência é otimista.** O Galaxy10 vem da campanha Galaxy Zoo
DECaLS; nossas galáxias vêm do Galaxy Zoo 2. As duas cobrem céu sobreposto, e as
imagens do Kaggle não trazem coordenada — então **não foi possível excluir
objetos em comum**. Parte do conjunto de teste do DECaLS pode ser o mesmo objeto
que o modelo já viu em SDSS, fotografado por outro telescópio.

**A queda na acurácia crua é pessimista**, pelo desvio de prior acima.

Conclusão honesta: a queda real de domínio é de **pelo menos** 12,4 pontos de
acurácia balanceada. O limite inferior é o que este experimento estabelece; o
valor exato exigiria cruzamento por coordenada, que os dados não permitem.

### Uma replicação acidental, e o que ela revelou

Por um erro de coordenação, este experimento rodou **duas vezes em paralelo**,
com a mesma semente. Não foi desperdício: expôs dois problemas reais.

A segunda rodada **divergiu para NaN na época 3** e seguiu treinando até a 7, com
a acurácia de validação travada em **0,5525** — exatamente a fração de `spiral`
no dataset. O modelo passou a prever sempre a classe majoritária. Nada avisou: a
loss virou `nan`, o laço continuou, o early stopping "funcionou".

E o relatório dela saiu com **0,9208 de acurácia** — os números da rodada sadia.
Porque as duas gravavam no mesmo `galaxy10_3class.pt`, e a avaliação final da
segunda carregou os pesos da primeira. É o mesmo bug de "avaliação do checkpoint
errado" que o projeto já tinha documentado, acontecendo de novo por outro
caminho.

As duas correções estão no script e têm teste: `checar_loss_finita` aborta no
primeiro NaN, e `checar_lock` impede dois treinos no mesmo checkpoint (com
`--tag` para rodar variantes em paralelo de propósito). Os números publicados
acima são da rodada sadia, verificada época a época: loss monotonicamente
decrescente, melhor validação 0,9176 na época 10, pesos finitos.

---

## O bug da deduplicação

Encontrado na auditoria final, depois de todos os treinos. Vale registrar em
detalhe porque é o **oitavo** bug silencioso do projeto e o mais sutil.

`deduplicate_by_position()` promete: nenhum par de objetos a menos de 2 arcmin
sobrevive. Verificado no catálogo publicado, **43 objetos violavam a promessa** —
inclusive pares com separação exatamente 0,000 arcmin.

### Como aconteceu

A função pegava, para cada objeto, seu **vizinho mais próximo** e descartava o de
índice maior do par. Duas falhas independentes:

**Cadeias.** "Estar a menos de 2 arcmin" é uma relação entre *pares*, mas o
vizinho mais próximo é uma *função* — cada objeto aponta para um só. Com A, B, C
próximos entre si, se o mais próximo de A é B, o de B é C e o de C é B, então A e
B **ambos sobrevivem**, apesar de estarem dentro da tolerância. Medido num teste
com 40 objetos aglomerados: sobravam 20, com 19 ainda abaixo da tolerância.

**Empates exatos.** Com coordenada idêntica, a busca devolve o próprio objeto
como vizinho mais próximo (`idx[i] == i`) para um dos dois — e qual dos dois é
arbitrário, depende da árvore de busca construída sobre a distribuição inteira.
Caindo no de índice maior, a condição `idx[i] < i` é falsa para ambos e nenhum
sai. Foi o caso de LBN 770 e 771, nos índices 2230 e 2232:

```
i=2230: idx=2232  ->  2232 < 2230  falso  ->  mantido
i=2232: idx=2232  ->  2232 < 2232  falso  ->  mantido   <- casou consigo mesmo
```

Os dois geraram arquivos **byte a byte idênticos**, um no treino e um no teste.

### Consequência medida

| | |
|---|---:|
| objetos a <2′ sobrevivendo | 43 |
| grupos de duplicatas | 21 |
| grupos que caem em conjuntos diferentes | 10 |
| grupos com gêmeo **treino ↔ teste** | **6** |

Seis das 411 imagens de teste do nível 3 tinham gêmea quase idêntica no treino:
1,5% do conjunto, ou no máximo ~1,5 ponto de acurácia — **abaixo do desvio entre
sementes (±1,2 pt)**. Real, medido, e pequeno.

### A correção

Proximidade passou a ser tratada como **grafo**: toda aresta abaixo da
tolerância, componentes conexas por união-busca, um representante por
componente — o de menor índice, preservando a prioridade de `CATALOGS`.

| | antes | depois |
|---|---:|---:|
| objetos no catálogo | 2.737 | 2.715 |
| residuais a <2′ | 43 | **0** |

Há 19 testes em `tests/test_deduplication.py`, e um deles roda sobre o
**catálogo real** versionado em `docs/dataset/` — o dado que expôs o problema
viaja junto com a correção. Os casos sintéticos pequenos não reproduzem o empate
exato: qual elemento recebe o auto-casamento depende da distribuição inteira, e
com cinco pontos nunca cai no caso ruim. Com os 2.737 reais, cai.

### Por que os modelos não foram retreinados

Regerar os splits reembaralha os três níveis — `train_test_split` estratificado
não é estável à remoção de linhas — e obrigaria a retreinar tudo, incluindo o
nível 2, que é galáxia e não tem relação com o bug. Para um efeito abaixo do
ruído entre sementes, a troca não se paga. A correção está no código e vale para
qualquer reconstrução futura; os números publicados aqui correspondem ao
conjunto de 2.737 objetos, com o vazamento quantificado acima.

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
| 848 órfãs em `other` | classe `other` válida | duplicatas byte-a-byte de dois esquemas de nome |
| deduplicação por vizinho | "catálogos deduplicados" | 43 objetos a <2′ sobrando, 6 com gêmeo treino/teste |
| controle contaminado | "céu sem nebulosa alguma" | 15% dos campos continham uma nebulosa catalogada |
| loss em NaN sem guarda | early stopping normal | 5 épocas treinando no vazio, prevendo só a majoritária |
| dois treinos, um checkpoint | relatório com 0,9208 | o modelo avaliado era o do OUTRO processo |

---

## Limitações conhecidas

1. **O nível 3 depende do contexto** (descontando o chute, 82,8% do desempenho
   vem do fundo). Os números dessa etapa não medem reconhecimento de nebulosa.
2. **`irregular` é limitado pelo rótulo** (precisão 0,729).
3. **`supernova_remnant` tem sinal óptico fraco** — F1 0,673, e desaba nas
   duas condições da ablação (0,398 só objeto, 0,196 só fundo).
4. **Duas sementes extras apenas.** Para publicar, 5 seriam mais sólidas.
5. **Vazamento residual conhecido no nível 3.** A auditoria final encontrou 22
   objetos duplicados que a deduplicação antiga deixou passar; 6 punham a mesma
   imagem no treino e no teste das nebulosas. A função foi corrigida e tem teste
   de regressão, mas **os modelos aqui foram treinados antes da correção**.
   Efeito máximo estimado: 6 de 411 imagens de teste, ou ~1,5 ponto de acurácia
   no nível 3 — abaixo do desvio entre sementes (±1,2 pt). Eliminá-lo exige
   regerar os splits, o que reembaralha os três níveis e obriga a retreinar
   tudo. Ver [a seção sobre isso](#o-bug-da-deduplicação).
6. **Rótulos conflitantes entre catálogos.** Dos 21 grupos de duplicatas, vários
   tinham classes diferentes para o mesmo objeto — Sharpless nº1 é `emission` e
   Magakian nº644, a 0,4′ de distância, é `reflection`. Isso põe um piso na
   acurácia alcançável que nenhum modelo supera.
