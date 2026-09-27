# Datasets — de onde vem cada imagem

Este é o documento mais importante do projeto do ponto de vista acadêmico. A
arquitetura da rede é padrão e conhecida; **o dataset é a contribuição
original**. Documente aqui cada decisão, porque é isso que uma banca vai
questionar.

---

## O problema em uma frase

Para galáxias existe um dataset rotulado, grande e famoso. Para nebulosas não
existe nada equivalente. Este projeto resolve isso construindo o dataset de
nebulosas a partir de **catálogos astronômicos publicados** + um **serviço de
recorte do céu**.

---

## Galáxias — Galaxy Zoo

**Fonte:** Galaxy Zoo 2, através da competição no Kaggle
(*Galaxy Zoo — The Galaxy Challenge*).
Requer conta no Kaggle e aceitar as regras da competição para baixar.

**O que vem:** ~61.578 imagens de treino (recortes coloridos do SDSS, 424×424)
e um CSV com **37 colunas de probabilidade** por galáxia — a fração de
voluntários que respondeu cada item da árvore de decisão morfológica.

> ### ⚠️ Se você for usar o Galaxy10 DECaLS: o `.h5` tem uma armadilha de I/O
>
> O arquivo guarda `images` com `chunks=(555, 16, 16, 1)` e compressão gzip. Esse
> layout é ótimo para ler "o pixel (x,y) de todas as imagens" e **péssimo** para
> ler "uma imagem inteira" — que é exatamente o que um DataLoader faz.
>
> A conta: uma imagem de 256×256×3 se espalha por (256/16)×(256/16)×3 = **768
> chunks**, e cada chunk contém a mesma região de **555 imagens**. Para extrair
> 196 KB, o h5py descomprime ~109 MB. Amplificação de ~560×.
>
> Medido nesta máquina:
>
> | leitura | taxa | uma época de 12.414 imagens |
> |---|---:|---:|
> | imagem por imagem, direto do `.h5` | 1,7 img/s | ~2 horas |
> | fatia alinhada ao chunk (555) | 338 img/s | 37 s o dataset **todo** |
> | `.npy` memory-mapped, aleatória | **3.791 img/s** | **3,3 s** |
>
> A primeira versão de `benchmark_galaxy10.py` lia imagem por imagem: o treino
> ficava com a **GPU a 3–6% de uso**, esperando disco, e as 12 épocas levariam
> ~20 horas em vez de ~30 minutos. Nada acusava erro — o treino rodava, só
> rodava 2.000× mais devagar do que devia.
>
> `preparar_memmap()` resolve convertendo uma vez em fatias alinhadas ao chunk
> (cada chunk descomprimido uma vez, servindo as 555 imagens que contém) para um
> `.npy` de 3,5 GB lido com `mmap_mode='r'`. Depois disso o acesso aleatório é
> paginação do sistema operacional, não descompressão. A conversão é verificada
> byte a byte contra o `.h5`.
>
> **E há um segundo motivo, que só apareceu na prática.** A rodada lenta não
> apenas demorava: ela **quebrou**, depois de 45 minutos, com
>
> ```
> OSError: Can't synchronously read data (filter returned failure during read)
> ```
>
> uma falha do filtro gzip do h5py sob contenção de I/O (outro processo lia o
> mesmo arquivo ao mesmo tempo). Lendo do `.npy`, essa classe de falha
> desaparece: não há filtro para falhar. Milhões de descompressões por época não
> são só lentas — são milhões de oportunidades de erro transitório.

**Alternativa mais fácil:** o dataset *Galaxy10 DECaLS* (astroNN) já vem com
classes discretas e é bem menor. Serve para destravar o projeto rápido; o
Galaxy Zoo completo é mais rico para o lado acadêmico.

### A decisão metodológica que você precisa defender

O Galaxy Zoo **não diz** "esta galáxia é espiral". Ele diz "73% dos
voluntários viram um disco com braços". Transformar isso em três classes é uma
escolha sua, e `scripts/prepare_galaxy_zoo.py` implementa esta:

| Regra | Classe |
|---|---|
| `Class1.3 > 0.5` (estrela/artefato) | **descartada** — não é galáxia |
| `Class6.1 ≥ 0.8` ("tem algo estranho") | `irregular` |
| `Class1.1 ≥ 0.7` (lisa, sem features) | `elliptical` |
| `Class1.2 ≥ 0.7` (tem disco/features) | `spiral` |
| nenhuma acima | **descartada** — humanos não concordaram |

Duas consequências a assumir por escrito:

1. **Descartar ambíguas encolhe o dataset e melhora o rótulo.** É um
   trade-off deliberado, e aqui está medido — o efeito do `--min-vote` sobre
   quantas das 61.578 galáxias recebem rótulo:

   | `--min-vote` | aceitas | % |
   |---:|---:|---:|
   | 0,5 | 60.093 | 98% |
   | 0,6 | 49.003 | 80% |
   | **0,7** (padrão) | **37.716** | **61%** |
   | 0,8 | 25.591 | 42% |

   O padrão de 0,7 descarta **39%** das imagens. É muito, e é intencional: o
   preço é dado, o ganho é que cada rótulo restante tem concordância humana
   real. As 23.862 descartadas continuam em disco e simplesmente não entram em
   conjunto nenhum — não são deletadas, para que baixar o limiar seja só
   regerar um CSV.
2. **`irregular` é a classe mais frágil.** O Galaxy Zoo 2 não tem uma pergunta
   direta para "irregular"; `Class6.1` mistura galáxias genuinamente
   irregulares com fusões, anéis e artefatos. Se o F1 dessa classe ficar
   baixo, esta é a primeira suspeita — e dizer isso vale mais que esconder.

### O caminho mais interessante (Fase 2)

Em vez de reduzir as probabilidades a um rótulo único, **treinar com soft
labels**: usar a própria distribuição de votos como alvo, com KL-divergence no
lugar de cross-entropy. A incerteza humana entra no treino.

Isso é uma pergunta de pesquisa de verdade, e do tamanho certo para um
trabalho de curso técnico:

> Treinar com a distribuição completa de votos dos voluntários melhora o
> desempenho de um classificador morfológico em relação a treinar com o rótulo
> de voto majoritário?

---

## Nebulosas — catálogos + recortes do céu

Não existe "Galaxy Zoo das nebulosas". O que existe são **catálogos**: listas
de objetos com coordenadas e tipo, compiladas por astrônomos ao longo de
décadas e publicadas. A pipeline monta o dataset assim:

```
catálogo (nome, ra, dec, tipo)
        │        consultado via VizieR / astroquery
        ▼
    hips2fits (serviço do CDS)
        │        "me dê a imagem do céu nesta coordenada, neste tamanho"
        ▼
   imagem rotulada em disco
```

São **oito catálogos**. A coluna `Objetos` é a contribuição **depois** da
deduplicação — é ela que importa, porque catálogos se sobrepõem muito e a
contagem bruta engana.

| Classe | Catálogo | VizieR | Objetos | Referência |
|---|---|---|---:|---|
| `reflection` | Magakian (nebulosas de reflexão) | `J/A+A/399/141` | 888 | Magakian 2003, A&A 399, 141 |
| `emission` + `reflection` | Lynds Bright Nebulae | `VII/9` | 486 | Lynds 1965, ApJS 12, 163 |
| `planetary` | Kohoutek (planetárias galácticas) | `V/127A` | 346 | Kohoutek 2001 |
| `emission` | Sharpless (regiões HII, norte) | `VII/20` | 305 | Sharpless 1959, ApJS 4, 257 |
| `supernova_remnant` | SNRs galácticos | `VII/272` | 291 | Green 2009 |
| `planetary` | Strasbourg-ESO de PNe | `V/84/main` | 257 | Acker et al. 1992 |
| `emission` | RCW (regiões Hα, sul) | `VII/216` | 161 | Rodgers et al. 1960 |
| `reflection` | van den Bergh | `VII/21` | 3 | van den Bergh 1966, AJ 71, 990 |

Total depois de deduplicar: **2.737 objetos** — `reflection` 988, `emission`
855, `planetary` 603, `supernova_remnant` 291.

> **Os 3 objetos do van den Bergh não são erro de consulta.** O catálogo entra
> com 158 e sai com 3: praticamente todas as suas nebulosas de reflexão já
> estavam no Magakian, que é uma compilação mais recente e vem antes na ordem de
> prioridade. É a deduplicação funcionando, e o cartão do dataset registra o
> mesmo fato para os outros catálogos absorvidos.

> **Deduplicação corrigida em 26/09.** A versão anterior deixava 43 objetos a
> menos de 2 arcmin no catálogo, seis deles com a mesma imagem no treino e no
> teste. Uma reconstrução feita hoje produz **2.715** objetos, não 2.737. Os
> modelos publicados foram treinados com os 2.737 — ver
> [o bug da deduplicação](results.md#o-bug-da-deduplicação) para a medida do
> efeito.

### Por que o LBN foi acrescentado

O ramo das nebulosas entregava 0,774 na cascata contra 0,923 das galáxias, e
o Grad-CAM mostrava o modelo olhando o fundo. A causa é aritmética: **722
imagens de treino para quatro classes, contra 26 mil galáxias**. Com tão
pouco dado, decorar o contexto sai mais barato que aprender o objeto.

(Hoje o nível 3 tem 1.916 imagens de treino — as 722 são o número de antes da
expansão, que é o que motivou acrescentar catálogos.)

O Lynds Bright Nebulae resolve isso pelo caminho certo. Sua coluna `Color`
indica a cor dominante, que é o **critério físico** que separa os dois tipos:

| `Color` | significado | classe |
|---|---|---|
| 1 | azul dominante — luz estelar espalhada | `reflection` |
| 2 | neutro | **descartado** (ambíguo) |
| 3, 4 | vermelho dominante — emissão em Hα | `emission` |

O filtro `Bright <= 4` remove as nebulosas fracas demais para aparecer no
DSS2. Verificamos visualmente antes de adotar: `Color=1` mostra nebulosidade
azulada, `Color=4` mostra tons alaranjados difusos. A separação se sustenta.

**Ressalva honesta:** o rótulo vem da mesma propriedade (cor) que o modelo
vai usar para classificar. Não é circular — é assim que astrônomos
classificam, e a cor É a assinatura física — mas merece um parágrafo no
texto em vez de passar despercebido.

### Deduplicação: catálogos se sobrepõem

O LBN inclui dezenas de nebulosas que já estão em Sharpless, com outra
nomenclatura. Comparar nomes não resolve; comparar **coordenadas** sim — dois
objetos a menos de 2 arcmin um do outro são o mesmo objeto. Foram 70
duplicatas.

Sem isso, a mesma nebulosa entraria duas vezes e poderia cair metade no
treino e metade no teste — vazamento silencioso.

### Uma armadilha que custou 18% do dataset

Os catálogos numeram seus objetos de forma **independente**. A região nº 1 de
Sharpless, a nº 1 de RCW e a nº 1 de Lynds são três nebulosas diferentes, em
pontos distantes do céu. Como o nome do arquivo vinha só desse número, as
três viravam `emission/1.jpg` e duas se perdiam. **301 dos 1.667 objetos de
então sumiram**, e o log dizia "1667 salvos" — porque a gravação funcionou, só que
no mesmo arquivo.

Hoje `cutout_filename()` prefixa o nome com o catálogo de origem, e há teste
travando a regressão. **Lição geral: confira a contagem em disco, não o que o
log afirma.**

**Levantamento usado:** DSS2 color (`CDS/P/DSS2/color`). É o único que cobre o
céu inteiro — nebulosas galácticas ficam no plano da Via Láctea, onde o SDSS
não observou.

### Como rodar

```powershell
# 1. SEMPRE primeiro: veja o que os catálogos devolvem
python scripts\build_nebula_dataset.py --inspect

# 2. Teste com poucos objetos
python scripts\build_nebula_dataset.py --limit 20

# 3. Rode completo (leva horas)
python scripts\build_nebula_dataset.py
```

### Achado: `supernova_remnant` tem sinal óptico fraco

**Verificado com inspeção visual e Grad-CAM, não suposto.**

Quase todos os recortes de SNR são campos estelares sem estrutura visível. A
razão é física: a maioria dos remanescentes do catálogo de Green foi
descoberta em **rádio**. Eles são fracos em luz óptica e o DSS2 não os mostra.
Diferente das planetárias — onde o enquadramento resolveu o problema — aqui
não há enquadramento que recupere um sinal que não está na imagem.

O modelo ainda assim tira F1 ≈ 0,67 nessa classe. **O Grad-CAM mostra por
quê: ele acende na textura do fundo, não em nenhum objeto.** Os SNRs ficam no
plano galáctico, onde o campo estelar é denso e alaranjado — o modelo aprendeu
esse *contexto*, não o remanescente.

Três consequências a assumir:

1. O F1 dessa classe **não mede reconhecimento do objeto**. Reportá-lo sem
   essa ressalva seria enganoso.
2. O mesmo atalho contamina as outras nebulosas galácticas em menor grau — o
   Grad-CAM de `planetary` também acende no fundo com frequência.
3. `star_field` (classe `other`) é sorteada no mesmo plano galáctico. Espere
   confusão entre ela e as nebulosas no nível 1, pelo mesmo motivo.

Caminhos, se quiserem atacar isso depois: usar um levantamento em Hα (o
SuperCOSMOS Hα Survey foi testado e tem cobertura parcial demais para servir);
restringir a classe aos SNRs opticamente conhecidos (Véu, Caranguejo, Vela,
IC 443 — poucas dezenas); ou remover a classe e declarar o porquê.

### O campo de visão: o erro que quase custou o projeto

A primeira versão usava um **campo de visão fixo por classe**. Resultado: das
20 planetárias inspecionadas, **zero** mostravam a nebulosa. Eram campos
estelares, visualmente idênticos à classe `star_field`. Nenhuma métrica de
treino revelaria isso — só a inspeção visual revelou.

A causa é que objetos astronômicos variam em tamanho aparente por ordens de
grandeza:

| Classe | Coluna do catálogo | Mediana |
|---|---|---|
| `planetary` | `oDiam` (V/84/diam) | **9 segundos** de arco |
| `reflection` | `BRadMax` (raio) | ⌀ 5,6 minutos |
| `emission` | `Diam` | 12 minutos |
| `supernova_remnant` | `Dmaj` | **18 minutos** de arco |

Das planetárias aos remanescentes há um fator de **120×**. Nenhum valor fixo
serve aos dois. Hoje o campo é calculado por objeto:

```
fov = diâmetro × fov_multiplier,  limitado a [fov_min, fov_max]
```

E objetos abaixo de `min_diam_arcmin` são **descartados**: ampliar 9 segundos
de arco do DSS2 produz um borrão, não uma nebulosa. O corte levou `planetary`
de 1143 para 269 objetos — e, de quebra, equilibrou as quatro classes.

Depois da mudança, ~15 de 20 planetárias mostram o anel característico.

Os parâmetros ficam em `CATALOGS` (`data/catalogs.py`); `FOV_BY_LABEL` virou
só o valor de reserva para objetos sem diâmetro conhecido.

### Atenção: inspeção visual não é opcional

Depois de baixar, **folheie uma amostra de cada classe antes de treinar.**
Recortes vazios (objeto fraco demais para o DSS) e FOV errado são comuns e
envenenam o treino em silêncio. Um notebook simples em `notebooks/` que
mostra 50 imagens por classe numa grade vale mais que confiar no catálogo.

### Desbalanceamento

No conjunto de treino das nebulosas a razão entre a maior e a menor classe é
**3,4×** (`reflection` 692, `emission` 598, `planetary` 422,
`supernova_remnant` 204). Nas galáxias é **9,4×** (`spiral` 14.981,
`elliptical` 9.827, `irregular` 1.593) — o desbalanceamento pior está ali, não
nas nebulosas.

Três frentes tratam isso:

- **`class_weights: sqrt_balanced`** na loss. O peso cheio (inverso da
  frequência) mostrou-se agressivo demais: com 5,5× no `irregular`, o modelo
  passou a prevê-lo em excesso — recall 0,886 com precisão 0,579. A raiz do
  peso (2,3×) manteve o favorecimento sem o excesso: precisão subiu para
  0,695 e o F1 de 0,700 para 0,754.
- **Teto por classe** no nível 1 (`--cap-object`), senão 37 mil galáxias
  afogam mil nebulosas.
- **Macro-F1** como métrica de seleção do checkpoint, nunca acurácia.

---

## Classe `other` — ensinar o modelo a recusar

Sem esta classe, o modelo é obrigado a chamar tudo de galáxia ou nebulosa.
Uma foto da Lua vira "nebulosa planetária, 97%".

Cinco subpastas, todas pela mesma pipeline de recorte — **1.984 imagens**:

| Subpasta | O que é | n | Como obter |
|---|---|---:|---|
| `open_cluster/` | aglomerados abertos | 585 | NGC 2000.0 (VizieR `VII/118`) |
| `empty_field/` | céu vazio | 300 | sorteio em alta latitude galáctica |
| `star_field/` | campos estelares densos | 300 | sorteio no plano galáctico |
| `globular_cluster/` | aglomerados globulares | 199 | Harris (VizieR `VII/202`) |
| `non_astronomical/` | fotos comuns | 600 | `scripts/fetch_non_astronomical.py` (STL-10) |

As quatro primeiras mantêm o domínio (recortes de levantamento); a última ensina
a rejeitar o que não é astronomia. As duas coisas são necessárias.

Os aglomerados — abertos e globulares — são a parte mais útil da classe, e não os
campos sorteados. Um globular brilhante é **exatamente** o caso difícil: objeto
extenso, simétrico, num recorte escuro, visualmente próximo de uma galáxia
elíptica. Se o nível 1 aprende a separar isso, aprendeu algo; separar uma selfie
de uma galáxia ele consegue pelo brilho médio.

> **Recoletada em 26/09.** Uma coleta anterior, com outro esquema de nomes, tinha
> deixado **848 imagens órfãs** em disco, e a varredura de diretório as somava ao
> dataset: 735 estavam duplicadas byte a byte, com a mesma imagem podendo cair no
> treino sob um nome e no teste sob o outro. O `build_other_dataset.py` ganhou
> `--prune`, e hoje a árvore tem **zero** duplicatas — verificado por hash sobre
> as 66.299 imagens.

Sobre a última: usamos o **STL-10** (96×96) e não o CIFAR-10 (32×32) de
propósito. Ampliar 32×32 para 224 produz borrões, e o modelo aprenderia o
atalho "imagem borrada = other" em vez do conteúdo.

**Ressalva metodológica:** fotos naturais são claras e coloridas; recortes de
céu são escuros. Essa diferença é enorme e o modelo vai explorá-la. Não
conclua daí que ele "aprendeu a reconhecer astronomia" — o teste honesto é a
confusão entre `galaxy` e `nebula`, não contra esta classe.

---

## Estrutura final em disco

Tudo mora em `ASTRO_DATA_ROOT` (padrão `C:/astro-data`), **fora do OneDrive**:

```
C:/astro-data/
├── raw/
│   ├── galaxy_zoo/images_training_rev1/      61.578 .jpg, SEM subpasta por classe
│   ├── nebulae/{emission,reflection,planetary,supernova_remnant}/
│   └── other/{open_cluster,globular_cluster,empty_field,star_field,non_astronomical}/
├── external/        Galaxy10_DECals.h5 (benchmark, 2,6 GB)
├── ood/{mellinger,panstarrs,allwise}/        conjuntos fora-de-domínio
├── catalogs/        índices CSV gerados pelos scripts
├── splits/          assignment.csv + object_train.csv, galaxy_val.csv, ...
├── checkpoints/     *_best.pt, ood_threshold.json, ood_mahalanobis.pt
└── runs/            métricas e figuras por experimento
```

**As galáxias não são organizadas em pastas por classe, e isso é deliberado.**
Elas ficam como o Kaggle as entrega, e o rótulo vive em
`catalogs/galaxies_index.csv`. Motivo: das 61.578 imagens, só 37.716 recebem
rótulo confiável da votação — mover arquivos obrigaria a decidir o que fazer com
as outras 23.862, e mudar o critério de confiança exigiria remexer o disco em vez
de regerar um CSV.

**Os CSVs de `splits/` são pequenos — copie-os para `docs/` quando publicar um
resultado.** Reprodutibilidade não é só "usei seed 42": é poder apontar
exatamente quais imagens estavam em cada conjunto quando aquele número foi
medido.
