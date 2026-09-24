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
   trade-off deliberado. Registre quantas imagens sobraram com cada
   `--min-vote` — isso vira um gráfico no trabalho.
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

| Classe | Catálogo | Referência |
|---|---|---|
| `emission` | Sharpless (regiões HII) | Sharpless 1959, ApJS 4, 257 |
| `reflection` | van den Bergh | van den Bergh 1966, AJ 71, 990 |
| `planetary` | Strasbourg-ESO de PNe | Acker et al. 1992 |
| `supernova_remnant` | Catálogo de SNRs galácticos | Green 2009 |

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

### O campo de visão importa mais do que parece

Uma nebulosa planetária tem segundos de arco e **some** num recorte de 1 grau.
A Nebulosa de Órion tem graus e **não cabe** num recorte de 0,05. Por isso o
campo de visão é por tipo, em `FOV_BY_LABEL` (`data/catalogs.py`):

| Classe | FOV | Por quê |
|---|---|---|
| `planetary` | 0,05° | compactas |
| `reflection` | 0,5° | médias |
| `supernova_remnant` | 0,7° | extensas |
| `emission` | 0,8° | regiões HII são grandes |

**Estes valores são um chute inicial informado, não verdade.** Ajuste depois
de olhar as imagens.

### Atenção: inspeção visual não é opcional

Depois de baixar, **folheie uma amostra de cada classe antes de treinar.**
Recortes vazios (objeto fraco demais para o DSS) e FOV errado são comuns e
envenenam o treino em silêncio. Um notebook simples em `notebooks/` que
mostra 50 imagens por classe numa grade vale mais que confiar no catálogo.

### Desbalanceamento esperado

Há milhares de nebulosas planetárias catalogadas contra ~150 de reflexão. O
projeto trata isso em três frentes: `class_weights: balanced` na loss, teto por
classe em `make_splits.py --cap-per-class`, e **macro-F1** como métrica de
seleção do modelo (nunca acurácia).

---

## Classe `other` — ensinar o modelo a recusar

Sem esta classe, o modelo é obrigado a chamar tudo de galáxia ou nebulosa.
Uma foto da Lua vira "nebulosa planetária, 97%".

Fontes sugeridas, todas pela mesma pipeline de recorte:

| Subpasta | O que é | Como obter |
|---|---|---|
| `globular_cluster/` | aglomerados globulares | catálogo de Harris (VizieR `VII/202`) |
| `empty_field/` | céu vazio | coordenadas aleatórias longe de objetos catalogados |
| `star_field/` | campos estelares densos | coordenadas no plano galáctico |
| `non_astronomical/` | fotos comuns | qualquer conjunto de imagens do dia a dia |

As três primeiras mantêm o domínio (recortes de levantamento); a última ensina
a rejeitar o que não é astronomia. As duas coisas são necessárias.

---

## Estrutura final em disco

Tudo mora em `ASTRO_DATA_ROOT` (padrão `C:/astro-data`), **fora do OneDrive**:

```
C:/astro-data/
├── raw/
│   ├── galaxies/{spiral,elliptical,irregular}/
│   ├── nebulae/{emission,reflection,planetary,supernova_remnant}/
│   └── other/{globular_cluster,empty_field,star_field,non_astronomical}/
├── catalogs/        índices CSV gerados pelos scripts
├── splits/          object_train.csv, galaxy_val.csv, ...
├── checkpoints/     *_best.pt, ood_threshold.json
└── runs/            métricas e figuras por experimento
```

**Os CSVs de `splits/` são pequenos — copie-os para `docs/` quando publicar um
resultado.** Reprodutibilidade não é só "usei seed 42": é poder apontar
exatamente quais imagens estavam em cada conjunto quando aquele número foi
medido.
