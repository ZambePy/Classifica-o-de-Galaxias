# Dataset de nebulosas — cartão de proveniência

Catálogo de **2737 nebulosas galácticas** com coordenadas, tipo e
enquadramento, montado a partir de oito catálogos astronômicos publicados.

As imagens não acompanham este arquivo. Reconstrua-as com:

```powershell
python scripts\build_nebula_dataset.py
```

O script consulta os mesmos catálogos no VizieR e baixa os recortes pelo
serviço `hips2fits` do CDS. O resultado é idêntico, porque tudo que varia
está fixado aqui: os identificadores, os filtros e a semente.

---

## Composição

| classe | objetos |
|---|---:|
| `emission` | 855 |
| `planetary` | 603 |
| `reflection` | 988 |
| `supernova_remnant` | 291 |

## Catálogos de origem

| identificador | objetos | referência |
|---|---:|---|
| `J/A+A/399/141/table1` | 888 | Magakian (2003), Merged catalogue of reflection nebulae, A&A 399, 141 |
| `VII/9` | 486 | Lynds (1965), Catalogue of Bright Nebulae, ApJS 12, 163 |
| `V/127A` | 346 | Kohoutek (2001), Catalogue of Galactic Planetary Nebulae |
| `VII/20` | 305 | Sharpless (1959), Catalogue of HII Regions, ApJS 4, 257 |
| `VII/272` | 291 | Green (2009), A Catalogue of Galactic Supernova Remnants |
| `V/84/main` | 257 | Acker+ (1992), Strasbourg-ESO Catalogue of Galactic Planetary Nebulae |
| `VII/216` | 161 | Rodgers, Campbell & Whiteoak (1960), Catalogue of Ha-emission regions |
| `VII/21` | 3 | van den Bergh (1966), Catalogue of Reflection Nebulae, AJ 71, 990 |

## Como cada rótulo foi decidido

- **`planetary`** — V/127A: todo o catálogo pertence à classe; objetos abaixo de 0.5 arcmin descartados
- **`reflection`** — J/A+A/399/141/table1: todo o catálogo pertence à classe; sem filtro de tamanho
- **`emission`** — VII/20: todo o catálogo pertence à classe; objetos abaixo de 1 arcmin descartados
- **`reflection`** — VII/21: todo o catálogo pertence à classe; objetos abaixo de 0.5 arcmin descartados
- **`planetary`** — V/84/main: todo o catálogo pertence à classe; objetos abaixo de 0.5 arcmin descartados
- **`supernova_remnant`** — VII/272: todo o catálogo pertence à classe; objetos abaixo de 2 arcmin descartados
- **`emission/reflection`** — VII/9: rótulo derivado da coluna `Color`; objetos abaixo de 2 arcmin descartados
- **`emission`** — VII/216: todo o catálogo pertence à classe; objetos abaixo de 2 arcmin descartados

## Enquadramento

O campo de visão é **proporcional ao diâmetro de cada objeto**, não fixo por
classe. Isso não é um detalhe: com campo fixo, **nenhuma** das 20 planetárias
que inspecionamos mostrava a nebulosa — as medianas de tamanho vão de 9
segundos de arco (planetárias) a 18 minutos (remanescentes), um fator de 120.

Objetos menores que o limite de resolução do levantamento são descartados, e
não ampliados: aumentar 9 arcsec do DSS2 produz um borrão.

| coluna | significado |
|---|---|
| `ra`, `dec` | coordenadas ICRS, em graus |
| `label` | classe |
| `fov_deg` | campo de visão usado no recorte, em graus |
| `diam_arcmin` | diâmetro do catálogo, quando disponível |
| `source` | identificador VizieR de origem |
| `name` | designação no catálogo de origem |

## Deduplicação

Catálogos se sobrepõem — o Lynds Bright Nebulae inclui dezenas de nebulosas
de Sharpless sob outra nomenclatura. Objetos a menos de **2 arcmin** um do
outro são tratados como o mesmo e mantém-se a primeira ocorrência, na ordem
declarada em `data/catalogs.py`. Foram removidos 259 objetos.

**Catálogos majoritariamente absorvidos.** Um número baixo aqui não é
erro de consulta — é a deduplicação funcionando:

- `VII/21` entrou com 158 objetos e ficou com **3**: os demais já estavam num catálogo anterior da lista.

---

## Vieses conhecidos

**Este dataset carrega um viés geográfico forte, e ele foi medido.**

Nebulosas galácticas ficam no plano da Via Láctea. A latitude galáctica
correlaciona com a classe:

| classe | \|b\| mediana | em \|b\| < 10° |
|---|---:|---:|
| `emission` | 1.9° | 86% |
| `planetary` | 4.0° | 81% |
| `reflection` | 3.1° | 72% |
| `supernova_remnant` | 0.5° | 99% |

Um classificador treinado nestas imagens **pode aprender a geografia em vez
do objeto**. Medimos isso por ablação: desfocando o objeto e mantendo só o
fundo, um ResNet18 mantém **94% da acurácia**. Para comparação, o mesmo teste
num classificador de galáxias (Galaxy Zoo, alta latitude) derruba o
desempenho para abaixo do chute.

Quem usar este dataset deve reportar essa ablação junto com a acurácia.

**Outras limitações:**

- `supernova_remnant` tem sinal óptico fraco — a maioria foi descoberta em
  rádio, e o DSS2 mal os mostra.
- O rótulo `emission` vs `reflection` do catálogo Lynds vem da **cor
  dominante**, que é também a feature que um modelo usaria. Não é circular
  (a cor É o critério físico), mas merece menção.
- O DSS2 são placas fotográficas digitalizadas dos anos 1990. Para objetos
  fora do plano galáctico há levantamentos melhores.

---

## Licença e citação

Este **catálogo** (o CSV e a curadoria) é MIT, como o resto do repositório.

As **imagens** reconstruídas a partir dele pertencem aos levantamentos de
origem, distribuídos pelo CDS/Observatoire de Strasbourg. Verifique os termos
de cada um antes de redistribuir pixels.

Ao usar, cite os catálogos originais listados acima e o serviço hips2fits:

> Boch T., Fernique P., et al., *hips2fits*, Centre de Données
> astronomiques de Strasbourg.
