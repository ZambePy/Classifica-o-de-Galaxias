# Contrato da API — guia para o dashboard

Este documento é para quem constrói o **frontend**. Ele descreve tudo que a
API devolve, e é suficiente para construir o dashboard inteiro **sem que
nenhum modelo tenha sido treinado ainda**.

Versão do contrato: **1.0**

---

## Subindo a API em modo mock

```powershell
powershell -ExecutionPolicy Bypass -File scripts\serve_api.ps1
```

A API sobe em `http://127.0.0.1:8000` respondendo com dados sintéticos, mas
**no formato final e definitivo**. Documentação interativa (Swagger) em
`http://127.0.0.1:8000/docs` — dá para testar upload de imagem pelo navegador.

O mock é **determinístico**: a mesma imagem sempre devolve o mesmo resultado.
Isso permite escrever testes de frontend que não quebram sozinhos.

---

## `GET /health`

Use para saber se o backend está de pé e em que modo.

```json
{
  "status": "ok",
  "contract_version": "1.0",
  "mode": "mock",
  "device": "cuda",
  "models_loaded": ["object", "galaxy"]
}
```

`models_loaded` diz quais níveis da cascata já têm modelo treinado. Enquanto
`"nebula"` não aparecer ali, nebulosas voltam sem subtipo — o dashboard deve
lidar com isso (ver abaixo).

---

## `POST /predict`

Envio: `multipart/form-data`, campo obrigatório `file`.
Limite: 20 MB. Formatos: qualquer um que o Pillow abra (JPG, PNG, WEBP…).

```js
const form = new FormData();
form.append("file", arquivo);

const resposta = await fetch("http://127.0.0.1:8000/predict", {
  method: "POST",
  body: form,
});
const dados = await resposta.json();
```

### Resposta

```json
{
  "contract_version": "1.0",
  "mock": true,

  "object": "galaxy",
  "confidence": 0.943,
  "subtype": "spiral",
  "subtype_confidence": 0.871,
  "summary_pt": "Galáxia espiral",

  "levels": [
    {
      "level": "object",
      "predicted": "galaxy",
      "predicted_pt": "Galáxia",
      "confidence": 0.943,
      "scores": [
        { "label": "galaxy", "label_pt": "Galáxia",      "probability": 0.943 },
        { "label": "nebula", "label_pt": "Nebulosa",     "probability": 0.041 },
        { "label": "other",  "label_pt": "Outro objeto", "probability": 0.016 }
      ],
      "model_version": "object-e12"
    },
    {
      "level": "galaxy",
      "predicted": "spiral",
      "predicted_pt": "Espiral",
      "confidence": 0.871,
      "scores": [
        { "label": "spiral",     "label_pt": "Espiral",   "probability": 0.871 },
        { "label": "elliptical", "label_pt": "Elíptica",  "probability": 0.102 },
        { "label": "irregular",  "label_pt": "Irregular", "probability": 0.027 }
      ],
      "model_version": "galaxy-e18"
    }
  ],

  "domain": {
    "out_of_domain": false,
    "score": 0.943,
    "threshold": 0.612,
    "message_pt": "Imagem dentro do domínio de treino."
  },

  "inference_ms": 23.4
}
```

### Como usar cada parte

| Campo | Para que serve no dashboard |
|---|---|
| `summary_pt` | O título grande. Já vem pronto e traduzido — não monte a frase no frontend. |
| `confidence` | A barra/percentual principal. |
| `levels` | O detalhamento. Cada nível vira um gráfico de barras com todas as classes. |
| `label_pt` | Sempre exiba este, nunca `label` (que é o nome interno em inglês). |
| `domain.out_of_domain` | **Quando `true`, exiba o aviso em destaque**, junto do resultado. |
| `mock` | Quando `true`, mostre um selo "dados simulados" — evita demonstrar um mock achando que é real. |
| `model_version` | Rodapé/debug. Permite saber qual checkpoint gerou aquele número. |

---

## Três casos que o dashboard precisa tratar

**1. Objeto sem subtipo.** Quando `object` é `"other"`, `subtype` vem `null` e
`levels` tem um item só. A cascata para ali de propósito — não existe subtipo
de "outro".

**2. Subtipo ainda não treinado.** Se o modelo de nível 2 daquele ramo ainda
não existe, a resposta volta com `subtype: null` mesmo para galáxia ou
nebulosa. É o estado normal do projeto entre uma fase e outra. Trate
`subtype === null` como "ainda não disponível", não como erro.

**3. Imagem fora do domínio.** `domain.out_of_domain === true` significa que a
imagem não se parece com o que o modelo viu no treino (recortes de
levantamentos DSS2/SDSS). O resultado ainda vem preenchido, mas **não deve ser
apresentado como uma resposta confiável**. Sugestão de tratamento:

```jsx
{dados.domain.out_of_domain && (
  <Aviso tipo="atencao">
    {dados.domain.message_pt}
  </Aviso>
)}
```

Este caso vai acontecer bastante — é o esperado quando alguém sobe uma foto
processada do Hubble. Não é bug; é o sistema sendo honesto.

---

## Erros

| Código | Quando | Corpo |
|---|---|---|
| `400` | arquivo vazio | `{"detail": "Arquivo vazio."}` |
| `413` | maior que 20 MB | `{"detail": "Imagem maior que o limite de 20 MB."}` |
| `415` | não é imagem | `{"detail": "Arquivo nao reconhecido como imagem."}` |
| `422` | campo `file` ausente | erro de validação do FastAPI |

---

## Regra de convivência entre as duas pessoas do projeto

Este contrato é o acordo entre os dois lados. Enquanto ele não mudar, cada um
trabalha sozinho sem quebrar o outro.

**Se precisar mudar um campo:** avise a outra pessoa, suba
`CONTRACT_VERSION` em `src/astro_classifier/api/schemas.py`, e atualize este
documento. Os testes em `tests/test_api_contract.py` vão falhar até que a
mudança seja consciente — é exatamente o que eles existem para fazer.

## CORS

A API libera todas as origens por padrão, para que o dashboard rodando em
outra porta (Vite em `:5173`, por exemplo) funcione sem configuração. Se algum
dia isso for para a internet, restrinja `allow_origins` em
`src/astro_classifier/api/main.py`.
