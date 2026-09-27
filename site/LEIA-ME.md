# Cassyn: site

Site do projeto (antes "Projeto Galáxia"). Mostra galáxias, nebulosas e planetas em 3D e deixa a pessoa enviar uma foto para a IA classificar. A IA é a API FastAPI deste repositório (`src/astro_classifier/api`), e o contrato está em `docs/api.md`.

| Pasta | O que tem |
|---|---|
| `dashboard/` | O site: React + Vite (`client/`), servidor Node + Express (`server/`) e taxonomia compartilhada (`shared/`) |
| `ferramentas/integracao/` | Scripts do Windows para conferir o ambiente e ligar a API do classificador |

## Como rodar

O passo a passo completo está no `README.md` da raiz, na seção "Como executar". Resumo no Windows, na pasta `site\dashboard`:

```powershell
npm install
npm run verificar                        # confere o ambiente
npm run dev:all                          # site em http://localhost:5173 + API da IA em modo simulado
npm run dev:all:real                     # o mesmo, com os modelos de verdade
```

Ou em duas janelas: `npm run dev:classificador` (ou `npm run dev:classificador:real`) e `npm run dev`. O site conversa com a API por `server/lib/classificador.js`, que traduz a resposta para os nomes e abas do site.

Os pesos dos modelos ficam **fora** do repositório, em `..\modelos\checkpoints` (ao lado da pasta do repositório), como o `.gitignore` da raiz pede.
