# Cassyn — Dashboard (esboço)

> O projeto se chamava **Projeto Galáxia** até a sessão 12. O nome **Cassyn** homenageia Giovanni Domenico Cassini, que descobriu a divisão nos anéis de Saturno.

Painel para exibir a classificação automática de **galáxias** e **nebulosas** feita por uma rede neural convolucional (CNN).

```
navegador (React) ──► Node.js / Express ──────────► FastAPI do classificador (raiz do repositório)
                         │  serve o site              │  src/astro_classifier/api, contrato 1.0
                         │  /api/classify: limite,    │  valida a imagem, roda a cascata de 3 CNNs
                         │  traduz a resposta         │  (objeto → tipo) e o aviso de domínio
                         └── taxonomia compartilhada: shared/taxonomy.json (rótulo da IA → aba)
```

| Pasta | O que tem |
|---|---|
| `client/` | Front-end React 19 + Vite 8 + React Router 7 |
| `server/` | API Node.js (Express 5 + Helmet). `server/lib/classificador.js` conversa com a API do classificador |
| `shared/taxonomy.json` | Categorias, sub-abas e o mapeamento *rótulo do modelo → sub-aba* |

## Abertura do site (logo animado)
- Ao abrir o site, o logo animado 10 ("Salto do espaço", `client/src/assets/intro/logo-abertura.svg`, gerado por `branding/animacoes/gerar-animacoes.mjs`) toca no centro da tela e depois voa até o logo do menu, no canto superior esquerdo (`client/src/components/IntroSplash.jsx`).
- Uma vez por visita (sessionStorage `pg-abertura-vista`); clique, toque, Esc ou "Pular" encerram na hora; quem pede menos movimento no sistema não vê.
- O logo do menu agora fica parado (antes girava devagar).

## Aba Planetas (os 8 planetas em 3D)
- Rota `/planetas` (`client/src/pages/PlanetsPage.jsx`): cartões de Mercúrio a Netuno, com bolinha girando (CSS) e grupo (rochoso, gigante de gás, gigante de gelo). Clicar abre a mesma janela 3D dos outros exemplos.
- Modelo 3D (`client/src/three/planets.js`): esfera com textura, luz do Sol vindo de um lado, halo da atmosfera só no lado iluminado, nuvens na Terra, anéis em Saturno (de longe parecem uma placa lisa; ao aproximar viram névoa e grãos de partículas) e em Urano, inclinação real do eixo e giro proporcional à duração do dia (acelerado). Dados em `objects3d.js` (números: NASA Planetary Fact Sheet).
- Relevo (sessão 11): mapas de relevo (normal maps) em Mercúrio (crateras com fundo, borda, pico central e raios claros; planícies lisas), Marte (Monte Olimpo e vulcões de Tharsis, Valles Marineris com o labirinto de Noctis, bacias Hellas e Argyre, planícies do norte, crateras) e Terra (cordilheiras com neve nos picos + oceano que reflete o Sol, por um mapa de rugosidade). Gigantes com turbulência fina nas faixas, ovais brancas em Júpiter, região polar em Saturno, capa polar em Urano e manchas escuras em Netuno.
- Texturas **ilustrativas** (2048 × 1024; 4096 × 2048 no HD) geradas por código: `ferramentas/texturas-planetas/gerar_texturas.py` (`python3 gerar_texturas.py saida 2048` e `python3 gerar_texturas.py saida-4k 4096`; um 3º argumento opcional gera só alguns planetas, ex. `marte,terra`). Os continentes da Terra vêm do Natural Earth (domínio público) via `terra-geojson.mjs`. Para usar fotos reais no futuro, basta trocar os arquivos em `client/src/assets/planets/` (mesmos nomes) e dar o crédito na página Créditos.

## Aba futura: Estrelas, e a câmera em Planetas
- Rota `/estrelas` (`client/src/pages/ComingSoon.jsx`), só com o aviso "Em breve".
- Ideia para Planetas: apontar a câmera do celular para o céu e identificar o planeta. Pontos para quando for construir:
  - a câmera só funciona em site com HTTPS e depois que a pessoa autoriza no navegador (`navigator.mediaDevices.getUserMedia`);
  - de preferência, analisar a imagem no próprio celular e não enviar o vídeo ao servidor (privacidade/LGPD); explicar isso antes de pedir a permissão (já está na lista da página Conformidade);
  - a pré-visualização publicada no claude.ai não permite câmera, então esse teste terá de ser feito no site rodando de verdade.

## Modelos 3D dos exemplos
- Nas sub-abas de Galáxias e Nebulosas, cada exemplo famoso é clicável e abre um modelo 3D (arrastar gira, rolar aproxima).
- 31 modelos **ilustrativos** (incluindo os Pilares da Criação), gerados por código com partículas (`client/src/three/scenes.js`), sem arquivos 3D externos. O site avisa que o formato é aproximado e não é foto real.
- Dados de cada objeto (nome, distância, curiosidade em PT/EN, tipo de modelo e ângulo inicial): `client/src/three/objects3d.js`. O elo exemplo → modelo fica em `exampleIds` no `shared/taxonomy.json`.
- Para adicionar um exemplo: inclua o nome em `examples`/`en.examples`, o id em `exampleIds` e os dados em `OBJECTS`.
- A biblioteca three.js (MIT) só é carregada quando alguém abre um modelo, então o site continua leve.
- **Zoom em qualquer ponto** (todos os modelos: galáxias, nebulosas e planetas):
  - rolar o mouse (ou fazer pinça no celular) sobre uma parte aproxima **daquela parte**; antes de cada zoom o site acha o ponto do modelo embaixo do mouse/dedos e usa a profundidade dele, então a aproximação chega perto sem atravessar o modelo. Sobre o espaço vazio, o zoom vai para o centro, como antes;
  - clique duplo / dois toques: voo suave até o ponto; depois do zoom, clique duplo de novo volta suave à vista inicial; botão direito ou dois dedos: mover para os lados; "Voltar ao início" desfaz tudo;
  - giro livre de 360° em todos os modelos (inclusive as nebulosas que antes só giravam um pouco);
  - ao dar zoom, o giro automático para (senão o detalhe "foge"); o botão "Girar sozinho" liga de novo;
  - de muito perto, cada partícula tem tamanho máximo na tela (40 px × densidade da tela) e as que encostam na câmera somem — evita "bolhas" gigantes e mantém o desenho leve;
  - planetas no HD usam texturas 4096 × 2048 (Leve: 2048 × 1024), para a superfície ficar nítida de perto; a câmera não entra no planeta;
  - correção de um detalhe do three.js: na pinça, o centro era calculado com a rolagem da página por trás da janela (o zoom ia para o lugar errado).
- **Qualidade HD / Leve** (botão na janela 3D, a escolha fica salva no navegador):
  - HD (padrão em computador): 3x mais partículas (~140 mil por modelo em média), textura de nuvem com ruído fractal, halo de brilho (bloom), mapeamento de tons ACES e resolução nativa da tela (até 2,5x).
  - Leve (padrão em celular): ~46 mil partículas, sem bloom, resolução até 1,25x.
  - Botão **Tela cheia**. Gerar um modelo em HD leva em média ~0,2 s (o mais pesado ~0,6 s).

## Chat com a IA
- Aba **Converse com a IA** (`client/src/pages/Chat.jsx`) e rota `POST /api/chat` no servidor Node (`server/lib/chat.js`).
- **Hoje:** modo de demonstração, com respostas prontas sobre os tipos de galáxias e nebulosas, a IA do site, envio de fotos e privacidade (`shared/chatDemo.js`). Nenhuma empresa externa é chamada.
- **Para ligar uma API de IA depois:** preencha no `.env` `CHAT_PROVIDER=openai-compatible`, `CHAT_API_URL`, `CHAT_API_KEY` e `CHAT_MODEL` (qualquer API no formato "chat completions"). Para outro formato, crie uma função como `askOpenAICompatible` em `server/lib/chat.js`. A chave fica só no servidor.
- Proteções: limite de 20 mensagens por minuto por endereço, mensagens de até 1.000 caracteres, só as últimas 12 mensagens são enviadas, nada é guardado no servidor. As instruções de comportamento da IA (tema, linguagem simples, público jovem, não pedir dados pessoais) estão em `SYSTEM_PROMPT`.
- Ao escolher a empresa de IA: atualize a página de Privacidade com o nome e o país dela (exigência da LGPD).

## Tema e idioma
- Seletor no topo de cada página: **PT / EN** e tema **sistema / claro / escuro** ("sistema" segue a configuração do computador).
- A escolha fica salva só no navegador do visitante (`localStorage`).
- Textos da interface: `client/src/i18n/messages.js`. Textos da taxonomia em inglês: campo `en` de cada item em `shared/taxonomy.json`.
- Termos, Privacidade e Conformidade têm versão completa em inglês, com aviso de que a versão em português prevalece.
- Cores dos dois temas: tokens no topo de `client/src/styles/global.css` (`:root` = escuro, `[data-theme='light']` = claro).

## Visual e textos
- Fonte única: **Figtree** (licença livre SIL OFL), servida pelo próprio site.
- Textos do site em linguagem simples, sem termos técnicos. Os detalhes técnicos (rótulos do modelo, dataset, API) ficam só neste README e no `shared/taxonomy.json`.
- "Conformidade com a lei" fica no rodapé, ao lado de Termos de uso, Privacidade e Créditos.

## Efeitos visuais
- Tema claro: estrelinhas coloridas que brilham, linhas de constelação e estrelas cadentes em azul/roxo.
- Céu animado em canvas (`components/Starfield.jsx`): estrelas que cintilam em 3 camadas com paralaxe do mouse, nebulosas em movimento e estrelas cadentes (só no tema escuro).
- Galáxia espiral em partículas com rotação diferencial (`components/GalaxySpin.jsx`) na Visão geral e na página 404.
- Painéis de vidro, brilho que segue o mouse, contadores animados, títulos com degradê, sub-aba ativa com borda animada.
- Tudo desliga sozinho se o sistema estiver com "reduzir movimento" ativado; a animação pausa quando a aba fica em segundo plano.

## Requisitos
- Node.js 22.9 ou mais novo (lê o arquivo `.env` automaticamente)
- Python 3.10 ou mais novo e o `.venv` do classificador, criado na raiz do repositório com `scripts\setup_env.ps1`

## Como rodar (desenvolvimento)

A IA é a API FastAPI do classificador, na raiz deste repositório (`src/astro_classifier/api`, contrato em `docs/api.md`).
O antigo `ml-service/` provisório foi aposentado. No Windows, na pasta `site\dashboard`:

```powershell
npm install
npm run verificar          # confere Node, Python, modelos, portas e git
npm run dev:all            # site (:3001 e :5173) + API da IA em modo simulado (:8000)
npm run dev:all:real       # o mesmo, com os modelos de verdade
```

Ou em janelas separadas: `npm run dev:classificador` (ou `npm run dev:classificador:real`) e `npm run dev`.
Abra http://localhost:5173. Com a API desligada, a tela "Classificar" mostra um aviso e o botão fica desativado.

- Os modelos ficam **fora** do repositório, em `..\modelos\checkpoints\` (`object_best.pt`, `galaxy_best.pt`, `nebula_best.pt`, `ood_threshold.json`).
- No modo simulado a API responde sem modelo, e a tela mostra o selo "Dados simulados".
- Copie `.env.example` para `.env` para mudar `ML_SERVICE_URL`, `CLASSIFY_PER_MINUTE` (padrão 10) e `CLASSIFIER_WARMUP`.

Produção: coloque `NODE_ENV=production` no `.env`, rode `npm run build` e depois `npm start` (o Express serve o site compilado com HTTPS forçado).

## Como o site usa a IA
- `POST /api/classify` repassa a foto (stream, sem gravar em disco) para `POST /predict` da API e traduz a resposta em `server/lib/classificador.js`:
  - `kind` (`galaxia`, `nebulosa` ou `outro`), `category` e `type` (aba do site, via `shared/taxonomy.json`), `confidence`, `steps` (as probabilidades de cada nível), `ambiguous` (as 2 maiores probabilidades do nível que decidiu a aba diferem menos de 0,15), `outOfDomain`, `mock` e `modelVersion`;
  - o site mostra sempre os próprios nomes, nunca o `label_pt` da API (que vem sem acento).
- Contrato: se o número principal de `contract_version` não for 1, o site responde 502 ("A IA foi atualizada e o site ainda não").
- Erros: 400, 415 e 422 da API → 422; 413 → 413; 500 ou API fora → 503; timeout de 60 s; mais de `CLASSIFY_PER_MINUTE` envios por minuto e por endereço → 429.
- `/api/health`: `model.connected` (API no ar com contrato 1.x), `model.demo` (modo mock) e `model.version` (versões da última classificação).
- `CLASSIFIER_WARMUP=1`: com a API em modo real, o Node manda uma imagem pequena ao ligar para os modelos já carregarem (senão o 1º envio leva alguns segundos). Se a API estiver fora, tenta de novo a cada 30 s, no máximo 5 vezes.
- A foto não é guardada e nada disso entra nas estatísticas do painel.
- Para preencher as sub-abas em lote, envie cada resultado para `POST /api/classifications` com o cabeçalho `x-api-key: $MODEL_API_KEY`. Crédito e licença da imagem são obrigatórios.

Testes do servidor: `npm run test:server` (taxonomia, adaptador com os exemplos de `server/test/fixtures/contrato-1.0/` e uma API falsa).

## API

| Método | Rota | Descrição |
|---|---|---|
| GET | `/api/health` | Status do site e do serviço de IA |
| GET | `/api/taxonomy` | Categorias e sub-abas |
| GET | `/api/stats` | Contagens por categoria/sub-aba |
| GET | `/api/classifications?category=&type=&subfilter=` | Resultados de uma sub-aba |
| POST | `/api/classifications` | Publica um resultado (requer `x-api-key`) |
| POST | `/api/chat` | Chat: recebe `{ lang, messages }` e devolve `{ reply, demo }` |
| POST | `/api/classify` | Envio de imagem pelo usuário → repassado à API do classificador (`/predict`) e traduzido |

API do classificador (porta 8000, só na rede interna): `GET /health`, `POST /predict`, documentação automática em `/docs`.

## Antes de publicar
- Preencher os campos entre colchetes em `client/src/data/site.js` (responsável, e-mail de privacidade, comarca, hospedagem).
- Hospedar com HTTPS, de preferência com dados no Brasil ou na UE.
- Trocar o armazenamento em arquivo (`server/lib/store.js`) por um banco de dados.
- Ver a lista completa na página **/conformidade** do próprio painel.
