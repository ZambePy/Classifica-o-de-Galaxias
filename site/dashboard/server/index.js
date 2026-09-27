// Servidor Node.js (Express) do dashboard Cassyn (antes: Projeto Galáxia).
// - Expõe a API REST que o modelo (CNN) usa para publicar classificações.
// - Repassa as imagens enviadas à API do classificador do colega (FastAPI na raiz do repositório,
//   src/astro_classifier/api) e traduz a resposta com lib/classificador.js.
// - Em produção, serve o front-end React compilado (client/dist).
import express from 'express';
import helmet from 'helmet';
import path from 'node:path';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { taxonomy, resolveLabel, findCategory, findType, isKnownClientRoute } from './lib/taxonomy.js';
import * as store from './lib/store.js';
import { chatReply, chatIsDemo } from './lib/chat.js';
import { createClassifier } from './lib/classificador.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const CLIENT_DIST = path.resolve(__dirname, '../client/dist');
const PORT = Number(process.env.PORT) || 3001;
const MODEL_API_KEY = process.env.MODEL_API_KEY || '';
const IS_PROD = process.env.NODE_ENV === 'production';

// API do classificador (só na rede interna). CLASSIFY_PER_MINUTE: envios por minuto e por endereço.
const classifier = createClassifier({
  baseUrl: process.env.ML_SERVICE_URL,
  perMinute: Number(process.env.CLASSIFY_PER_MINUTE) || 10,
});

const app = express();
app.disable('x-powered-by');

// Cabeçalhos de segurança (LGPD art. 46 — medidas técnicas de segurança).
app.use(
  helmet({
    contentSecurityPolicy: {
      directives: {
        defaultSrc: ["'self'"],
        imgSrc: ["'self'", 'data:', 'https:'],
        // Fontes auto-hospedadas (@fontsource): nenhuma requisição a terceiros.
        styleSrc: ["'self'", "'unsafe-inline'"],
        fontSrc: ["'self'", 'data:'],
        scriptSrc: ["'self'"],
        connectSrc: ["'self'"],
        // Em produção (HTTPS) força https; em localhost (http) fica desligado.
        upgradeInsecureRequests: IS_PROD ? [] : null,
      },
    },
    hsts: IS_PROD,
  }),
);
app.use(express.json({ limit: '1mb' }));

// ---------------------------------------------------------------- API
const api = express.Router();

// Estado da IA vindo do GET /health do colega. connected = API no ar com contrato 1.x;
// demo = modo mock; version = versões vistas na última classificação (ou null).
api.get('/health', async (_req, res) => {
  const ml = await classifier.health();
  res.json({
    status: 'ok',
    mlService: ml.online ? 'online' : 'offline',
    model: {
      connected: ml.connected,
      version: ml.version,
      demo: ml.demo,
      mode: ml.mode,
      device: ml.device,
      modelsLoaded: ml.modelsLoaded,
      message: ml.message,
    },
    chat: { demo: chatIsDemo() },
    time: new Date().toISOString(),
  });
});

// ------------------------------------------------------------ chat
// Limite simples por endereço (evita abuso e custo quando houver uma API paga).
const CHAT_LIMIT = 20; // mensagens por minuto
const chatHits = new Map();
function chatRateLimited(ip) {
  const now = Date.now();
  const recent = (chatHits.get(ip) ?? []).filter((t) => now - t < 60_000);
  recent.push(now);
  chatHits.set(ip, recent);
  if (chatHits.size > 5000) chatHits.clear();
  return recent.length > CHAT_LIMIT;
}

// As mensagens NÃO são guardadas pelo servidor: chegam, são respondidas e descartadas.
api.post('/chat', async (req, res) => {
  if (chatRateLimited(req.ip)) {
    return res.status(429).json({ error: 'Muitas mensagens seguidas. Espere um minuto e tente de novo.' });
  }
  const lang = req.body?.lang === 'en' ? 'en' : 'pt';
  const raw = Array.isArray(req.body?.messages) ? req.body.messages : [];
  const messages = raw
    .filter((m) => m && (m.role === 'user' || m.role === 'assistant') && typeof m.content === 'string')
    .map((m) => ({ role: m.role, content: m.content.trim().slice(0, 1000) }))
    .filter((m) => m.content)
    .slice(-12);
  if (!messages.length || messages[messages.length - 1].role !== 'user') {
    return res.status(422).json({ error: 'Escreva uma mensagem.' });
  }
  try {
    res.json(await chatReply({ messages, lang }));
  } catch (err) {
    console.error('[chat]', err.message);
    res.status(502).json({ error: 'O assistente está indisponível agora. Tente mais tarde.' });
  }
});

api.get('/taxonomy', (_req, res) => res.json(taxonomy));

api.get('/stats', (_req, res) => res.json(store.stats()));

api.get('/classifications', (req, res) => {
  const { category, type, subfilter } = req.query;
  if (category && !findCategory(category)) return res.status(400).json({ error: 'Categoria inválida.' });
  if (category && type && !findType(category, type)) return res.status(400).json({ error: 'Tipo inválido.' });
  const limit = Math.min(Number(req.query.limit) || 100, 500);
  res.json({ items: store.list({ category, type, subfilter, limit }) });
});

// Endpoint usado pelo pipeline do modelo para publicar um resultado.
// Autenticação por chave (defina MODEL_API_KEY no ambiente).
api.post('/classifications', (req, res) => {
  if (!MODEL_API_KEY) {
    return res.status(503).json({ error: 'Publicação desativada: defina a variável MODEL_API_KEY no servidor.' });
  }
  if (req.get('x-api-key') !== MODEL_API_KEY) {
    return res.status(401).json({ error: 'Chave de API inválida.' });
  }

  const { label, confidence, probabilities, imageUrl, objectName, credit, license, source, modelVersion } = req.body ?? {};
  const resolved = resolveLabel(label);
  const errors = [];
  if (!resolved) errors.push(`Rótulo "${label}" não está mapeado em shared/taxonomy.json.`);
  if (typeof confidence !== 'number' || confidence < 0 || confidence > 1) errors.push('confidence deve ser número entre 0 e 1.');
  if (typeof imageUrl !== 'string' || !/^(https:\/\/|\/)/.test(imageUrl)) errors.push('imageUrl deve ser https:// ou caminho local.');
  // Crédito e licença são obrigatórios (Lei 9.610/1998 art. 79 §1º; licenças CC BY 4.0 das agências).
  if (typeof credit !== 'string' || !credit.trim()) errors.push('credit (linha de crédito da imagem) é obrigatório.');
  if (typeof license !== 'string' || !license.trim()) errors.push('license é obrigatório (ex.: "CC BY 4.0").');
  if (errors.length) return res.status(422).json({ errors });

  const record = store.add({
    ...resolved,
    label,
    confidence,
    probabilities: probabilities && typeof probabilities === 'object' ? probabilities : null,
    imageUrl,
    objectName: typeof objectName === 'string' ? objectName.slice(0, 120) : null,
    credit: credit.slice(0, 300),
    license: license.slice(0, 60),
    source: typeof source === 'string' ? source.slice(0, 120) : null,
    modelVersion: typeof modelVersion === 'string' ? modelVersion.slice(0, 40) : null,
  });
  res.status(201).json(record);
});

// Upload de imagem pelo usuário: repassado (stream) ao POST /predict do colega e traduzido
// para os nomes e abas do site. Nada é gravado em disco e nada entra nas estatísticas:
// a API lê a foto na memória e a descarta logo depois da análise.
// Erros: 429 (envios demais), 422 (imagem inválida), 413 (grande demais), 502 (contrato novo), 503 (IA fora).
api.post('/classify', classifier.route);

api.use((_req, res) => res.status(404).json({ error: 'Rota da API não encontrada.' }));

app.use('/api', api);

// ---------------------------------------------------------- Front-end
if (existsSync(CLIENT_DIST)) {
  app.use(express.static(CLIENT_DIST, { index: false }));
  // Rotas conhecidas -> 200; desconhecidas -> página 404 do React com status HTTP 404 real.
  app.use((req, res) => {
    const status = isKnownClientRoute(req.path) ? 200 : 404;
    res.status(status).sendFile(path.join(CLIENT_DIST, 'index.html'));
  });
} else {
  app.use((_req, res) =>
    res.status(404).send('Front-end não compilado. Rode "npm run build" ou use "npm run dev" (Vite em http://localhost:5173).'),
  );
}

app.listen(PORT, () => {
  console.log(`Cassyn — API em http://localhost:${PORT}/api`);
  // CLASSIFIER_WARMUP=1: com a API em modo real, os modelos já carregam ao ligar (o 1º envio fica rápido).
  if (process.env.CLASSIFIER_WARMUP === '1') classifier.warmup();
});
