// Adaptador entre o site e a API do classificador do colega (FastAPI, contrato 1.0).
// - Traduz a resposta de POST /predict para o formato da tela (nomes e abas do site, via shared/taxonomy.json).
// - Confere a versão do contrato: se o número principal não for 1, recusa com 502 (não tenta adivinhar o formato).
// - Converte os erros da API para os status do site e limita os envios por minuto.
// - A imagem é repassada em stream: nada é gravado em disco e nada entra nas estatísticas do site.
// Documentação da API: docs/api.md e src/astro_classifier/api/schemas.py (na raiz do repositório).
import { deflateSync, crc32 } from 'node:zlib';
import { resolveLabel } from './taxonomy.js';

export const CONTRACT_MAJOR = 1;
export const AMBIGUOUS_GAP = 0.15;
export const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;
const TIMEOUT_MS = 60_000;

const KIND = { galaxy: 'galaxia', nebula: 'nebulosa' };

export const MESSAGES = {
  invalid: 'Imagem inválida. Use JPG, PNG ou WebP.',
  tooLarge: 'A imagem precisa ter até 10 MB.',
  unavailable: 'IA indisponível. Tente mais tarde.',
  contract: 'A IA foi atualizada e o site ainda não. Avise a equipe.',
  unexpected: 'A IA respondeu num formato que o site não conhece. Avise a equipe.',
  tooMany: 'Muitos envios seguidos. Espere um minuto.',
};

/** Erro com o status HTTP que o site deve devolver. */
export class ClassifierError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

/** true quando o número principal de contract_version é o que o site entende (1.x). */
export function contractSupported(version) {
  const major = Number.parseInt(String(version ?? '').split('.')[0], 10);
  return major === CONTRACT_MAJOR;
}

/** Diferença entre as 2 maiores probabilidades de um nível é menor que 0,15? */
function isAmbiguous(level) {
  const probs = (level?.scores ?? []).map((s) => s.probability).sort((a, b) => b - a);
  if (probs.length < 2) return false;
  return probs[0] - probs[1] < AMBIGUOUS_GAP;
}

/** Resposta de POST /predict (contrato 1.x) -> formato da tela. Ignora label_pt e summary_pt. */
export function traduzirPredicao(body) {
  if (!body || typeof body !== 'object') throw new ClassifierError(502, MESSAGES.unexpected);
  if (!contractSupported(body.contract_version)) throw new ClassifierError(502, MESSAGES.contract);
  if (typeof body.object !== 'string' || !Array.isArray(body.levels) || !body.levels.length) {
    throw new ClassifierError(502, MESSAGES.unexpected);
  }

  const subtype = typeof body.subtype === 'string' ? body.subtype : null;
  const label = subtype ?? body.object;
  const kind = KIND[body.object] ?? 'outro';
  // "other" não tem aba; subtipo desconhecido cai na categoria do objeto.
  const place = kind === 'outro' ? null : (resolveLabel(label) ?? resolveLabel(body.object));
  // Nível que decidiu a aba: o do subtipo, ou o do objeto quando não há subtipo.
  const decider = subtype ? (body.levels.find((l) => l.predicted === subtype) ?? body.levels.at(-1)) : body.levels[0];

  return {
    contractVersion: String(body.contract_version),
    mock: Boolean(body.mock),
    kind,
    category: place?.category ?? null,
    type: subtype ? (place?.type ?? null) : null,
    subfilter: null,
    label,
    confidence: subtype ? (body.subtype_confidence ?? decider.confidence) : body.confidence,
    objectConfidence: body.confidence,
    ambiguous: isAmbiguous(decider),
    outOfDomain: Boolean(body.domain?.out_of_domain),
    steps: body.levels.map((l) => ({
      level: l.level,
      predicted: l.predicted,
      confidence: l.confidence,
      scores: (l.scores ?? []).map((s) => ({ label: s.label, probability: s.probability })),
    })),
    modelVersion: body.levels.map((l) => l.model_version).filter(Boolean).join(' + ') || null,
    inferenceMs: typeof body.inference_ms === 'number' ? body.inference_ms : null,
  };
}

/** Resposta de GET /health -> estado da IA para o site. null = API fora do ar. */
export function traduzirSaude(body) {
  if (!body) return { connected: false, demo: false, mode: null, device: null, modelsLoaded: [], contractOk: false };
  const contractOk = contractSupported(body.contract_version);
  return {
    connected: body.status === 'ok' && contractOk,
    demo: body.mode === 'mock',
    mode: body.mode ?? null,
    device: body.device ?? null,
    // Vazio é normal: os modelos só carregam no 1º uso.
    modelsLoaded: Array.isArray(body.models_loaded) ? body.models_loaded : [],
    contractOk,
  };
}

/** Status de erro da API -> status e mensagem do site (fixtures/contrato-1.0/erros.json). */
export function traduzirErro(status) {
  if ([400, 415, 422].includes(status)) return new ClassifierError(422, MESSAGES.invalid);
  if (status === 413) return new ClassifierError(413, MESSAGES.tooLarge);
  return new ClassifierError(503, MESSAGES.unavailable);
}

/** Limite de envios por minuto e por endereço (mesmo estilo do limite do chat). */
export function createRateLimiter(limit, windowMs = 60_000) {
  const hits = new Map();
  return function limited(ip) {
    const now = Date.now();
    const recent = (hits.get(ip) ?? []).filter((t) => now - t < windowMs);
    recent.push(now);
    hits.set(ip, recent);
    if (hits.size > 5000) hits.clear();
    return recent.length > limit;
  };
}

/** PNG pequeno gerado em memória (usado só no aquecimento; nada vai para o disco). */
export function tinyPng(size = 32) {
  const chunk = (type, data) => {
    const len = Buffer.alloc(4);
    len.writeUInt32BE(data.length);
    const body = Buffer.concat([Buffer.from(type, 'ascii'), data]);
    const crc = Buffer.alloc(4);
    crc.writeUInt32BE(crc32(body));
    return Buffer.concat([len, body, crc]);
  };
  const header = Buffer.alloc(13);
  header.writeUInt32BE(size, 0);
  header.writeUInt32BE(size, 4);
  header[8] = 8; // 8 bits por canal
  header[9] = 2; // RGB
  const rows = [];
  const c = size / 2;
  for (let y = 0; y < size; y++) {
    const row = Buffer.alloc(1 + size * 3); // byte 0 = filtro "nenhum"
    for (let x = 0; x < size; x++) {
      const v = Math.max(0, 255 - Math.round(Math.hypot(x - c, y - c) * 24)); // mancha clara no centro
      row.set([v, v, Math.min(255, v + 20)], 1 + x * 3);
    }
    rows.push(row);
  }
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', header),
    chunk('IDAT', deflateSync(Buffer.concat(rows))),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/**
 * Cliente da API do colega.
 * baseUrl: endereço da API (ML_SERVICE_URL). perMinute: envios por minuto e por endereço (CLASSIFY_PER_MINUTE).
 */
export function createClassifier({ baseUrl, perMinute = 10, timeoutMs = TIMEOUT_MS } = {}) {
  const url = String(baseUrl || 'http://127.0.0.1:8000').replace(/\/+$/, '');
  const limited = createRateLimiter(perMinute);
  let lastVersion = null;

  /** GET /health da API; null quando ela não responde. */
  async function rawHealth() {
    try {
      const r = await fetch(`${url}/health`, { signal: AbortSignal.timeout(3000) });
      return r.ok ? await r.json() : null;
    } catch {
      return null;
    }
  }

  /** Estado da IA para /api/health. */
  async function health() {
    const raw = await rawHealth();
    const h = traduzirSaude(raw);
    return {
      online: Boolean(raw),
      ...h,
      version: lastVersion,
      message: h.connected
        ? h.demo
          ? 'IA no ar em modo simulado.'
          : 'IA conectada.'
        : raw
          ? MESSAGES.contract
          : 'IA fora do ar.',
    };
  }

  /** Repassa o corpo multipart (stream) para POST /predict e devolve a resposta traduzida. */
  async function classify(body, contentType, contentLength) {
    let r;
    try {
      r = await fetch(`${url}/predict`, {
        method: 'POST',
        headers: { 'content-type': contentType, 'content-length': String(contentLength) },
        body,
        duplex: 'half',
        signal: AbortSignal.timeout(timeoutMs),
      });
    } catch {
      throw new ClassifierError(503, MESSAGES.unavailable); // fora do ar ou passou de 60 s
    }
    if (!r.ok) {
      await r.body?.cancel().catch(() => {});
      throw traduzirErro(r.status);
    }
    const json = await r.json().catch(() => null);
    const result = traduzirPredicao(json);
    lastVersion = result.modelVersion;
    return result;
  }

  /** Rota Express de /api/classify. */
  async function route(req, res) {
    if (limited(req.ip)) return res.status(429).json({ error: MESSAGES.tooMany });
    const type = req.get('content-type') || '';
    if (!type.startsWith('multipart/form-data')) return res.status(422).json({ error: MESSAGES.invalid });
    const length = Number(req.get('content-length') || 0);
    if (!length || length > MAX_UPLOAD_BYTES + 64 * 1024) return res.status(413).json({ error: MESSAGES.tooLarge });
    try {
      res.json(await classify(req, type, length));
    } catch (err) {
      const status = err instanceof ClassifierError ? err.status : 503;
      if (status >= 500) console.error('[classificador]', err.message);
      res.status(status).json({ error: err instanceof ClassifierError ? err.message : MESSAGES.unavailable });
    }
  }

  /**
   * Aquecimento: com a API em modo real, manda uma imagem pequena para os modelos já carregarem.
   * Se a API estiver fora, tenta de novo a cada `intervalMs` (padrão 30 s), no máximo `retries` vezes.
   * Devolve 'warmed', 'skipped' (modo simulado ou contrato diferente) ou 'offline'.
   */
  async function warmup({ retries = 5, intervalMs = 30_000, log = console.log } = {}) {
    for (let attempt = 0; attempt <= retries; attempt++) {
      if (attempt > 0) await sleep(intervalMs);
      const raw = await rawHealth();
      if (!raw) continue; // fora do ar
      const h = traduzirSaude(raw);
      if (!h.connected || h.demo) {
        log(`[classificador] Aquecimento não precisa (modo ${h.mode ?? 'desconhecido'}).`);
        return 'skipped';
      }
      const form = new FormData();
      form.append('file', new Blob([tinyPng()], { type: 'image/png' }), 'aquecimento.png');
      try {
        const r = await fetch(`${url}/predict`, { method: 'POST', body: form, signal: AbortSignal.timeout(timeoutMs) });
        if (r.ok) {
          const result = traduzirPredicao(await r.json());
          lastVersion = result.modelVersion;
          log(`[classificador] Modelos carregados (${result.modelVersion}).`);
          return 'warmed';
        }
        await r.body?.cancel().catch(() => {});
      } catch {
        // tenta de novo
      }
    }
    log('[classificador] IA fora do ar; o aquecimento desistiu.');
    return 'offline';
  }

  return { health, classify, route, warmup, get lastVersion() { return lastVersion; } };
}
