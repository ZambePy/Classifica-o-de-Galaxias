// Testes do adaptador da API do colega (server/lib/classificador.js).
// - Todos os casos de fixtures/contrato-1.0/esperado.json e os erros de erros.json.
// - Uma API falsa (servidor http do Node numa porta livre): API fora do ar (503), contrato 2.0 (502),
//   11º envio no mesmo minuto (429) e o aquecimento.
// Rode com:  npm run test:server
import { test, describe, before, after } from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import express from 'express';
import {
  traduzirPredicao,
  traduzirSaude,
  traduzirErro,
  createClassifier,
  ClassifierError,
  tinyPng,
} from '../lib/classificador.js';

const DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), 'fixtures', 'contrato-1.0');
const read = (f) => JSON.parse(readFileSync(path.join(DIR, f), 'utf8'));
const { casos } = read('esperado.json');

describe('esperado.json', () => {
  for (const [file, expected] of Object.entries(casos)) {
    test(file, () => {
      const body = read(file);
      if (expected.rejeitar) {
        assert.throws(() => traduzirPredicao(body), (err) => err instanceof ClassifierError && err.status === 502);
        return;
      }
      if (file.startsWith('health-')) {
        const h = traduzirSaude(body);
        assert.equal(h.connected, expected.connected);
        assert.equal(h.demo, expected.demo);
        return;
      }
      const r = traduzirPredicao(body);
      for (const key of Object.keys(expected)) assert.equal(r[key], expected[key], `${file}: ${key}`);
    });
  }
});

describe('formato da resposta traduzida', () => {
  test('Galáxia espiral no formato da tela', () => {
    const r = traduzirPredicao(read('predict-galaxia-espiral.json'));
    assert.deepEqual(r, {
      contractVersion: '1.0',
      mock: false,
      kind: 'galaxia',
      category: 'galaxias',
      type: 'espirais',
      subfilter: null,
      label: 'spiral',
      confidence: 0.871,
      objectConfidence: 0.962,
      ambiguous: false,
      outOfDomain: false,
      steps: [
        {
          level: 'object',
          predicted: 'galaxy',
          confidence: 0.962,
          scores: [
            { label: 'galaxy', probability: 0.962 },
            { label: 'nebula', probability: 0.03 },
            { label: 'other', probability: 0.008 },
          ],
        },
        {
          level: 'galaxy',
          predicted: 'spiral',
          confidence: 0.871,
          scores: [
            { label: 'spiral', probability: 0.871 },
            { label: 'elliptical', probability: 0.102 },
            { label: 'irregular', probability: 0.027 },
          ],
        },
      ],
      modelVersion: 'object-e12 + galaxy-e18',
      inferenceMs: 41.7,
    });
  });

  test('Sem subtipo: label é o objeto e a confiança é a do objeto', () => {
    const r = traduzirPredicao(read('predict-sem-subtipo.json'));
    assert.equal(r.label, 'nebula');
    assert.equal(r.confidence, 0.955);
    assert.equal(r.steps.length, 1);
  });

  test('Não repassa label_pt nem summary_pt', () => {
    const json = JSON.stringify(traduzirPredicao(read('predict-nebulosa-emissao.json')));
    assert.ok(!json.includes('label_pt') && !json.includes('summary'));
  });

  test('Resposta sem campos obrigatórios vira 502', () => {
    assert.throws(() => traduzirPredicao({ contract_version: '1.0' }), (e) => e.status === 502);
    assert.throws(() => traduzirPredicao(null), (e) => e.status === 502);
  });

  test('Saúde com contrato 2.0 não conta como conectada', () => {
    assert.equal(traduzirSaude({ ...read('health-real.json'), contract_version: '2.0' }).connected, false);
    assert.equal(traduzirSaude(null).connected, false);
  });
});

describe('erros.json', () => {
  const esperado = { 400: 422, 413: 413, 415: 422, 422: 422, 500: 503 };
  for (const { status } of read('erros.json').erros) {
    test(`${status} vira ${esperado[status]}`, () => {
      assert.equal(traduzirErro(status).status, esperado[status]);
    });
  }
});

// ------------------------------------------------------------ API falsa

function listen(server) {
  return new Promise((resolve) => server.listen(0, '127.0.0.1', () => resolve(server.address().port)));
}
const close = (server) => new Promise((resolve) => server.close(resolve));

/** API falsa do colega. `state.mode`, `state.predict` (nome do fixture ou status numérico). */
function fakeApi(state) {
  return http.createServer((req, res) => {
    let size = 0;
    req.on('data', (c) => (size += c.length));
    req.on('end', () => {
      state.calls.push({ path: req.url, size });
      res.setHeader('content-type', 'application/json');
      if (req.url === '/health') {
        return res.end(JSON.stringify({ ...read('health-mock.json'), mode: state.mode }));
      }
      if (typeof state.predict === 'number') {
        res.statusCode = state.predict;
        return res.end(JSON.stringify({ detail: 'erro' }));
      }
      res.end(readFileSync(path.join(DIR, state.predict)));
    });
  });
}

/** Servidor Express mínimo com a mesma rota do site. */
function siteApp(classifier) {
  const app = express();
  app.post('/api/classify', classifier.route);
  return http.createServer(app);
}

async function send(port) {
  const form = new FormData();
  form.append('file', new Blob([tinyPng()], { type: 'image/png' }), 'foto.png');
  const r = await fetch(`http://127.0.0.1:${port}/api/classify`, { method: 'POST', body: form });
  return { status: r.status, body: await r.json() };
}

describe('API falsa', () => {
  const state = { mode: 'real', predict: 'predict-galaxia-espiral.json', calls: [] };
  let apiServer;
  let apiPort;

  before(async () => {
    apiServer = fakeApi(state);
    apiPort = await listen(apiServer);
  });
  after(() => close(apiServer));

  test('Repassa a imagem e devolve a resposta traduzida', async () => {
    const site = siteApp(createClassifier({ baseUrl: `http://127.0.0.1:${apiPort}` }));
    const port = await listen(site);
    try {
      const r = await send(port);
      assert.equal(r.status, 200);
      assert.equal(r.body.type, 'espirais');
      assert.ok(state.calls.at(-1).size > tinyPng().length, 'a imagem chegou à API');
    } finally {
      await close(site);
    }
  });

  test('API fora do ar: 503', async () => {
    const off = http.createServer();
    const offPort = await listen(off);
    await close(off); // porta livre, ninguém escutando
    const site = siteApp(createClassifier({ baseUrl: `http://127.0.0.1:${offPort}` }));
    const port = await listen(site);
    try {
      const r = await send(port);
      assert.equal(r.status, 503);
      assert.match(r.body.error, /^IA indisponível/);
    } finally {
      await close(site);
    }
  });

  test('Contrato 2.0: 502', async () => {
    state.predict = 'predict-contrato-2.0.json';
    const site = siteApp(createClassifier({ baseUrl: `http://127.0.0.1:${apiPort}` }));
    const port = await listen(site);
    try {
      const r = await send(port);
      assert.equal(r.status, 502);
      assert.match(r.body.error, /atualizada/);
    } finally {
      state.predict = 'predict-galaxia-espiral.json';
      await close(site);
    }
  });

  test('Erro 500 da API: 503', async () => {
    state.predict = 500;
    const site = siteApp(createClassifier({ baseUrl: `http://127.0.0.1:${apiPort}` }));
    const port = await listen(site);
    try {
      assert.equal((await send(port)).status, 503);
    } finally {
      state.predict = 'predict-galaxia-espiral.json';
      await close(site);
    }
  });

  test('11º envio no mesmo minuto: 429', async () => {
    const site = siteApp(createClassifier({ baseUrl: `http://127.0.0.1:${apiPort}`, perMinute: 10 }));
    const port = await listen(site);
    try {
      for (let i = 1; i <= 10; i++) assert.equal((await send(port)).status, 200, `envio ${i}`);
      const r = await send(port);
      assert.equal(r.status, 429);
      assert.match(r.body.error, /^Muitos envios/);
    } finally {
      await close(site);
    }
  });

  test('Aquecimento em modo real manda uma imagem; em modo simulado, não', async () => {
    const c = createClassifier({ baseUrl: `http://127.0.0.1:${apiPort}` });
    const quiet = () => {};
    state.calls.length = 0;
    assert.equal(await c.warmup({ log: quiet }), 'warmed');
    assert.equal(state.calls.filter((x) => x.path === '/predict').length, 1);
    assert.equal(c.lastVersion, 'object-e12 + galaxy-e18');

    state.mode = 'mock';
    state.calls.length = 0;
    assert.equal(await c.warmup({ log: quiet }), 'skipped');
    assert.equal(state.calls.filter((x) => x.path === '/predict').length, 0);
    state.mode = 'real';
  });

  test('Aquecimento com a API fora tenta de novo e desiste', async () => {
    const off = http.createServer();
    const offPort = await listen(off);
    await close(off);
    const c = createClassifier({ baseUrl: `http://127.0.0.1:${offPort}` });
    assert.equal(await c.warmup({ retries: 2, intervalMs: 10, log: () => {} }), 'offline');
  });
});
