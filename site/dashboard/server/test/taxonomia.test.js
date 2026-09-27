// Confere se a taxonomia do site cobre todos os rótulos do classificador do colega (contrato 1.0)
// e se os exemplos em fixtures/contrato-1.0 estão coerentes com ela.
// Rode com:  npm run test:server
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { resolveLabel, isKnownClientRoute, findLegacyType } from '../lib/taxonomy.js';

const DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), 'fixtures', 'contrato-1.0');
const read = (f) => JSON.parse(readFileSync(path.join(DIR, f), 'utf8'));

// Rótulos de src/astro_classifier/taxonomy.py (repositório ZambePy/Classifica-o-de-Galaxias)
const CONTRATO = {
  object: ['galaxy', 'nebula', 'other'],
  galaxy: ['spiral', 'elliptical', 'irregular'],
  nebula: ['emission', 'reflection', 'planetary', 'supernova_remnant'],
};

test('Todo rótulo do colega tem lugar no site (menos "other", que não tem aba)', () => {
  for (const label of [...CONTRATO.object, ...CONTRATO.galaxy, ...CONTRATO.nebula]) {
    if (label === 'other') {
      assert.equal(resolveLabel(label), null, '"other" não deve cair em nenhuma aba');
      continue;
    }
    assert.ok(resolveLabel(label), `Rótulo sem aba no site: ${label}`);
  }
});

test('Subtipos de galáxia caem em Galáxias e de nebulosa em Nebulosas', () => {
  for (const l of CONTRATO.galaxy) assert.equal(resolveLabel(l).category, 'galaxias', l);
  for (const l of CONTRATO.nebula) assert.equal(resolveLabel(l).category, 'nebulosas', l);
  assert.deepEqual(resolveLabel('galaxy'), { category: 'galaxias', type: null, subfilter: null });
  assert.deepEqual(resolveLabel('nebula'), { category: 'nebulosas', type: null, subfilter: null });
});

test('As 3 abas de espirais viraram uma só, e os endereços antigos continuam válidos', () => {
  assert.equal(resolveLabel('spiral').type, 'espirais');
  assert.deepEqual(resolveLabel('galaxy10:5'), { category: 'galaxias', type: 'espirais', subfilter: 'barradas' });
  assert.deepEqual(findLegacyType('galaxias', 'espirais-abertas'), { type: 'espirais', subfilter: 'abertas' });
  assert.ok(isKnownClientRoute('/galaxias/espirais'));
  assert.ok(isKnownClientRoute('/galaxias/espirais-barradas'));
  assert.ok(!isKnownClientRoute('/galaxias/espirais-inexistentes'));
});

test('Exemplos do contrato batem com o esperado.json', () => {
  const { casos } = read('esperado.json');
  const arquivos = readdirSync(DIR).filter((f) => f.startsWith('predict-'));
  assert.ok(arquivos.length >= 8);
  for (const f of arquivos) {
    const body = read(f);
    const exp = casos[f];
    assert.ok(exp, `Sem expectativa para ${f}`);
    if (exp.rejeitar) {
      assert.notEqual(body.contract_version.split('.')[0], '1');
      continue;
    }
    assert.equal(body.contract_version.split('.')[0], '1', f);
    const where = body.object === 'other' ? null : resolveLabel(body.subtype ?? body.object);
    assert.equal(where?.category ?? null, exp.category, `${f}: categoria`);
    assert.equal(where?.type ?? null, exp.type, `${f}: aba`);
    assert.equal(body.domain.out_of_domain, exp.outOfDomain, `${f}: fora do domínio`);
    assert.equal(body.mock, exp.mock, `${f}: mock`);
  }
});
