// Carrega a taxonomia compartilhada (shared/taxonomy.json) e resolve
// rótulos do modelo -> categoria/sub-aba do dashboard.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const TAXONOMY_PATH = path.resolve(__dirname, '../../shared/taxonomy.json');

export const taxonomy = JSON.parse(readFileSync(TAXONOMY_PATH, 'utf8'));

/** Índice: rótulo do modelo (minúsculo) -> { category, type, subfilter } */
const labelIndex = new Map();
for (const category of taxonomy.categories) {
  // nível 1 da cascata (ex.: "galaxy"): só a categoria, sem aba
  for (const label of category.modelLabels ?? []) {
    labelIndex.set(label.toLowerCase(), { category: category.id, type: null, subfilter: null });
  }
  for (const type of category.types) {
    for (const label of type.modelLabels ?? []) {
      labelIndex.set(label.toLowerCase(), { category: category.id, type: type.id, subfilter: null });
    }
    for (const sub of type.subfilters ?? []) {
      for (const label of sub.modelLabels ?? []) {
        labelIndex.set(label.toLowerCase(), { category: category.id, type: type.id, subfilter: sub.id });
      }
    }
  }
}

export function resolveLabel(label) {
  if (typeof label !== 'string') return null;
  return labelIndex.get(label.trim().toLowerCase()) ?? null;
}

export function findCategory(categoryId) {
  return taxonomy.categories.find((c) => c.id === categoryId) ?? null;
}

export function findType(categoryId, typeId) {
  return findCategory(categoryId)?.types.find((t) => t.id === typeId) ?? null;
}

/** Endereço antigo de aba que foi juntada (ex.: "espirais-barradas") -> { type, subfilter } */
export function findLegacyType(categoryId, legacyId) {
  for (const type of findCategory(categoryId)?.types ?? []) {
    const sub = type.subfilters?.find((s) => s.legacyId === legacyId);
    if (sub) return { type: type.id, subfilter: sub.id };
  }
  return null;
}

/** Rotas válidas do front-end (usadas para devolver HTTP 404 real em rotas desconhecidas). */
export function isKnownClientRoute(urlPath) {
  const p = urlPath.replace(/\/+$/, '') || '/';
  const staticRoutes = ['/', '/classificar', '/modelo', '/termos', '/privacidade', '/creditos', '/conformidade', '/chat', '/planetas', '/estrelas'];
  if (staticRoutes.includes(p)) return true;
  const m = p.match(/^\/(galaxias|nebulosas)(?:\/([a-z0-9-]+))?$/);
  if (!m) return false;
  const [, cat, type] = m;
  return type ? Boolean(findType(cat, type) || findLegacyType(cat, type)) : true;
}
