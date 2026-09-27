// Armazenamento simples em arquivo JSON (suficiente para o esboço).
// Troque por um banco de dados (ex.: PostgreSQL/SQLite) quando o modelo estiver em produção.
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { randomUUID } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DATA_DIR = path.resolve(__dirname, '../data');
const DATA_FILE = path.join(DATA_DIR, 'classifications.json');

function load() {
  if (!existsSync(DATA_FILE)) return [];
  try {
    const parsed = JSON.parse(readFileSync(DATA_FILE, 'utf8'));
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

let items = load();

function persist() {
  if (!existsSync(DATA_DIR)) mkdirSync(DATA_DIR, { recursive: true });
  writeFileSync(DATA_FILE, JSON.stringify(items, null, 2));
}

export function list({ category, type, subfilter, limit = 100 } = {}) {
  return items
    .filter((i) => (!category || i.category === category))
    .filter((i) => (!type || i.type === type))
    .filter((i) => (!subfilter || i.subfilter === subfilter))
    .sort((a, b) => b.createdAt.localeCompare(a.createdAt))
    .slice(0, limit);
}

export function add(entry) {
  const record = { id: randomUUID(), createdAt: new Date().toISOString(), ...entry };
  items.push(record);
  persist();
  return record;
}

export function stats() {
  const byCategory = {};
  const byType = {};
  let confidenceSum = 0;
  for (const i of items) {
    byCategory[i.category] = (byCategory[i.category] ?? 0) + 1;
    const key = `${i.category}/${i.type}`;
    byType[key] = (byType[key] ?? 0) + 1;
    confidenceSum += i.confidence;
  }
  return {
    total: items.length,
    byCategory,
    byType,
    meanConfidence: items.length ? confidenceSum / items.length : null,
    lastUpdate: items.length ? items.map((i) => i.createdAt).sort().at(-1) : null,
  };
}
