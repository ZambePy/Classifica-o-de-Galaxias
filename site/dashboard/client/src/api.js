// Cliente da API. Se o servidor não estiver disponível (ex.: pré-visualização estática),
// devolve dados vazios em vez de quebrar a página.
const EMPTY_STATS = { total: 0, byCategory: {}, byType: {}, meanConfidence: null, lastUpdate: null };

async function get(path, fallback) {
  try {
    const res = await fetch(`/api${path}`, { headers: { Accept: 'application/json' } });
    if (!res.ok) return fallback;
    return await res.json();
  } catch {
    return fallback;
  }
}

export const api = {
  health: () => get('/health', { status: 'offline', model: { connected: false } }),
  stats: () => get('/stats', EMPTY_STATS),
  classifications: (params = {}) => {
    const q = new URLSearchParams(Object.entries(params).filter(([, v]) => v)).toString();
    return get(`/classifications${q ? `?${q}` : ''}`, { items: [] });
  },
};
