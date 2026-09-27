import taxonomy from '@shared/taxonomy.json';

export { taxonomy };
export const getCategory = (id) => taxonomy.categories.find((c) => c.id === id) ?? null;
export const getType = (catId, typeId) => getCategory(catId)?.types.find((t) => t.id === typeId) ?? null;

/** Endereço antigo de aba juntada (ex.: "espirais-barradas") -> { type, subfilter } */
export const getLegacyType = (catId, legacyId) => {
  for (const type of getCategory(catId)?.types ?? []) {
    const sub = type.subfilters?.find((s) => s.legacyId === legacyId);
    if (sub) return { type: type.id, subfilter: sub.id };
  }
  return null;
};
