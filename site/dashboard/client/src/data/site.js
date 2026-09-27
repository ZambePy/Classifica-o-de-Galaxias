// Dados institucionais usados nas páginas legais.
// PREENCHA os campos marcados antes de publicar o site.
export const site = {
  name: 'Cassyn',
  nameEn: 'Cassyn',
  responsible: '[NOME DO(A) RESPONSÁVEL / INSTITUIÇÃO]',
  advisor: '[NOME DO(A) PROFESSOR(A) ORIENTADOR(A)]',
  privacyEmail: '[E-MAIL DE PRIVACIDADE]',
  city: '[CIDADE/UF]',
  hosting: '[PROVEDOR E REGIÃO DE HOSPEDAGEM — ex.: região São Paulo (Brasil)]',
  lastUpdate: '23/09/2026',
  version: '0.1.0',
};

export const siteName = (lang) => (lang === 'en' ? site.nameEn : site.name);
