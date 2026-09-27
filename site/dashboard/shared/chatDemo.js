// Respostas de DEMONSTRAÇÃO do chat (sem IA de verdade).
// Usadas enquanto nenhuma API de IA estiver configurada — pelo servidor (/api/chat)
// e pela pré-visualização estática do site. Responde com base em shared/taxonomy.json.
//
// Uso: demoReply(taxonomy, 'pt' | 'en', 'texto do usuário') -> string

const norm = (s) =>
  String(s || '')
    .toLowerCase()
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    .replace(/[^a-z0-9 ]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();

// palavras curtas (até 3 letras) precisam bater inteiras; as outras podem ser começo de palavra
const has = (text, words) => words.some((w) => new RegExp(w.length <= 3 ? `(^| )${w}( |$)` : `(^| )${w}`).test(text));

const TXT = {
  pt: {
    hello:
      'Olá! Sou o assistente do Cassyn. Posso explicar os tipos de galáxias e nebulosas, como a IA do site funciona ou como enviar uma foto. Por onde quer começar?',
    galaxies: 'Separamos as galáxias em {n} tipos:',
    nebulae: 'Separamos as nebulosas em {n} tipos:',
    askMore: 'Quer saber mais sobre algum deles? É só perguntar pelo nome.',
    recognize: 'Como reconhecer',
    examples: 'Exemplos famosos',
    kinds: 'Tipos',
    how: 'A IA do site olha a foto de um objeto do espaço e diz com qual tipo ele mais se parece, mostrando o quanto ela tem certeza. Ela aprendeu vendo milhares de fotos que voluntários já tinham separado. Ela pode errar, por isso os casos com dúvida ficam marcados.',
    upload: 'Para enviar uma foto, abra "Enviar imagem" no menu, escolha uma foto do espaço (JPG, PNG ou WebP, até 10 MB) e confirme que você pode usá-la. A foto é sempre apagada logo depois da análise.',
    privacy: 'Não pedimos dados pessoais e não vendemos nada. Nesta conversa, evite escrever nome, telefone ou endereço. Os detalhes estão na página de Privacidade, no rodapé.',
    thanks: 'De nada! Se tiver outra dúvida sobre o espaço, é só perguntar.',
    space:
      'O universo tem centenas de bilhões de galáxias, e cada uma tem bilhões de estrelas. As nebulosas são nuvens de gás e poeira dentro das galáxias, onde estrelas nascem ou morrem.',
    fallback:
      'Ainda estou em modo de demonstração, então só sei responder sobre os tipos de galáxias e nebulosas, como a IA funciona e como enviar fotos. Tente perguntar, por exemplo: "O que é uma nebulosa planetária?"',
  },
  en: {
    hello:
      "Hi! I'm the Cassyn assistant. I can explain the types of galaxies and nebulae, how the site's AI works or how to upload a photo. Where would you like to start?",
    galaxies: 'We sort galaxies into {n} types:',
    nebulae: 'We sort nebulae into {n} types:',
    askMore: 'Want to know more about one of them? Just ask by name.',
    recognize: 'How to recognise it',
    examples: 'Famous examples',
    kinds: 'Kinds',
    how: "The site's AI looks at a photo of a space object and says which type it resembles most, showing how sure it is. It learned by looking at thousands of photos that volunteers had already sorted. It can be wrong, so uncertain cases are flagged.",
    upload: 'To upload a photo, open "Upload an image" in the menu, choose a space photo (JPG, PNG or WebP, up to 10 MB) and confirm you may use it. The photo is always deleted right after the analysis.',
    privacy: 'We ask for no personal data and sell nothing. In this chat, avoid writing your name, phone or address. Details are on the Privacy page, in the footer.',
    thanks: "You're welcome! If you have another question about space, just ask.",
    space:
      'The universe has hundreds of billions of galaxies, each with billions of stars. Nebulae are clouds of gas and dust inside galaxies, where stars are born or die.',
    fallback:
      'I am still in demo mode, so I can only answer about the types of galaxies and nebulae, how the AI works and how to upload photos. Try asking, for example: "What is a planetary nebula?"',
  },
};

// palavras extras para achar cada tipo (além do nome)
const ALIASES = {
  elipticas: ['eliptic', 'ellipti', 'oval', 'redond'],
  alongadas: ['alongad', 'elongat', 'charuto', 'cigar'],
  espirais: ['espira', 'spiral', 'barrad', 'barred', 'via lactea', 'milky way', 'andromeda', 'cata vento', 'pinwheel', 'redemoinho', 'whirlpool'],
  'de-perfil': ['perfil', 'edge on', 'de lado', 'sombrero'],
  'em-fusao': ['colis', 'collid', 'fusao', 'merg', 'antenas', 'antennae'],
  irregulares: ['irregul', 'magalhaes', 'magellan'],
  emissao: ['emiss', 'orion', 'aguia', 'eagle', 'pilares', 'pillars'],
  reflexao: ['reflex', 'reflect', 'pleiades', 'pleiad'],
  escuras: ['escur', 'dark', 'cabeca de cavalo', 'horsehead'],
  planetarias: ['planetar', 'planetary', 'anel', 'ring nebula', 'helice', 'helix'],
  remanescentes: ['supernova', 'caranguejo', 'crab', 'explod', 'restos', 'remain'],
  'bolhas-jatos': ['bolha', 'bubble', 'jato', 'jet'],
};

function pick(item, lang, field) {
  return (lang === 'en' ? item.en?.[field] : undefined) ?? item[field];
}

export function demoReply(taxonomy, lang, userText) {
  const L = TXT[lang === 'en' ? 'en' : 'pt'];
  const text = norm(userText);
  if (!text) return L.fallback;

  // 1) um tipo específico
  for (const cat of taxonomy.categories) {
    for (const ty of cat.types) {
      const names = [norm(ty.name), norm(ty.en?.name), ...(ALIASES[ty.id] ?? [])].filter(Boolean);
      if (names.some((n) => text.includes(n))) {
        const cues = pick(ty, lang, 'visualCues').map((c) => `• ${c}`).join('\n');
        const ex = pick(ty, lang, 'examples').join(', ');
        const kinds = ty.subfilters?.length ? `\n\n${L.kinds}: ${ty.subfilters.map((s) => pick(s, lang, 'name')).join(', ')}.` : '';
        return `${pick(ty, lang, 'name')} (${pick(cat, lang, 'name').toLowerCase()}): ${pick(ty, lang, 'description')}\n\n${L.recognize}:\n${cues}${kinds}\n\n${L.examples}: ${ex}.`;
      }
    }
  }

  // 2) assuntos gerais
  if (has(text, ['oi', 'ola', 'hello', 'hi', 'hey', 'bom dia', 'boa tarde', 'boa noite', 'good'])) return L.hello;
  if (has(text, ['obrigad', 'valeu', 'thank', 'thanks'])) return L.thanks;
  if (has(text, ['privacidade', 'dados', 'privacy', 'data', 'lgpd'])) return L.privacy;
  if (has(text, ['envi', 'foto', 'imagem', 'upload', 'photo', 'image'])) return L.upload;
  if (has(text, ['como funciona', 'ia', 'inteligencia', 'ai', 'how', 'rede neural', 'modelo', 'model'])) return L.how;
  const list = (cat, key) =>
    `${L[key].replace('{n}', cat.types.length)}\n${cat.types.map((ty) => `• ${pick(ty, lang, 'name')}`).join('\n')}\n\n${L.askMore}`;
  if (has(text, ['galax'])) return list(taxonomy.categories[0], 'galaxies');
  if (has(text, ['nebul'])) return list(taxonomy.categories[1], 'nebulae');
  if (has(text, ['universo', 'universe', 'espaco', 'space', 'estrela', 'star'])) return L.space;
  return L.fallback;
}
