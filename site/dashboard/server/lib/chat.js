// Chat do site: escolhe quem responde.
//
// Hoje: modo DEMONSTRAÇÃO (respostas prontas de shared/chatDemo.js), sem nenhuma IA externa.
// Depois: para usar uma API de IA, defina no .env:
//   CHAT_PROVIDER=openai-compatible
//   CHAT_API_URL=https://<provedor>/v1/chat/completions   (qualquer API no formato "chat completions")
//   CHAT_API_KEY=<sua chave>                                 (fica só no servidor, nunca vai para o navegador)
//   CHAT_MODEL=<nome do modelo>
// Para outro formato de API, crie uma função como `askOpenAICompatible` e chame-a em `chatReply`.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { demoReply } from '../../shared/chatDemo.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const taxonomy = JSON.parse(readFileSync(path.resolve(__dirname, '../../shared/taxonomy.json'), 'utf8'));

const PROVIDER = process.env.CHAT_PROVIDER || 'demo';
const API_URL = process.env.CHAT_API_URL || '';
const API_KEY = process.env.CHAT_API_KEY || '';
const MODEL = process.env.CHAT_MODEL || '';

export const chatIsDemo = () => !(PROVIDER === 'openai-compatible' && API_URL && API_KEY && MODEL);

// Instruções para a IA externa (quando houver): foco no tema, linguagem simples, segurança.
const SYSTEM_PROMPT = {
  pt: `Você é o assistente do Cassyn, um site educacional brasileiro que classifica fotos de galáxias e nebulosas com inteligência artificial.
Responda em português do Brasil, com linguagem simples, frases curtas e sem jargão. O público pode incluir estudantes e adolescentes.
Fale sobre astronomia, sobre os tipos usados no site e sobre como o site funciona. Se a pergunta fugir do tema, explique com gentileza que você só ajuda com o espaço e com o site.
Não peça dados pessoais. Se não souber algo, diga que não sabe. Tipos do site: ${taxonomy.categories
    .map((c) => `${c.name}: ${c.types.map((t) => t.name).join(', ')}`)
    .join('. ')}.`,
  en: `You are the assistant of Cassyn, a Brazilian educational site that classifies photos of galaxies and nebulae with artificial intelligence.
Answer in English, in plain language, with short sentences and no jargon. The audience may include students and teenagers.
Talk about astronomy, the types used on the site and how the site works. If a question is off topic, kindly explain you only help with space and the site.
Do not ask for personal data. If you do not know something, say so. Site types: ${taxonomy.categories
    .map((c) => `${c.en.name}: ${c.types.map((t) => t.en.name).join(', ')}`)
    .join('. ')}.`,
};

async function askOpenAICompatible(messages, lang) {
  const res = await fetch(API_URL, {
    method: 'POST',
    headers: { 'content-type': 'application/json', authorization: `Bearer ${API_KEY}` },
    body: JSON.stringify({
      model: MODEL,
      max_tokens: 500,
      temperature: 0.4,
      messages: [{ role: 'system', content: SYSTEM_PROMPT[lang] }, ...messages],
    }),
    signal: AbortSignal.timeout(30_000),
  });
  if (!res.ok) throw new Error(`API de IA respondeu ${res.status}`);
  const data = await res.json();
  const text = data?.choices?.[0]?.message?.content;
  if (typeof text !== 'string' || !text.trim()) throw new Error('Resposta vazia da API de IA');
  return text.trim();
}

/**
 * @param {{ messages: {role:'user'|'assistant', content:string}[], lang: 'pt'|'en' }} input
 * @returns {Promise<{ reply: string, demo: boolean }>}
 */
export async function chatReply({ messages, lang }) {
  if (!chatIsDemo()) {
    return { reply: await askOpenAICompatible(messages, lang), demo: false };
  }
  const last = messages[messages.length - 1]?.content ?? '';
  return { reply: demoReply(taxonomy, lang, last), demo: true };
}
