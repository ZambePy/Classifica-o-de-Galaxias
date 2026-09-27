import { useEffect, useRef, useState } from 'react';
import { usePrefs } from '../prefs.jsx';
import { useData } from '../components/DataContext.jsx';
import { taxonomy } from '../data/taxonomy.js';
import { demoReply } from '@shared/chatDemo.js';
import logo from '../assets/logo.svg';
import { IconInfo, IconSend } from '../components/Icons.jsx';

const MAX_LEN = 1000;
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

// Pergunta ao servidor; se ele não existir (pré-visualização estática), responde com o modo demonstração local.
async function ask(messages, lang) {
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ messages, lang }),
    });
    const body = await res.json().catch(() => null);
    if (res.ok && body?.reply) return { reply: body.reply, demo: body.demo };
    if (res.status === 429 || res.status === 502 || res.status === 422) return { error: res.status };
  } catch {
    /* sem servidor: cai no modo demonstração */
  }
  return { reply: demoReply(taxonomy, lang, messages[messages.length - 1].content), demo: true };
}

function Bubble({ role, children }) {
  const mine = role === 'user';
  return (
    <div className={`msg ${mine ? 'msg-user' : 'msg-ai'}`}>
      {!mine && <img src={logo} alt="" className="msg-avatar" />}
      <div className="msg-bubble">{children}</div>
    </div>
  );
}

export default function Chat() {
  const { t, lang } = usePrefs();
  const { health } = useData();
  const [messages, setMessages] = useState([]); // só nesta aba; nada é guardado
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const listRef = useRef(null);
  const inputRef = useRef(null);
  const demo = health?.chat?.demo ?? true;

  useEffect(() => {
    const el = listRef.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
  }, [messages, busy]);

  async function send(text) {
    const trimmed = text.trim().slice(0, MAX_LEN);
    // todo texto começa com letra maiúscula, inclusive a mensagem da pessoa
    const content = trimmed.charAt(0).toUpperCase() + trimmed.slice(1);
    if (!content || busy) return;
    const next = [...messages, { role: 'user', content }];
    setMessages(next);
    setInput('');
    setError('');
    setBusy(true);
    const started = Date.now();
    const result = await ask(next, lang);
    const elapsed = Date.now() - started;
    if (elapsed < 600) await wait(600 - elapsed); // pequena pausa para a resposta não "pular" na tela
    setBusy(false);
    if (result.error) {
      setError(t(result.error === 429 ? 'chat.err429' : 'chat.errGeneric'));
      return;
    }
    setMessages((m) => [...m, { role: 'assistant', content: result.reply }]);
    inputRef.current?.focus();
  }

  const onKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send(input);
    }
  };

  const suggestions = ['chat.s1', 'chat.s2', 'chat.s3', 'chat.s4'];

  return (
    <>
      <header className="page-head">
        <span className="eyebrow">{t('cl.eyebrow')}</span>
        <h1>{t('chat.title')}</h1>
        <p>{t('chat.lead')}</p>
      </header>

      <section className="panel chat" aria-label={t('chat.title')}>
        <div className="chat-top">
          <span className={`pill ${demo ? 'warn' : 'aqua'}`}>
            <span className={`dot${demo ? '' : ' on'}`} /> {demo ? t('chat.demo') : t('chat.live')}
          </span>
          {messages.length > 0 && (
            <button type="button" className="btn ghost chat-new" onClick={() => setMessages([])}>
              {t('chat.new')}
            </button>
          )}
        </div>

        <div className="chat-list" ref={listRef} aria-live="polite">
          <Bubble role="assistant">{t('chat.welcome')}</Bubble>
          {messages.map((m, i) => (
            <Bubble key={i} role={m.role}>
              {m.content}
            </Bubble>
          ))}
          {busy && (
            <Bubble role="assistant">
              <span className="typing" aria-label={t('chat.typing')}>
                <span />
                <span />
                <span />
              </span>
            </Bubble>
          )}
          {messages.length === 0 && (
            <div className="chat-suggestions">
              {suggestions.map((k) => (
                <button key={k} type="button" className="suggestion" onClick={() => send(t(k))}>
                  {t(k)}
                </button>
              ))}
            </div>
          )}
        </div>

        {error && (
          <p role="alert" className="chat-error">
            {error}
          </p>
        )}

        <form
          className="chat-form"
          onSubmit={(e) => {
            e.preventDefault();
            send(input);
          }}
        >
          <label htmlFor="chat-input" className="sr-only">
            {t('chat.placeholder')}
          </label>
          <textarea
            id="chat-input"
            autoCapitalize="sentences"
            ref={inputRef}
            rows={1}
            value={input}
            maxLength={MAX_LEN}
            placeholder={t('chat.placeholder')}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
          />
          <button type="submit" className="btn chat-send" disabled={!input.trim() || busy} aria-label={t('chat.send')}>
            <IconSend />
          </button>
        </form>
        <p className="chat-note">
          <IconInfo /> {t('chat.privacy')}
        </p>
      </section>
    </>
  );
}
