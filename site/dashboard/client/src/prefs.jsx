// Preferências do visitante: idioma (PT/EN) e tema (sistema/claro/escuro).
// Ficam salvas só no navegador (localStorage), com tratamento de erro caso o armazenamento esteja bloqueado.
import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { messages } from './i18n/messages.js';

const LANG_KEY = 'pg-idioma';
const THEME_KEY = 'pg-tema';

function read(key) {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}
function write(key, value) {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    /* armazenamento indisponível: a preferência vale só nesta visita */
  }
}

function initialLang() {
  const saved = read(LANG_KEY);
  if (saved === 'pt' || saved === 'en') return saved;
  return 'pt'; // português é o idioma padrão do projeto
}

function initialTheme() {
  const saved = read(THEME_KEY);
  if (saved === 'light' || saved === 'dark' || saved === 'system') return saved;
  return 'system';
}

const PrefsContext = createContext(null);

export function PrefsProvider({ children }) {
  const [lang, setLangState] = useState(initialLang);
  const [theme, setThemeState] = useState(initialTheme);
  const [systemDark, setSystemDark] = useState(() => window.matchMedia?.('(prefers-color-scheme: dark)').matches ?? true);

  useEffect(() => {
    const mq = window.matchMedia?.('(prefers-color-scheme: dark)');
    if (!mq) return undefined;
    const onChange = (e) => setSystemDark(e.matches);
    mq.addEventListener('change', onChange);
    return () => mq.removeEventListener('change', onChange);
  }, []);

  const resolvedTheme = theme === 'system' ? (systemDark ? 'dark' : 'light') : theme;

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', resolvedTheme);
  }, [resolvedTheme]);

  useEffect(() => {
    document.documentElement.lang = lang === 'en' ? 'en' : 'pt-BR';
    document.title = 'Cassyn';
  }, [lang]);

  const setLang = useCallback((l) => {
    setLangState(l);
    write(LANG_KEY, l);
  }, []);
  const setTheme = useCallback((t) => {
    setThemeState(t);
    write(THEME_KEY, t);
  }, []);

  const value = useMemo(() => {
    const dict = messages[lang];
    /** t('chave', {var}) — texto da interface no idioma atual */
    const t = (key, vars) => {
      let s = dict[key] ?? messages.pt[key] ?? key;
      if (vars) for (const [k, v] of Object.entries(vars)) s = s.replaceAll(`{${k}}`, v);
      return s;
    };
    /** tx(item, 'campo') — campo traduzido de um item da taxonomia */
    const tx = (item, field) => (lang === 'en' ? item?.en?.[field] : undefined) ?? item?.[field];
    const nf = new Intl.NumberFormat(lang === 'en' ? 'en-US' : 'pt-BR');
    return { lang, setLang, theme, setTheme, resolvedTheme, t, tx, nf };
  }, [lang, setLang, theme, setTheme, resolvedTheme]);

  return <PrefsContext.Provider value={value}>{children}</PrefsContext.Provider>;
}

export const usePrefs = () => useContext(PrefsContext);
