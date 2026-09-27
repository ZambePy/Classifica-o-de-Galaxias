import { usePrefs } from '../prefs.jsx';
import { IconMonitor, IconMoon, IconSun } from './Icons.jsx';

/** Seletores de idioma (PT/EN) e tema (sistema/claro/escuro). */
export default function Prefs() {
  const { lang, setLang, theme, setTheme, t } = usePrefs();
  const themes = [
    ['system', IconMonitor, t('prefs.system')],
    ['light', IconSun, t('prefs.light')],
    ['dark', IconMoon, t('prefs.dark')],
  ];
  return (
    <div className="prefs">
      <div className="seg" role="group" aria-label={t('prefs.lang')}>
        {[
          ['pt', 'PT', 'Português'],
          ['en', 'EN', 'English'],
        ].map(([id, label, full]) => (
          <button key={id} type="button" lang={id === 'en' ? 'en' : 'pt-BR'} aria-pressed={lang === id} title={full} onClick={() => setLang(id)}>
            {label}
          </button>
        ))}
      </div>
      <div className="seg" role="group" aria-label={t('prefs.theme')}>
        {themes.map(([id, Icon, label]) => (
          <button key={id} type="button" aria-pressed={theme === id} title={label} aria-label={label} onClick={() => setTheme(id)}>
            <Icon />
          </button>
        ))}
      </div>
    </div>
  );
}
