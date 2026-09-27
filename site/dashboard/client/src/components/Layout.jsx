import { useEffect, useState } from 'react';
import { NavLink, Link, Outlet, useLocation } from 'react-router-dom';
import logo from '../assets/logo.svg';
import { useData } from './DataContext.jsx';
import { IconOverview, IconGalaxy, IconNebula, IconUpload, IconModel, IconInfo, IconChat, IconPlanet, IconStar } from './Icons.jsx';
import Prefs from './Prefs.jsx';
import Starfield from './Starfield.jsx';
import IntroSplash from './IntroSplash.jsx';
import { usePrefs } from '../prefs.jsx';
import { siteName } from '../data/site.js';

const COOKIE_NOTE_KEY = 'pg-aviso-cookies-v1';

function readNoteDismissed() {
  try {
    return window.localStorage.getItem(COOKIE_NOTE_KEY) === '1';
  } catch {
    return false;
  }
}

function CookieNote() {
  const { t } = usePrefs();
  const [hidden, setHidden] = useState(readNoteDismissed);
  if (hidden) return null;
  const dismiss = () => {
    try {
      window.localStorage.setItem(COOKIE_NOTE_KEY, '1');
    } catch {
      /* navegador sem armazenamento: o aviso some só nesta visita */
    }
    setHidden(true);
  };
  return (
    <div className="cookie-note" role="region" aria-label={t('cookie.region')}>
      <strong style={{ color: 'var(--text)' }}>{t('cookie.title')}</strong>
      <p>
        {t('cookie.body')} <Link to="/privacidade#cookies">{t('cookie.more')}</Link>
      </p>
      <div>
        <button type="button" className="btn ghost" onClick={dismiss}>
          {t('cookie.ok')}
        </button>
      </div>
    </div>
  );
}

export default function Layout() {
  const { stats, health } = useData();
  const { t, lang } = usePrefs();
  const { pathname } = useLocation();
  const connected = Boolean(health?.model?.connected);

  // volta ao topo ao trocar de página
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [pathname]);

  const count = (cat) => stats.byCategory?.[cat] ?? 0;

  return (
    <div className="shell">
      <Starfield />
      <aside className="sidebar">
        <Link to="/" className="brand" aria-label={t('brand.home')}>
          <img src={logo} alt="" />
          <span className="brand-text">
            <small>{t('brand.top')}</small>
            <strong>{t('brand.main')}</strong>
          </span>
        </Link>

        <nav className="nav" aria-label={t('nav.main')}>
          <span className="eyebrow nav-label">{t('nav.panel')}</span>
          <NavLink to="/" end>
            <IconOverview /> {t('nav.overview')}
          </NavLink>
          <NavLink to="/galaxias">
            <IconGalaxy /> {t('nav.galaxies')} <span className="count">{count('galaxias')}</span>
          </NavLink>
          <NavLink to="/nebulosas">
            <IconNebula /> {t('nav.nebulae')} <span className="count">{count('nebulosas')}</span>
          </NavLink>
          <NavLink to="/planetas">
            <IconPlanet /> {t('nav.planets')}
          </NavLink>
          <NavLink to="/estrelas">
            <IconStar /> {t('nav.stars')}
          </NavLink>
          <span className="eyebrow nav-label">{t('nav.tools')}</span>
          <NavLink to="/classificar">
            <IconUpload /> {t('nav.classify')}
          </NavLink>
          <NavLink to="/chat">
            <IconChat /> {t('nav.chat')}
          </NavLink>
          <NavLink to="/modelo">
            <IconModel /> {t('nav.model')}
          </NavLink>
        </nav>

        <div className="sidebar-foot">
          <div className="model-status" role="status">
            <span className="row">
              <span className={`dot${connected ? ' on' : ''}`} />
              {connected ? t('model.connected') : t('model.waiting')}
            </span>
          </div>
        </div>
      </aside>

      <div className="main">
        <div className="ai-banner">
          <IconInfo />
          <span>
            {t('ai.banner')} <Link to="/modelo">{t('ai.how')}</Link>
          </span>
          <Prefs />
        </div>
        <main className="content" id="conteudo">
          <Outlet />
        </main>
        <footer className="site-footer">
          <nav className="footer-links" aria-label={t('nav.legal')}>
            <Link to="/termos">{t('nav.terms')}</Link>
            <Link to="/conformidade">{t('nav.compliance')}</Link>
            <Link to="/privacidade">{t('nav.privacy')}</Link>
            <Link to="/creditos">{t('nav.credits')}</Link>
          </nav>
          <span>
            {t('footer.left', { name: siteName(lang) })} · {t('footer.noAff')}
          </span>
        </footer>
      </div>
      <CookieNote />
      <IntroSplash />
    </div>
  );
}
