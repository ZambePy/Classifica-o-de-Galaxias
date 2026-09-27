import { Link, useLocation } from 'react-router-dom';
import { usePrefs } from '../prefs.jsx';
import GalaxySpin from '../components/GalaxySpin.jsx';

export default function NotFound() {
  const { pathname } = useLocation();
  const { t } = usePrefs();
  return (
    <section className="nf" aria-labelledby="nf-title">
      <span className="eyebrow">{t('nf.eyebrow')}</span>
      <div className="nf-galaxy">
        <GalaxySpin size={320} arms={3} />
        <div className="code" aria-hidden="true">
          404
        </div>
      </div>
      <h1 id="nf-title">{t('nf.title')}</h1>
      <p>
        {t('nf.body1')} <strong>{pathname}</strong>{t('nf.body2')}
      </p>
      <div className="actions">
        <Link to="/" className="btn">
          {t('nf.home')}
        </Link>
        <Link to="/galaxias" className="btn ghost">
          {t('nf.gal')}
        </Link>
        <Link to="/nebulosas" className="btn ghost">
          {t('nf.neb')}
        </Link>
      </div>
    </section>
  );
}
