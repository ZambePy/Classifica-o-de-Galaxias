import { usePrefs } from '../prefs.jsx';
import GalaxySpin from '../components/GalaxySpin.jsx';

// Aba reservada para um recurso futuro (Planetas, Estrelas). Conteúdo ainda não definido.
export default function ComingSoon({ titleKey, icon: Icon }) {
  const { t } = usePrefs();
  return (
    <>
      <header className="page-head">
        <span className="eyebrow">{t('cat.eyebrow')}</span>
        <h1>{t(titleKey)}</h1>
      </header>
      <section className="empty soon">
        <div className="soon-art" aria-hidden="true">
          <GalaxySpin size={150} arms={2} />
          <Icon className="soon-icon" />
        </div>
        <h3>{t('soon.title')}</h3>
        <p>{t('soon.body')}</p>
      </section>
    </>
  );
}
