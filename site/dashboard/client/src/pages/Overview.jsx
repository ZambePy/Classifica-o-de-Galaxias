import { Link } from 'react-router-dom';
import { useData } from '../components/DataContext.jsx';
import { usePrefs } from '../prefs.jsx';
import { taxonomy } from '../data/taxonomy.js';
import GalaxySpin from '../components/GalaxySpin.jsx';
import CountUp from '../components/CountUp.jsx';
import { IconGalaxy, IconModel, IconOverview, IconSpark, IconUpload } from '../components/Icons.jsx';

function Distribution({ category, byType }) {
  const { t, tx, nf } = usePrefs();
  const accent = `var(--cat-${category.id}-strong)`;
  const values = category.types.map((ty) => byType[`${category.id}/${ty.id}`] ?? 0);
  const max = Math.max(...values, 0);
  return (
    <section className="panel" aria-labelledby={`dist-${category.id}`}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 12, marginBottom: 16 }}>
        <h2 id={`dist-${category.id}`}>{t('ov.byType', { cat: tx(category, 'name') })}</h2>
      </div>
      <div className="dist">
        {category.types.map((ty, i) => (
          <Link key={ty.id} to={`/${category.id}/${ty.id}`} className="dist-row" title={t('ov.open', { name: tx(ty, 'name') })}>
            <span>{tx(ty, 'name')}</span>
            <span className="track">
              <span className="fill" style={{ width: max ? `${(values[i] / max) * 100}%` : 0, background: accent }} />
            </span>
            <span className="val">{nf.format(values[i])}</span>
          </Link>
        ))}
      </div>
      {max === 0 && (
        <p style={{ marginTop: 14, color: 'var(--muted)', fontSize: '0.85rem' }}>{t('ov.noData')}</p>
      )}
    </section>
  );
}

function Step({ title, icon: Icon, children, pending }) {
  const { t } = usePrefs();
  return (
    <div className={`step${pending ? ' pending' : ''}`}>
      <span className="step-icon">
        <Icon />
      </span>
      <span className="eyebrow" data-label={t('ov.step')} />
      <strong>{title}</strong>
      {children}
    </div>
  );
}

export default function Overview() {
  const { stats, health } = useData();
  const { t, nf } = usePrefs();
  const connected = Boolean(health?.model?.connected);
  const [gal, neb] = taxonomy.categories;

  const kpis = [
    { label: t('ov.kpi.total'), value: <CountUp value={stats.total} format={nf.format} />, hint: t('ov.kpi.totalHint') },
    {
      label: t('nav.galaxies'),
      value: <CountUp value={stats.byCategory?.galaxias ?? 0} format={nf.format} />,
      hint: t('ov.kpi.galHint', { n: gal.types.length }),
    },
    {
      label: t('nav.nebulae'),
      value: <CountUp value={stats.byCategory?.nebulosas ?? 0} format={nf.format} />,
      hint: t('ov.kpi.nebHint', { n: neb.types.length }),
    },
    {
      label: t('ov.kpi.conf'),
      value:
        stats.meanConfidence == null ? '—' : <CountUp value={Math.round(stats.meanConfidence * 100)} format={(n) => `${n}%`} />,
      hint: stats.meanConfidence == null ? t('ov.kpi.confEmpty') : t('ov.kpi.confHint'),
    },
  ];

  return (
    <>
      <header className="hero">
        <div className="page-head">
          <span className="eyebrow">{t('ov.eyebrow')}</span>
          <h1>{t('ov.title')}</h1>
          <p>{t('ov.lead')}</p>
          <div className="chips" style={{ marginTop: 4 }}>
            <span className={`pill ${connected ? 'aqua' : 'warn'}`}>
              <span className={`dot${connected ? ' on' : ''}`} /> {connected ? t('model.connected') : t('model.waiting')}
            </span>
          </div>
        </div>
        <GalaxySpin size={230} />
      </header>

      <section className="grid kpis" aria-label={t('ov.kpis')}>
        {kpis.map((k) => (
          <div key={k.label} className="kpi">
            <span className="eyebrow">{k.label}</span>
            <span className="value">{k.value}</span>
            <span className="hint">{k.hint}</span>
          </div>
        ))}
      </section>

      <div className="grid two">
        <Distribution category={gal} byType={stats.byType ?? {}} />
        <Distribution category={neb} byType={stats.byType ?? {}} />
      </div>

      <section className="panel" aria-labelledby="pipeline-title">
        <h2 id="pipeline-title" style={{ marginBottom: 16 }}>
          {t('ov.flow')}
        </h2>
        <div className="pipeline">
          <Step title={t('ov.s1.t')} icon={IconUpload}>
            <span>{t('ov.s1.d')}</span>
          </Step>
          <Step title={t('ov.s2.t')} icon={IconOverview}>
            <span>{t('ov.s2.d')}</span>
          </Step>
          <Step title={t('ov.s3.t')} icon={IconModel} pending={!connected}>
            <span className={`pill ${connected ? 'aqua' : 'warn'}`} style={{ justifySelf: 'start' }}>
              {connected ? (health.model.demo ? t('ov.s3.demo') : t('ov.s3.on')) : t('ov.s3.pending')}
            </span>
            <span>{t('ov.s3.d')}</span>
          </Step>
          <Step title={t('ov.s4.t')} icon={IconGalaxy}>
            <span>{t('ov.s4.d')}</span>
          </Step>
          <Step title={t('ov.s5.t')} icon={IconSpark}>
            <span>{t('ov.s5.d')}</span>
          </Step>
        </div>
      </section>

    </>
  );
}
