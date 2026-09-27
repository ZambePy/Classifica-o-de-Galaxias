import { usePrefs } from '../prefs.jsx';
import { IconGalaxy, IconInfo, IconModel, IconSpark } from '../components/Icons.jsx';

// Página "Como funciona": explica a IA em linguagem simples.
// Detalhes técnicos (arquitetura, dataset, mapeamento de rótulos) ficam no README do projeto, para a equipe.
// Números copiados de docs/results.md do classificador (seções "Resumo", "Ablação por oclusão",
// "Nível 1", "Nível 2", "Nível 3" e "Três métodos de fora-de-domínio"). Não invente: atualize junto com o docs.
const RESULTS_URL = 'https://github.com/ZambePy/Classifica-o-de-Galaxias/blob/main/docs/results.md';
const RESULTS = [
  // [texto, acurácia no teste, fotos de teste]
  ['m.row.object', 0.9656, 1048],
  ['m.row.galaxy', 0.9459, 5656],
  ['m.row.nebula', 0.854, 411],
  ['m.row.final', 0.8941, 1048],
];

export default function ModelCard() {
  const { t, lang } = usePrefs();
  const locale = lang === 'en' ? 'en-US' : 'pt-BR';
  const pct = new Intl.NumberFormat(locale, { style: 'percent', minimumFractionDigits: 1, maximumFractionDigits: 1 });
  const int = new Intl.NumberFormat(locale);
  const qa = [
    [IconModel, 'm.q1', 'm.a1'],
    [IconGalaxy, 'm.q2', 'm.a2'],
    [IconSpark, 'm.q3', 'm.a3'],
  ];
  return (
    <>
      <header className="page-head">
        <span className="eyebrow">{t('m.eyebrow')}</span>
        <h1>{t('m.title')}</h1>
        <p>{t('m.lead')}</p>
      </header>

      <section className="pipeline" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))' }}>
        {qa.map(([Icon, q, a]) => (
          <div key={q} className="step">
            <span className="step-icon">
              <Icon />
            </span>
            <strong>{t(q)}</strong>
            <span>{t(a)}</span>
          </div>
        ))}
      </section>

      <section className="panel" style={{ display: 'grid', gap: 12 }}>
        <h2>{t('m.numbers')}</h2>
        <p style={{ color: 'var(--text-2)' }}>{t('m.numbersLead')}</p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>{t('m.col.step')}</th>
                <th style={{ textAlign: 'right' }}>{t('m.col.acc')}</th>
                <th style={{ textAlign: 'right' }}>{t('m.col.n')}</th>
              </tr>
            </thead>
            <tbody>
              {RESULTS.map(([key, acc, n]) => (
                <tr key={key}>
                  <td style={{ color: 'var(--text)' }}>{t(key)}</td>
                  <td style={{ textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{pct.format(acc)}</td>
                  <td style={{ textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{int.format(n)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <h3>{t('m.weak')}</h3>
        <ul className="plain-list">
          {['m.w1', 'm.w2', 'm.w3', 'm.w4', 'm.w5'].map((k) => (
            <li key={k}>{t(k)}</li>
          ))}
        </ul>
        <p style={{ fontSize: '0.8rem', color: 'var(--muted)' }}>
          {t('m.source')}{' '}
          <a href={RESULTS_URL} target="_blank" rel="noreferrer">
            {t('m.sourceLink')}
          </a>
        </p>
      </section>

      <section className="panel" style={{ display: 'grid', gap: 12 }}>
        <h2>{t('m.limits')}</h2>
        <ul className="plain-list">
          {['m.l1', 'm.l2', 'm.l3', 'm.l4', 'm.l5'].map((k) => (
            <li key={k}>{t(k)}</li>
          ))}
        </ul>
      </section>

      <div className="notice">
        <IconInfo />
        <span>
          <strong>{t('m.status')}:</strong> {t('m.statusV')}
        </span>
      </div>
    </>
  );
}
