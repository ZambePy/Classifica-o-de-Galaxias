import { EmptyIllustration } from './Icons.jsx';
import { usePrefs } from '../prefs.jsx';
import { taxonomy } from '../data/taxonomy.js';

// rótulo técnico do modelo -> tipo (para mostrar nomes simples na tela)
const LABEL_TO_TYPE = new Map();
for (const c of taxonomy.categories) for (const ty of c.types) for (const l of ty.modelLabels) LABEL_TO_TYPE.set(l.toLowerCase(), ty);

const pct = (v) => `${Math.round(v * 100)}%`;

/** Cartão de um objeto já classificado pelo modelo (usado quando houver dados). */
export function ClassificationCard({ item, typeName }) {
  const { t, tx } = usePrefs();
  const top = item.probabilities
    ? Object.entries(item.probabilities)
        .sort((a, b) => b[1] - a[1])
        .slice(1, 3)
        .map(([label, p]) => {
          const ty = LABEL_TO_TYPE.get(label.toLowerCase());
          return ty ? `${tx(ty, 'name')} (${pct(p)})` : null;
        })
        .filter(Boolean)
    : [];
  return (
    <article className="card">
      <img className="thumb" src={item.imageUrl} alt={item.objectName ?? t('card.alt', { type: typeName })} loading="lazy" />
      <div className="body">
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8, alignItems: 'baseline' }}>
          <strong>{item.objectName ?? t('card.noName')}</strong>
          <span className="mono" style={{ color: 'var(--mint)' }}>
            {pct(item.confidence)}
          </span>
        </div>
        <div className="bar" aria-label={t('card.conf', { v: pct(item.confidence) })}>
          <span style={{ width: pct(item.confidence) }} />
        </div>
        {top.length > 0 && (
          <p style={{ fontSize: '0.8rem', color: 'var(--text-2)' }}>
            {t('card.also')}: {top.join(', ')}
          </p>
        )}
        <p className="credit">
          {t('card.credit')}: {item.credit}
        </p>
      </div>
    </article>
  );
}

export function EmptyState({ title, children, accent }) {
  return (
    <div className="empty">
      <EmptyIllustration accent={accent} />
      <h3>{title}</h3>
      <p>{children}</p>
    </div>
  );
}
