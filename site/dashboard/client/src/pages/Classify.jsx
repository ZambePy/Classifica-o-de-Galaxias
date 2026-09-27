import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { useData } from '../components/DataContext.jsx';
import { IconInfo, IconUpload } from '../components/Icons.jsx';
import { usePrefs } from '../prefs.jsx';
import { taxonomy, getCategory, getType } from '../data/taxonomy.js';

const MAX_MB = 10;
const ACCEPT = ['image/jpeg', 'image/png', 'image/webp'];
// Status do servidor Node (server/lib/classificador.js) -> mensagem da tela
const ERRORS = { 413: 'cl.err.413', 415: 'cl.err.422', 422: 'cl.err.422', 429: 'cl.err.429', 502: 'cl.err.502', 503: 'cl.err.503' };

// Rótulo do subtipo (ex.: "spiral") -> aba do site, para mostrar os nomes do site e nunca o label_pt da API.
const LABEL_TO_TYPE = new Map();
for (const c of taxonomy.categories) for (const ty of c.types) for (const l of ty.modelLabels ?? []) LABEL_TO_TYPE.set(l, ty);

function usePercent() {
  const { lang } = usePrefs();
  const nf = new Intl.NumberFormat(lang === 'en' ? 'en-US' : 'pt-BR', { maximumFractionDigits: 1 });
  return (v) => `${nf.format(v * 100)}%`;
}

/** Nome do site para um rótulo da API (nível 1: galaxy/nebula/other; nível 2: aba). */
function useLabelName() {
  const { t, tx } = usePrefs();
  return (label) => {
    if (['galaxy', 'nebula', 'other'].includes(label)) return t(`cl.label.${label}`);
    const ty = LABEL_TO_TYPE.get(label);
    return ty ? tx(ty, 'name') : label;
  };
}

function Scores({ step }) {
  const pct = usePercent();
  const name = useLabelName();
  return (
    <ul className="cl-scores">
      {step.scores.map((s) => (
        <li key={s.label} className={s.label === step.predicted ? 'top' : undefined}>
          <span className="cl-score-name">{name(s.label)}</span>
          <span className="bar" aria-hidden="true">
            <span style={{ width: `${Math.max(1, s.probability * 100)}%` }} />
          </span>
          <span className="cl-score-value">{pct(s.probability)}</span>
        </li>
      ))}
    </ul>
  );
}

function Result({ result }) {
  const { t, tx } = usePrefs();
  const pct = usePercent();
  const name = useLabelName();
  const category = getCategory(result.category);
  const type = getType(result.category, result.type);
  const isOther = result.kind === 'outro';

  let title;
  if (isOther) title = t('cl.other');
  else if (type) title = tx(type, 'name');
  else title = name(result.kind === 'galaxia' ? 'galaxy' : 'nebula');

  const link = isOther ? null : type ? `/${result.category}/${result.type}` : category ? `/${result.category}` : null;
  const linkName = type ? tx(type, 'name') : category ? tx(category, 'name') : '';

  return (
    <section className="panel cl-result" role="status" aria-live="polite">
      {result.outOfDomain && (
        <div className="notice strong">
          <IconInfo />
          <strong>{t('cl.ood')}</strong>
        </div>
      )}
      {result.mock && (
        <p className="cl-seal">
          <IconInfo />
          <span>
            <strong>{t('cl.simulated')}</strong> {t('cl.simulatedHint')}
          </span>
        </p>
      )}

      <div className="cl-main">
        <span className="eyebrow">{t('cl.result')}</span>
        <h2>{title}</h2>
        <p className="cl-conf">
          <span>{t('cl.sure')}</span> <strong>{pct(result.confidence)}</strong>
          {result.ambiguous && <span className="cl-doubt">{t('cl.ambiguous')}</span>}
        </p>
        {isOther && <p className="cl-note">{t('cl.otherHint')}</p>}
        {!isOther && !type && <p className="cl-note">{t(`cl.noSubtype.${result.kind}`)}</p>}
        {link && <Link to={link}>{t('cl.seeTab', { name: linkName })}</Link>}
      </div>

      <div className="cl-steps">
        <h3>{t('cl.steps')}</h3>
        <ol>
          {result.steps.map((step) => (
            <li key={step.level}>
              <h4>{t(step.level === 'object' ? 'cl.step.object' : 'cl.step.type')}</h4>
              <Scores step={step} />
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}

export default function Classify() {
  const { health } = useData();
  const { t } = usePrefs();
  const connected = Boolean(health?.model?.connected);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState('');
  const [error, setError] = useState('');
  const [rightsOk, setRightsOk] = useState(false);
  const [sending, setSending] = useState(false);
  const [result, setResult] = useState(null);

  useEffect(() => () => preview && URL.revokeObjectURL(preview), [preview]);

  const onPick = (e) => {
    const f = e.target.files?.[0];
    setError('');
    if (!f) return;
    if (!ACCEPT.includes(f.type)) return setError(t('cl.err.format'));
    if (f.size > MAX_MB * 1024 * 1024) return setError(t('cl.err.size', { mb: MAX_MB }));
    setFile(f);
    setResult(null);
    setPreview(URL.createObjectURL(f));
  };

  const submit = async (e) => {
    e.preventDefault();
    if (!file || !rightsOk) return;
    setSending(true);
    setError('');
    setResult(null);
    try {
      // Só a foto vai para o servidor; ela é analisada na memória e apagada logo depois.
      const form = new FormData();
      form.append('file', file);
      const res = await fetch('/api/classify', { method: 'POST', body: form });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(t(ERRORS[res.status] ?? 'cl.err.generic'));
      setResult(body);
    } catch (err) {
      setError(err.message || t('cl.err.generic'));
    } finally {
      setSending(false);
    }
  };

  return (
    <>
      <header className="page-head">
        <span className="eyebrow">{t('cl.eyebrow')}</span>
        <h1>{t('cl.title')}</h1>
        <p>{t('cl.lead')}</p>
      </header>

      {!connected && (
        <div className="notice">
          <IconInfo />
          <span>{t('cl.offline')}</span>
        </div>
      )}

      <div className="grid two" style={{ alignItems: 'start' }}>
        <label className="dropzone" htmlFor="arquivo-imagem" style={{ cursor: 'pointer', position: 'relative' }}>
          {preview ? (
            <img src={preview} alt={t('cl.previewAlt')} style={{ maxHeight: 260, borderRadius: 10 }} />
          ) : (
            <IconUpload style={{ width: 40, height: 40, color: 'var(--sky)' }} />
          )}
          <strong style={{ color: 'var(--text)' }}>{file ? file.name : t('cl.pick')}</strong>
          <span style={{ fontSize: '0.84rem' }}>{t('cl.formats', { mb: MAX_MB })}</span>
          <input
            id="arquivo-imagem"
            type="file"
            accept={ACCEPT.join(',')}
            onChange={onPick}
            style={{ position: 'absolute', opacity: 0, width: 1, height: 1 }}
          />
        </label>

        <form className="panel" style={{ display: 'grid', gap: 14 }} onSubmit={submit}>
          <h2>{t('cl.before')}</h2>
          <label className="check" htmlFor="direitos">
            <input id="direitos" type="checkbox" checked={rightsOk} onChange={(e) => setRightsOk(e.target.checked)} />
            <span>
              {t('cl.rights')} (<Link to="/termos#envio">{t('cl.rightsLink')}</Link>)
            </span>
          </label>
          {error && (
            <p role="alert" style={{ color: 'var(--lavender)' }}>
              {error}
            </p>
          )}
          <button type="submit" className="btn" disabled={!connected || !file || !rightsOk || sending}>
            {!connected ? t('cl.disabled') : sending ? t('cl.sending') : t('cl.submit')}
          </button>
          {health?.model?.demo && (
            <div className="notice">
              <IconInfo />
              <span>{t('cl.demo')}</span>
            </div>
          )}
          <p style={{ fontSize: '0.8rem', color: 'var(--muted)' }}>{t('cl.deleted')}</p>
        </form>
      </div>

      {result && <Result result={result} />}
    </>
  );
}
