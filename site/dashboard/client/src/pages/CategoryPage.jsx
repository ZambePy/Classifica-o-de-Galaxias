import { useEffect, useRef, useState } from 'react';
import { Navigate, NavLink, useParams, useSearchParams } from 'react-router-dom';
import { api } from '../api.js';
import { useData } from '../components/DataContext.jsx';
import { ClassificationCard, EmptyState } from '../components/Classification.jsx';
import { IconInfo, IconCube } from '../components/Icons.jsx';
import Viewer3D from '../components/Viewer3D.jsx';
import { usePrefs } from '../prefs.jsx';
import { getCategory, getLegacyType } from '../data/taxonomy.js';
import NotFound from './NotFound.jsx';

function SubTabs({ category, byType }) {
  const { t, tx } = usePrefs();
  const ref = useRef(null);
  const { tipo } = useParams();
  // no celular as sub-abas rolam na horizontal: mantém a ativa visível
  useEffect(() => {
    ref.current?.querySelector('.subtab.active')?.scrollIntoView({ block: 'nearest', inline: 'center' });
  }, [tipo]);
  const accent = `var(--cat-${category.id})`;
  const total = category.types.reduce((s, ty) => s + (byType[`${category.id}/${ty.id}`] ?? 0), 0);
  return (
    <div className="subtabs-wrap" ref={ref}>
      <nav className="subtabs" aria-label={t('cat.typesOf', { cat: tx(category, 'name').toLowerCase() })} style={{ '--accent': accent }}>
        <NavLink to={`/${category.id}`} end className={({ isActive }) => `subtab${isActive ? ' active' : ''}`}>
          <span className="name">{t('cat.all')}</span>
          <span className="n">{total}</span>
        </NavLink>
        {category.types.map((ty) => (
          <NavLink key={ty.id} to={`/${category.id}/${ty.id}`} className={({ isActive }) => `subtab${isActive ? ' active' : ''}`}>
            <span className="name">{tx(ty, 'name')}</span>
            <span className="n">{byType[`${category.id}/${ty.id}`] ?? 0}</span>
          </NavLink>
        ))}
      </nav>
    </div>
  );
}

function PlainList({ items }) {
  return (
    <ul className="plain-list">
      {items.map((c) => (
        <li key={c}>{c}</li>
      ))}
    </ul>
  );
}

function ExampleList({ type, onOpen }) {
  const { t, tx } = usePrefs();
  const names = tx(type, 'examples');
  return (
    <ul className="plain-list examples-3d">
      {names.map((name, i) => {
        const id = type.exampleIds?.[i];
        return (
          <li key={name}>
            {id ? (
              <button type="button" className="link-3d" onClick={() => onOpen(id)} aria-label={t('v3d.open', { name })}>
                {name}
                <IconCube />
              </button>
            ) : (
              name
            )}
          </li>
        );
      })}
    </ul>
  );
}

function TypeSheet({ type, onOpen }) {
  const { t, tx } = usePrefs();
  const warning = tx(type, 'warning');
  return (
    <section className="panel type-sheet" aria-label={t('cat.sheet', { name: tx(type, 'name') })}>
      <div style={{ display: 'grid', gap: 12, alignContent: 'start' }}>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
          <h2>{tx(type, 'name')}</h2>
          {!type.trainable && <span className="pill warn">{t('cat.informative')}</span>}
        </div>
        <p style={{ color: 'var(--text-2)' }}>{tx(type, 'description')}</p>
        {warning && (
          <div className="notice">
            <IconInfo />
            <span>{warning}</span>
          </div>
        )}
      </div>
      <dl>
        <div>
          <dt className="eyebrow">{t('cat.cues')}</dt>
          <dd>
            <PlainList items={tx(type, 'visualCues')} />
          </dd>
        </div>
        <div>
          <dt className="eyebrow">{t('cat.examples')}</dt>
          <dd>
            <ExampleList type={type} onOpen={onOpen} />
            <p className="examples-hint">{t('cat.examplesHint')}</p>
          </dd>
        </div>
      </dl>
    </section>
  );
}

export default function CategoryPage({ categoryId }) {
  const { tipo } = useParams();
  const { stats } = useData();
  const { t, tx } = usePrefs();
  const category = getCategory(categoryId);
  const type = tipo ? category?.types.find((ty) => ty.id === tipo) : null;
  const legacy = tipo && !type ? getLegacyType(categoryId, tipo) : null;
  // filtro dentro da aba (ex.: Espirais > Barradas) fica no endereço: ?filtro=barradas
  const [params, setParams] = useSearchParams();
  const wanted = params.get('filtro') ?? '';
  const sub = type?.subfilters?.some((s) => s.id === wanted) ? wanted : '';
  const setSub = (id) => setParams(id ? { filtro: id } : {}, { replace: true });
  const [viewer, setViewer] = useState(null);
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    api.classifications({ category: categoryId, type: type?.id, subfilter: sub }).then((r) => {
      if (!alive) return;
      setItems(r.items ?? []);
      setLoading(false);
    });
    return () => {
      alive = false;
    };
  }, [categoryId, type?.id, sub]);

  // abas antigas que foram juntadas (ex.: /galaxias/espirais-barradas) levam à aba nova já filtrada
  if (legacy) return <Navigate to={`/${categoryId}/${legacy.type}?filtro=${legacy.subfilter}`} replace />;
  if (!category || (tipo && !type)) return <NotFound />;

  const accent = `var(--cat-${category.id})`;
  const singular = tx(category, 'singular');
  const typeName = type ? tx(type, 'name').toLowerCase() : '';

  return (
    <>
      <header className="page-head">
        <span className="eyebrow">{t('cat.eyebrow')}</span>
        <h1 style={{ color: accent }}>{tx(category, 'name')}</h1>
        <p>{tx(category, 'description').split('. ')[0].replace(/\.?$/, '.')}</p>
      </header>

      <SubTabs category={category} byType={stats.byType ?? {}} />

      {type && <TypeSheet type={type} onOpen={setViewer} />}
      {viewer && <Viewer3D id={viewer} onClose={() => setViewer(null)} />}

      {type?.subfilters && (
        <div className="chips" role="group" aria-label={t('cat.filter')}>
          <button type="button" className={`btn ${sub === '' ? '' : 'ghost'}`} aria-pressed={sub === ''} onClick={() => setSub('')}>
            {t('cat.filterAll')}
          </button>
          {type.subfilters.map((s) => (
            <button key={s.id} type="button" className={`btn ${sub === s.id ? '' : 'ghost'}`} aria-pressed={sub === s.id} onClick={() => setSub(s.id)}>
              {tx(s, 'name')}
            </button>
          ))}
        </div>
      )}

      <section aria-live="polite" aria-busy={loading}>
        {items.length > 0 ? (
          <div className="cards">
            {items.map((it) => (
              <ClassificationCard key={it.id} item={it} typeName={type ? tx(type, 'name') : singular.charAt(0).toUpperCase() + singular.slice(1)} />
            ))}
          </div>
        ) : (
          <EmptyState
            accent={accent}
            title={
              loading
                ? t('cat.loading')
                : type
                  ? t('cat.emptyTitleType', { singular, type: typeName })
                  : t('cat.emptyTitleAll', { singular })
            }
          >
            {type ? t('cat.emptyBodyType', { type: typeName }) : t('cat.emptyBodyAll')}
          </EmptyState>
        )}
      </section>
    </>
  );
}
