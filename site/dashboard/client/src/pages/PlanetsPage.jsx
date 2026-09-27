import { useState } from 'react';
import { usePrefs } from '../prefs.jsx';
import { OBJECTS, PLANET_IDS } from '../three/objects3d.js';
import { PLANET_TEXTURES } from '../three/planetTextures.js';
import { IconCube, IconInfo } from '../components/Icons.jsx';
import Viewer3D from '../components/Viewer3D.jsx';

// Aba Planetas (sessão 8): os 8 planetas do Sistema Solar, do mais perto ao mais longe do Sol.
// Cada cartão abre o modelo 3D (three/planets.js). A classificação pela câmera do celular fica para depois.

// tamanho da bolinha no cartão: só dá uma ideia de "maior / menor" (fora de escala de verdade)
const BALL = { mercurio: 62, venus: 84, terra: 86, marte: 70, jupiter: 118, saturno: 104, urano: 96, netuno: 94 };
const GLOW = {
  mercurio: 'rgba(200,200,210,0.18)',
  venus: 'rgba(255,214,150,0.35)',
  terra: 'rgba(95,168,255,0.45)',
  marte: 'rgba(255,140,100,0.3)',
  jupiter: 'rgba(255,220,170,0.3)',
  saturno: 'rgba(255,230,180,0.3)',
  urano: 'rgba(150,240,255,0.4)',
  netuno: 'rgba(90,140,255,0.45)',
};

function PlanetBall({ id }) {
  const obj = OBJECTS[id];
  const tex = PLANET_TEXTURES[obj.planet.tex];
  const rings = obj.planet.rings;
  return (
    <span
      className={`planet-ball-wrap${rings ? ` has-rings rings-${rings}` : ''}`}
      style={{ '--size': `${BALL[id]}px`, '--pglow': GLOW[id] }}
      aria-hidden="true"
    >
      {rings && <span className="planet-ring back" />}
      <span className="planet-ball" style={{ backgroundImage: `url(${tex})` }} />
      {rings && <span className="planet-ring front" />}
    </span>
  );
}

export default function PlanetsPage() {
  const { t, lang } = usePrefs();
  const [viewer, setViewer] = useState(null);

  return (
    <>
      <header className="page-head">
        <span className="eyebrow">{t('pl.eyebrow')}</span>
        <h1>{t('nav.planets')}</h1>
        <p>{t('pl.lead')}</p>
      </header>

      <ul className="planet-groups">
        {['rocky', 'gas', 'ice'].map((g) => (
          <li key={g}>
            <span className={`planet-dot g-${g}`} />
            <span className="planet-group-text">
              <strong>{t(`pl.group.${g}`)}</strong>
              <span>{t(`pl.groupInfo.${g}`)}</span>
            </span>
          </li>
        ))}
      </ul>

      <ol className="planet-grid">
        {PLANET_IDS.map((id, i) => {
          const obj = OBJECTS[id];
          const info = obj[lang] ?? obj.pt;
          return (
            <li key={id}>
              <button type="button" className="panel planet-card" onClick={() => setViewer(id)} aria-label={`${info.name}: ${t('pl.open')}`}>
                <span className="planet-order">{i + 1}</span>
                <span className="planet-stage">
                  <PlanetBall id={id} />
                </span>
                <span className="planet-name">{info.name}</span>
                <span className={`planet-group g-${obj.group}`}>{t(`pl.group.${obj.group}`)}</span>
                <span className="planet-short">{info.short}</span>
                <span className="planet-open">
                  <IconCube /> {t('pl.open')}
                </span>
              </button>
            </li>
          );
        })}
      </ol>

      <div className="notice" style={{ marginTop: 24 }}>
        <IconInfo />
        <span>{t('pl.camera')}</span>
      </div>

      {viewer && <Viewer3D id={viewer} onClose={() => setViewer(null)} />}
    </>
  );
}
