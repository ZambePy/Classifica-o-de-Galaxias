import { site, siteName } from '../data/site.js';
import { usePrefs } from '../prefs.jsx';

// [fonte, pode usar? PT, EN, crédito PT, EN, site]
const SOURCES = [
  ['Galaxy10 DECaLS', 'Sim, dando o crédito', 'Yes, with credit', 'Autores Leung e Bovy, com Galaxy Zoo e Legacy Surveys', 'Authors Leung and Bovy, with Galaxy Zoo and Legacy Surveys', 'https://zenodo.org/records/10845026'],
  ['Legacy Surveys', 'Sim, dando o crédito', 'Yes, with credit', 'Legacy Surveys / D. Lang (Perimeter Institute)', 'Legacy Surveys / D. Lang (Perimeter Institute)', 'https://www.legacysurvey.org/acknowledgment/'],
  ['Galaxy Zoo', 'Sim, dando o crédito', 'Yes, with credit', 'Galaxy Zoo e seus voluntários', 'Galaxy Zoo and its volunteers', 'https://data.galaxyzoo.org/'],
  ['ESA/Hubble', 'Sim, dando o crédito', 'Yes, with credit', 'Texto de crédito de cada foto (ex.: ESA/Hubble & NASA)', 'Each photo credit line (e.g. ESA/Hubble & NASA)', 'https://esahubble.org/copyright/'],
  ['ESA/Webb', 'Sim, dando o crédito', 'Yes, with credit', 'Texto de crédito de cada foto (ex.: ESA/Webb, NASA & CSA)', 'Each photo credit line (e.g. ESA/Webb, NASA & CSA)', 'https://esawebb.org/copyright/'],
  ['ESO', 'Sim, dando o crédito', 'Yes, with credit', 'ESO, junto de cada foto', 'ESO, next to each photo', 'https://www.eso.org/public/copyright/'],
  ['NOIRLab', 'Sim, dando o crédito', 'Yes, with credit', 'Texto de crédito de cada foto', 'Each photo credit line', 'https://noirlab.edu/public/copyright/'],
  ['NASA', 'Sim, sem usar a marca da NASA', 'Yes, without using the NASA logo', 'NASA como fonte da imagem', 'NASA as the image source', 'https://www.nasa.gov/nasa-brand-center/images-and-media/'],
  ['Sloan Digital Sky Survey', 'Sim, dando o crédito', 'Yes, with credit', 'Sloan Digital Sky Survey (SDSS)', 'Sloan Digital Sky Survey (SDSS)', 'https://www.sdss.org/collaboration/image-use-policy/'],
  // sessão 8: aba Planetas
  ['Natural Earth', 'Sim, domínio público', 'Yes, public domain', 'Contorno dos continentes da Terra em 3D', 'Outline of the continents on the 3D Earth', 'https://www.naturalearthdata.com/about/terms-of-use/'],
  ['NASA Planetary Fact Sheet', 'Sim, dados públicos', 'Yes, public data', 'Números dos planetas (distância, tamanho, giro)', 'Planet numbers (distance, size, spin)', 'https://nssdc.gsfc.nasa.gov/planetary/factsheet/'],
];

export default function Credits() {
  const { t, lang } = usePrefs();
  const en = lang === 'en';
  const name = siteName(lang);
  return (
    <>
      <header className="page-head">
        <span className="eyebrow">{t('cr.eyebrow', { date: site.lastUpdate })}</span>
        <h1>{t('cr.title')}</h1>
        <p>{t('cr.lead', { name })}</p>
      </header>

      <div className="table-wrap panel" style={{ padding: 0 }}>
        <table>
          <thead>
            <tr>
              <th>{t('cr.col.source')}</th>
              <th>{t('cr.col.license')}</th>
              <th>{t('cr.col.credit')}</th>
              <th>{t('cr.col.ref')}</th>
            </tr>
          </thead>
          <tbody>
            {SOURCES.map(([src, licPt, licEn, crPt, crEn, url]) => (
              <tr key={src}>
                <td style={{ color: 'var(--text)' }}>{src}</td>
                <td>{en ? licEn : licPt}</td>
                <td>{en ? crEn : crPt}</td>
                <td>
                  <a href={url} target="_blank" rel="noreferrer">
                    {t('cr.official')}
                  </a>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p style={{ color: "var(--muted)", fontSize: "0.9rem", margin: "12px 0 20px" }}>{t("cr.planets")}</p>

      <div className="grid two">
        <section className="panel" style={{ display: 'grid', gap: 8 }}>
          <h2>{t('cr.software')}</h2>
          <p style={{ color: 'var(--text-2)' }}>{t('cr.softwareV')}</p>
        </section>
        <section className="panel" style={{ display: 'grid', gap: 8 }}>
          <h2>{t('cr.noAff')}</h2>
          <p style={{ color: 'var(--text-2)' }}>{t('cr.noAffV', { name })}</p>
        </section>
      </div>
    </>
  );
}
