import { site } from '../data/site.js';
import { usePrefs } from '../prefs.jsx';
import TranslationNote from '../components/TranslationNote.jsx';
import Sections from '../components/Sections.jsx';

// Conformidade com a lei, em linguagem simples (sessão 3). Acessada pelo rodapé, ao lado dos Termos de uso.
// Detalhes técnicos e artigos de lei: pesquisa/04-base-legal.md.
const ITEMS = [
  // [PT, EN, lei, pronto?]
  ['Termos de uso claros, em português', 'Clear terms of use', 'Marco Civil da Internet', true],
  ['Explicar quais dados usamos e por quê', 'Explain what data we use and why', 'LGPD', true],
  ['Nada de cookies de propaganda', 'No advertising cookies', 'LGPD', true],
  ['Dar crédito a todas as imagens', 'Credit every image', 'Lei de Direitos Autorais', true],
  ['Avisar que os resultados vêm de uma IA', 'Say that results come from an AI', 'Boa prática', true],
  ['Chat avisa para não escrever dados pessoais e não guarda mensagens', 'Chat warns not to share personal data and stores no messages', 'LGPD', true],
  ['Apagar as fotos logo depois da análise, sem guardar nada delas', 'Delete photos right after the analysis, keeping nothing from them', 'LGPD', true],
  ['Proteção contra ataques ao site', 'Protection against attacks', 'LGPD', true],
  ['Sem propaganda e sem conversa entre usuários', 'No ads and no chat between users', 'ECA Digital', true],
  ['Deixar claro que não somos da NASA ou da ESA', 'Make clear we are not NASA or ESA', 'Regras das agências', true],
  ['Preencher nome do responsável, e-mail e cidade', 'Fill in responsible name, email and city', 'LGPD', false],
  ['Site com cadeado (conexão segura)', 'Site with padlock (secure connection)', 'LGPD', false],
  ['Guardar os dados no Brasil ou na Europa', 'Store data in Brazil or Europe', 'LGPD', false],
  ['Botões para apagar e baixar seus dados', 'Buttons to delete and download your data', 'LGPD', false],
  ['Autorização dos pais para menores de 12 anos', 'Parent permission for under-12s', 'LGPD', false],
  ['Plano para caso de vazamento de dados', 'Plan in case of a data leak', 'LGPD', false],
  ['Dizer qual empresa de IA o chat usa, quando houver', 'Say which AI company the chat uses, once chosen', 'LGPD', false],
  ['Pedir permissão e explicar o uso da câmera (aba Planetas)', 'Ask permission and explain camera use (Planets tab)', 'LGPD', false],
  ['Revisão por um professor ou advogado', 'Review by a teacher or lawyer', 'Recomendado', false],
];

const LAW_EN = {
  'Marco Civil da Internet': 'Internet Civil Framework',
  'Lei de Direitos Autorais': 'Copyright Law',
  'Boa prática': 'Good practice',
  'ECA Digital': 'Digital Statute of Children',
  'Regras das agências': 'Agency rules',
  Recomendado: 'Recommended',
};

export default function Compliance() {
  const { lang } = usePrefs();
  const en = lang === 'en';
  const done = ITEMS.filter((i) => i[3]).length;
  return (
    <>
      <header className="page-head">
        <span className="eyebrow">{en ? `Updated on ${site.lastUpdate}` : `Atualizado em ${site.lastUpdate}`}</span>
        <h1>{en ? 'Legal compliance' : 'Conformidade com a lei'}</h1>
        <p>
          {en
            ? 'What Brazilian law asks of a site like this, and what we have already done.'
            : 'O que a lei brasileira pede para um site como este e o que já fizemos.'}
        </p>
        <div className="chips">
          <span className="pill aqua">
            <span className="dot on" /> {en ? `${done} of ${ITEMS.length} ready` : `${done} de ${ITEMS.length} prontos`}
          </span>
          <span className="pill warn">
            <span className="dot" /> {en ? `${ITEMS.length - done} to do when publishing` : `${ITEMS.length - done} para fazer ao publicar`}
          </span>
        </div>
      </header>
      {en && <TranslationNote />}

      <div className="table-wrap panel" style={{ padding: 0 }}>
        <table>
          <thead>
            <tr>
              <th>{en ? 'What the law asks' : 'O que a lei pede'}</th>
              <th>{en ? 'Law' : 'Lei'}</th>
              <th>{en ? 'Status' : 'Situação'}</th>
            </tr>
          </thead>
          <tbody>
            {ITEMS.map(([pt, eng, law, ok]) => (
              <tr key={pt}>
                <td style={{ color: 'var(--text)' }}>{en ? eng : pt}</td>
                <td>{en ? (LAW_EN[law] ?? law) : law}</td>
                <td className={ok ? 'status-ok' : 'status-todo'} style={{ whiteSpace: 'nowrap', fontWeight: 600 }}>
                  {ok ? (en ? '✓ Ready' : '✓ Pronto') : en ? '○ To do' : '○ Falta fazer'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <Sections openFirst={false}>
        <h2 id="leis">{en ? 'Which laws we follow' : 'Quais leis seguimos'}</h2>
        {en ? (
          <ul>
            <li><strong>LGPD</strong>: Brazil's law on how personal data is used.</li>
            <li><strong>Internet Civil Framework</strong>: rights of people who use the internet in Brazil.</li>
            <li><strong>Copyright Law</strong>: rules for using images made by others.</li>
            <li><strong>Digital Statute of Children</strong>: protection for children and teenagers online.</li>
            <li><strong>AI law</strong>: still being discussed in Congress. We already follow its transparency ideas.</li>
          </ul>
        ) : (
          <ul>
            <li><strong>LGPD</strong>: a lei que diz como dados pessoais podem ser usados.</li>
            <li><strong>Marco Civil da Internet</strong>: direitos de quem usa a internet no Brasil.</li>
            <li><strong>Lei de Direitos Autorais</strong>: regras para usar imagens feitas por outras pessoas.</li>
            <li><strong>ECA Digital</strong>: proteção de crianças e adolescentes na internet.</li>
            <li><strong>Lei de IA</strong>: ainda em discussão no Congresso. Já seguimos as ideias de transparência dela.</li>
          </ul>
        )}
        <p>{en ? 'This page is informative and does not replace advice from a lawyer.' : 'Esta página é informativa e não substitui a orientação de um advogado.'}</p>
      </Sections>
    </>
  );
}
