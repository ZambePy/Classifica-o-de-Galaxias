import { Link } from 'react-router-dom';
import { site } from '../data/site.js';
import { usePrefs } from '../prefs.jsx';
import TranslationNote from '../components/TranslationNote.jsx';
import Sections from '../components/Sections.jsx';

const Ph = ({ children }) => <span className="placeholder">{children}</span>;

// Política de privacidade em linguagem simples (sessão 3), seguindo a LGPD.
// A coluna "Por que podemos" traduz a base legal (LGPD art. 7º) para uma frase comum.
const DATA_PT = [
  ['Visitar o painel', 'Nada. Não pedimos nenhum dado.', '—', '—'],
  ['Endereço de internet (IP) e horário de acesso', 'Proteger o site contra ataques', 'Para o site funcionar com segurança', '30 dias'],
  ['E-mail e senha, se houver login', 'Criar e manter sua conta', 'Porque você pediu a conta', 'Até você apagar a conta'],
  ['Foto enviada', 'Analisar e mostrar o resultado', 'Porque você pediu a análise', 'Sempre apagada logo depois da análise'],
  ['Mensagens do chat', 'Responder suas perguntas', 'Porque você escreveu para o assistente', 'Não guardamos'],
  ['Idioma e tema escolhidos (no seu navegador)', 'Lembrar suas escolhas', 'Para o site funcionar como você prefere', 'Até você limpar o navegador'],
];
const DATA_EN = [
  ['Visiting the dashboard', 'Nothing. We ask for no data.', '—', '—'],
  ['Internet address (IP) and access time', 'Protect the site from attacks', 'To keep the site working safely', '30 days'],
  ['Email and password, if login exists', 'Create and keep your account', 'Because you asked for an account', 'Until you delete the account'],
  ['Uploaded photo', 'Analyse it and show the result', 'Because you asked for the analysis', 'Always deleted right after the analysis'],
  ['Chat messages', 'Answer your questions', 'Because you wrote to the assistant', 'Not stored'],
  ['Chosen language and theme (in your browser)', 'Remember your choices', 'So the site works the way you like', 'Until you clear your browser'],
];

function DataTable({ rows, head }) {
  return (
    <div className="table-wrap panel" style={{ padding: 0 }}>
      <table>
        <thead>
          <tr>
            {head.map((h) => (
              <th key={h}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r[0]}>
              {r.map((c, i) => (
                <td key={i} style={i === 0 ? { color: 'var(--text)' } : undefined}>
                  {c}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function PrivacyPt() {
  return (
    <>
      <header className="page-head">
        <span className="eyebrow">Atualizado em {site.lastUpdate}</span>
        <h1>Privacidade</h1>
        <p>
          Em resumo: pegamos o mínimo de dados, não vendemos nada, não temos propaganda e você pode pedir para apagar tudo
          quando quiser. Seguimos a Lei Geral de Proteção de Dados (LGPD).
        </p>
      </header>

      <Sections>
        <h2 id="bases">O que guardamos e por quê</h2>
        <DataTable rows={DATA_PT} head={['O quê', 'Para quê', 'Por que podemos', 'Por quanto tempo']} />
        <p>Fotos de galáxias e nebulosas não são dados pessoais. Nunca pedimos CPF, endereço ou telefone.</p>

        <h2 id="responsavel">Quem cuida dos seus dados</h2>
        <p>
          Responsável: <Ph>{site.responsible}</Ph>. Contato: <Ph>{site.privacyEmail}</Ph>.
        </p>

        <h2 id="cookies">Cookies</h2>
        <p>
          Não usamos cookies de propaganda nem de terceiros. Guardamos no seu navegador só o idioma e o tema que você escolheu.
          As letras do site vêm do nosso próprio servidor, então sua visita não é avisada a outras empresas. Se isso mudar um
          dia, vamos pedir sua permissão antes, com botões de aceitar e recusar do mesmo tamanho.
        </p>

        <h2 id="compartilhamento">Com quem dividimos</h2>
        <p>
          Com ninguém, a não ser a empresa que hospeda o site (<Ph>{site.hosting}</Ph>). Algumas fotos do painel podem ser
          carregadas direto dos sites das agências espaciais, que recebem seu endereço de internet, como em qualquer site.
        </p>
        <p>
          Quando o chat passar a usar uma IA de outra empresa, suas mensagens serão enviadas a ela só para gerar a resposta.
          Vamos dizer aqui qual empresa é e onde ela fica.
        </p>

        <h2 id="seguranca">Segurança</h2>
        <ul>
          <li>Conexão protegida (o cadeado no navegador).</li>
          <li>Senhas guardadas de forma embaralhada, nunca como texto.</li>
          <li>As fotos enviadas só passam pela memória do servidor e são apagadas logo depois da análise. Nada delas fica guardado, nem informações escondidas no arquivo, como o local onde foram tiradas.</li>
          <li>Se acontecer um vazamento que possa prejudicar você, avisamos você e a autoridade de proteção de dados (ANPD).</li>
        </ul>

        <h2 id="direitos">Seus direitos</h2>
        <p>Você pode pedir, de graça e quando quiser:</p>
        <ul>
          <li>Saber se temos dados seus e ver quais são.</li>
          <li>Corrigir dados errados.</li>
          <li>Apagar seus dados.</li>
          <li>Receber uma cópia dos seus dados.</li>
          <li>Retirar uma autorização que você deu.</li>
          <li>Saber com quem dividimos seus dados.</li>
        </ul>
        <p>
          Peça pelo e-mail <Ph>{site.privacyEmail}</Ph>. Respondemos em até 15 dias. Se não gostar da resposta, você pode
          reclamar com a ANPD em <a href="https://www.gov.br/anpd" target="_blank" rel="noreferrer">gov.br/anpd</a>.
        </p>

        <h2 id="menores">Crianças e adolescentes</h2>
        <p>
          Sempre pensamos primeiro no bem das crianças e adolescentes. Menores de 12 anos usam o site sem conta, ou com conta
          feita pela escola e autorização dos pais ou responsáveis. Não há propaganda nem conversa entre usuários.
        </p>

        <h2 id="ia">A IA e você</h2>
        <p>
          A IA só analisa fotos do espaço e não toma nenhuma decisão sobre pessoas. Veja em{' '}
          <Link to="/modelo">Como funciona</Link>.
        </p>

        <h2 id="mudancas">Mudanças</h2>
        <p>A data da versão atual fica no topo da página.</p>
      </Sections>
    </>
  );
}

function PrivacyEn() {
  return (
    <>
      <header className="page-head">
        <span className="eyebrow">Updated on {site.lastUpdate}</span>
        <h1>Privacy</h1>
        <p>
          In short: we collect as little data as possible, sell nothing, show no ads, and you can ask us to delete everything
          anytime. We follow Brazil's General Data Protection Law (LGPD).
        </p>
      </header>
      <TranslationNote />

      <Sections>
        <h2 id="bases">What we keep and why</h2>
        <DataTable rows={DATA_EN} head={['What', 'What for', 'Why we can', 'How long']} />
        <p>Photos of galaxies and nebulae are not personal data. We never ask for ID numbers, address or phone.</p>

        <h2 id="responsavel">Who looks after your data</h2>
        <p>
          Responsible: <Ph>{site.responsible}</Ph>. Contact: <Ph>{site.privacyEmail}</Ph>.
        </p>

        <h2 id="cookies">Cookies</h2>
        <p>
          We use no advertising or third-party cookies. We only store the language and theme you chose in your browser. The
          site's fonts come from our own server, so your visit is not reported to other companies. If this ever changes, we
          will ask for your permission first, with accept and reject buttons of the same size.
        </p>

        <h2 id="compartilhamento">Who we share with</h2>
        <p>
          No one, except the company that hosts the site (<Ph>{site.hosting}</Ph>). Some dashboard photos may load directly
          from space agency websites, which receive your internet address, as with any website.
        </p>
        <p>
          When the chat starts using an AI from another company, your messages will be sent to it only to create the answer.
          We will say here which company it is and where it is based.
        </p>

        <h2 id="seguranca">Security</h2>
        <ul>
          <li>Protected connection (the padlock in your browser).</li>
          <li>Passwords stored scrambled, never as plain text.</li>
          <li>Uploaded photos only pass through the server's memory and are deleted right after the analysis. Nothing from them is kept, not even hidden information in the file, such as where they were taken.</li>
          <li>If a leak could harm you, we tell you and Brazil's data protection authority (ANPD).</li>
        </ul>

        <h2 id="direitos">Your rights</h2>
        <p>You can ask, for free and anytime, to:</p>
        <ul>
          <li>Know if we have your data and see it.</li>
          <li>Fix wrong data.</li>
          <li>Delete your data.</li>
          <li>Get a copy of your data.</li>
          <li>Withdraw a permission you gave.</li>
          <li>Know who we share your data with.</li>
        </ul>
        <p>
          Ask by email at <Ph>{site.privacyEmail}</Ph>. We reply within 15 days. If you are not happy with the answer, you can
          complain to the ANPD at <a href="https://www.gov.br/anpd" target="_blank" rel="noreferrer">gov.br/anpd</a>.
        </p>

        <h2 id="menores">Children and teenagers</h2>
        <p>
          We always put the wellbeing of children and teenagers first. Children under 12 use the site without an account, or
          with an account made by the school and permission from their parents or guardians. There are no ads and no chat
          between users.
        </p>

        <h2 id="ia">The AI and you</h2>
        <p>
          The AI only analyses space photos and makes no decisions about people. See <Link to="/modelo">How it works</Link>.
        </p>

        <h2 id="mudancas">Changes</h2>
        <p>The date of the current version is at the top of the page.</p>
      </Sections>
    </>
  );
}

export default function Privacy() {
  const { lang } = usePrefs();
  return lang === 'en' ? <PrivacyEn /> : <PrivacyPt />;
}
