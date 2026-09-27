import { Link } from 'react-router-dom';
import { site } from '../data/site.js';
import { usePrefs } from '../prefs.jsx';
import TranslationNote from '../components/TranslationNote.jsx';
import Sections from '../components/Sections.jsx';

const Ph = ({ children }) => <span className="placeholder">{children}</span>;

// Termos de uso em linguagem simples (sessão 3). Base: Marco Civil da Internet, LGPD,
// Lei de Direitos Autorais e ECA Digital — detalhes em pesquisa/04-base-legal.md.
function TermsPt() {
  return (
    <>
      <header className="page-head">
        <span className="eyebrow">Atualizado em {site.lastUpdate}</span>
        <h1>Termos de uso</h1>
        <p>As regras para usar o {site.name}, explicadas de forma simples.</p>
      </header>

      <Sections>
        <h2 id="sobre">O que é o projeto</h2>
        <p>
          O {site.name} é um projeto de estudo, gratuito e sem fins lucrativos, feito por <Ph>{site.responsible}</Ph> com
          orientação de <Ph>{site.advisor}</Ph>. Ele usa inteligência artificial para separar fotos de galáxias e nebulosas
          por tipo. Não tem propaganda e não vendemos nenhum dado.
        </p>
        <p>O projeto é independente. Não temos ligação oficial com a NASA, a ESA ou outras agências espaciais.</p>

        <h2 id="aceite">Ao usar o site</h2>
        <p>
          Usar o site significa que você concorda com estes termos e com a <Link to="/privacidade">política de privacidade</Link>.
          Para só olhar o painel, não precisa criar conta.
        </p>

        <h2 id="ia">Sobre a inteligência artificial</h2>
        <ul>
          <li>As classificações são feitas automaticamente e podem ter erros.</li>
          <li>Os resultados servem para estudo e curiosidade, não como fonte científica oficial.</li>
          <li>Cada resultado mostra o quanto a IA tem certeza.</li>
          <li>A IA analisa só fotos do espaço. Ela não avalia pessoas nem dá notas a alunos.</li>
          <li>O chat também usa IA e pode errar. Não escreva dados pessoais nele.</li>
        </ul>

        <h2 id="conta">Conta</h2>
        <p>
          Se o site tiver login, pediremos só e-mail e senha. Guarde sua senha com cuidado. Você pode apagar sua conta quando
          quiser, pelo contato no fim desta página.
        </p>

        <h2 id="envio">Envio de fotos</h2>
        <p>Ao enviar uma foto, você confirma que:</p>
        <ul>
          <li>A foto é sua ou você tem permissão para usá-la.</li>
          <li>Ela mostra o espaço e não mostra pessoas.</li>
          <li>Podemos usar a foto só para analisar e mostrar o resultado para você.</li>
          <li>A foto é sempre apagada logo depois da análise. Ela não fica guardada e não é usada para treinar a IA.</li>
          <li>Nada do arquivo fica guardado, nem informações escondidas nele, como o local onde a foto foi tirada.</li>
        </ul>

        <h2 id="proibido">O que não é permitido</h2>
        <ul>
          <li>Enviar conteúdo ilegal, ofensivo, violento ou que desrespeite direitos de outras pessoas.</li>
          <li>Tentar invadir ou derrubar o site.</li>
          <li>Pegar dados de outras pessoas ou usar o site para ganhar dinheiro sem autorização.</li>
          <li>Dizer que os resultados da IA são oficiais de alguma instituição científica.</li>
        </ul>
        <p>Quem não seguir essas regras pode ter o acesso bloqueado.</p>

        <h2 id="pi">Imagens e créditos</h2>
        <p>
          As fotos do espaço pertencem a quem as fez. Nós as usamos com permissão e sempre damos o crédito, como pede a Lei de
          Direitos Autorais. Veja a lista em <Link to="/creditos">Créditos</Link>. Não usamos logotipos de agências espaciais.
        </p>

        <h2 id="privacidade">Seus dados</h2>
        <p>
          Cuidamos dos seus dados seguindo a Lei Geral de Proteção de Dados (LGPD). Tudo está explicado na{' '}
          <Link to="/privacidade">política de privacidade</Link>.
        </p>

        <h2 id="menores">Crianças e adolescentes</h2>
        <p>
          O site pode ser usado em escolas. Crianças com menos de 12 anos devem usar sem criar conta, ou com conta feita pela
          escola e autorização dos pais ou responsáveis. O site não tem propaganda nem conversa entre usuários.
        </p>

        <h2 id="responsabilidade">Funcionamento do site</h2>
        <p>
          O site é gratuito e pode ficar fora do ar às vezes. Como é um projeto de estudo, não garantimos que ele funcione o
          tempo todo nem que a IA sempre acerte. Isso não tira nenhum direito que a lei brasileira garante a você.
        </p>

        <h2 id="mudancas">Mudanças</h2>
        <p>Podemos atualizar estes termos. A data da última versão fica no topo da página.</p>

        <h2 id="lei">Leis e local para resolver problemas</h2>
        <p>
          Seguimos as leis do Brasil. Se houver alguma disputa, ela será resolvida na cidade de <Ph>{site.city}</Ph>, ou na
          cidade onde você mora, se preferir.
        </p>

        <h2 id="contato">Contato</h2>
        <p>
          Dúvidas, pedidos sobre seus dados ou denúncias: <Ph>{site.privacyEmail}</Ph>.
        </p>
      </Sections>
    </>
  );
}

function TermsEn() {
  return (
    <>
      <header className="page-head">
        <span className="eyebrow">Updated on {site.lastUpdate}</span>
        <h1>Terms of use</h1>
        <p>The rules for using the {site.nameEn}, explained simply.</p>
      </header>
      <TranslationNote />

      <Sections>
        <h2 id="sobre">What the project is</h2>
        <p>
          The {site.nameEn} is a free, non-profit study project made by <Ph>{site.responsible}</Ph> with guidance from{' '}
          <Ph>{site.advisor}</Ph>. It uses artificial intelligence to sort photos of galaxies and nebulae by type. It has no
          ads and we do not sell any data.
        </p>
        <p>The project is independent. We have no official link with NASA, ESA or other space agencies.</p>

        <h2 id="aceite">Using the site</h2>
        <p>
          Using the site means you agree with these terms and with the <Link to="/privacidade">privacy policy</Link>. You do
          not need an account just to browse the dashboard.
        </p>

        <h2 id="ia">About the artificial intelligence</h2>
        <ul>
          <li>Classifications are made automatically and may be wrong.</li>
          <li>Results are for study and curiosity, not an official scientific source.</li>
          <li>Each result shows how sure the AI is.</li>
          <li>The AI only looks at space photos. It does not judge people or grade students.</li>
          <li>The chat also uses AI and may be wrong. Do not write personal data in it.</li>
        </ul>

        <h2 id="conta">Account</h2>
        <p>
          If the site has login, we will only ask for email and password. Keep your password safe. You can delete your account
          anytime using the contact at the end of this page.
        </p>

        <h2 id="envio">Uploading photos</h2>
        <p>When you upload a photo, you confirm that:</p>
        <ul>
          <li>The photo is yours or you have permission to use it.</li>
          <li>It shows space and does not show people.</li>
          <li>We may use the photo only to analyse it and show you the result.</li>
          <li>The photo is always deleted right after the analysis. It is not kept and is not used to train the AI.</li>
          <li>Nothing from the file is kept, not even hidden information in it, such as where the photo was taken.</li>
        </ul>

        <h2 id="proibido">What is not allowed</h2>
        <ul>
          <li>Uploading illegal, offensive or violent content, or content that disrespects other people's rights.</li>
          <li>Trying to break into or take down the site.</li>
          <li>Taking other people's data or using the site to make money without permission.</li>
          <li>Saying the AI results are official results from a scientific institution.</li>
        </ul>
        <p>Anyone who breaks these rules may be blocked.</p>

        <h2 id="pi">Images and credits</h2>
        <p>
          Space photos belong to the people who made them. We use them with permission and always give credit, as Brazilian
          copyright law requires. See the list in <Link to="/creditos">Credits</Link>. We do not use space agency logos.
        </p>

        <h2 id="privacidade">Your data</h2>
        <p>
          We take care of your data following Brazil's General Data Protection Law (LGPD). Everything is explained in the{' '}
          <Link to="/privacidade">privacy policy</Link>.
        </p>

        <h2 id="menores">Children and teenagers</h2>
        <p>
          The site may be used in schools. Children under 12 should use it without an account, or with an account made by the
          school and permission from their parents or guardians. The site has no ads and no chat between users.
        </p>

        <h2 id="responsabilidade">How the site works</h2>
        <p>
          The site is free and may sometimes be offline. As a study project, we cannot promise it will always work or that
          the AI will always be right. This does not remove any right Brazilian law gives you.
        </p>

        <h2 id="mudancas">Changes</h2>
        <p>We may update these terms. The date of the latest version is at the top of the page.</p>

        <h2 id="lei">Laws and where to solve problems</h2>
        <p>
          We follow the laws of Brazil. Any dispute will be handled in the city of <Ph>{site.city}</Ph>, or in the city where
          you live, if you prefer.
        </p>

        <h2 id="contato">Contact</h2>
        <p>
          Questions, data requests or reports: <Ph>{site.privacyEmail}</Ph>.
        </p>
      </Sections>
    </>
  );
}

export default function Terms() {
  const { lang } = usePrefs();
  return lang === 'en' ? <TermsEn /> : <TermsPt />;
}
