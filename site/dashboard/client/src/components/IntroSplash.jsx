// Abertura do site (sessão 14): o logo animado 10 ("Salto do espaço") toca no centro da tela e,
// ao terminar, voa encolhendo até o logo do menu, no canto superior esquerdo, onde fica parado.
// - Aparece uma vez por visita (sessionStorage); navegar entre páginas não repete.
// - Clique, toque, Esc ou o botão "Pular" terminam a animação na hora e o logo vai para o canto.
// - Quem pede menos movimento no sistema não vê a abertura.
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import logoSvg from '../assets/intro/logo-abertura.svg?raw';
import { usePrefs } from '../prefs.jsx';

const KEY = 'pg-abertura-vista';
const SVG_W = 670; // largura do desenho do logo animado
const SYM = { x: 22, y: 22, size: 156 }; // posição do símbolo dentro do desenho
const T_PLAY = 2500; // a animação 10 termina em ~2,25 s; pequena pausa antes de voar
const T_FLY = 900;
const T_LAND = 350;

function shouldShow() {
  try {
    if (window.sessionStorage.getItem(KEY) === '1') return false;
  } catch {
    /* sem armazenamento: mostra normalmente */
  }
  try {
    if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) return false;
  } catch {
    /* navegador antigo */
  }
  return true;
}

export default function IntroSplash() {
  const { t } = usePrefs();
  const [phase, setPhase] = useState(() => (shouldShow() ? 'play' : 'done')); // play | fly | land | done
  const logoRef = useRef(null);
  const timers = useRef([]);
  const flying = useRef(false);

  const later = (fn, ms) => timers.current.push(window.setTimeout(fn, ms));

  // o logo do menu fica escondido enquanto a abertura acontece (reaparece ao pousar)
  useLayoutEffect(() => {
    const root = document.documentElement;
    if (phase === 'play' || phase === 'fly') root.classList.add('intro-on');
    else root.classList.remove('intro-on');
    return () => root.classList.remove('intro-on');
  }, [phase]);

  const fly = useCallback(() => {
    if (flying.current) return;
    flying.current = true;
    timers.current.forEach(clearTimeout);
    timers.current = [];
    const wrap = logoRef.current;
    // termina qualquer parte da animação que ainda esteja rodando (ex.: ao pular)
    wrap?.getAnimations?.({ subtree: true }).forEach((a) => {
      try {
        a.finish();
      } catch {
        /* animação infinita ou já terminada */
      }
    });
    const target = document.querySelector('.sidebar .brand img');
    const r = target?.getBoundingClientRect();
    if (wrap && r && r.width > 0) {
      const box = wrap.getBoundingClientRect();
      const k = box.width / SVG_W;
      const s = r.width / (SYM.size * k);
      const tx = r.left - (box.left + SYM.x * k * s);
      const ty = r.top - (box.top + SYM.y * k * s);
      wrap.style.transform = `translate(${tx}px, ${ty}px) scale(${s})`;
    }
    setPhase('fly');
    later(() => setPhase('land'), T_FLY);
    later(() => setPhase('done'), T_FLY + T_LAND);
  }, []);

  useEffect(() => {
    if (phase !== 'play') return undefined;
    try {
      window.sessionStorage.setItem(KEY, '1');
    } catch {
      /* sem armazenamento: pode repetir na próxima visita */
    }
    later(fly, T_PLAY);
    const onKey = (e) => e.key === 'Escape' && fly();
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => () => timers.current.forEach(clearTimeout), []);

  if (phase === 'done') return null;

  return (
    <div className={`intro intro-${phase}`} onClick={fly} role="presentation">
      <div className="intro-bg" />
      <div className="intro-logo" ref={logoRef} aria-hidden="true" dangerouslySetInnerHTML={{ __html: logoSvg }} />
      <button
        type="button"
        className="intro-skip"
        onClick={(e) => {
          e.stopPropagation();
          fly();
        }}
      >
        {t('intro.skip')}
      </button>
    </div>
  );
}
