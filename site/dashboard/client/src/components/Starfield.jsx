// Fundo animado em canvas: estrelas que cintilam em 3 camadas (paralaxe com o mouse),
// nuvens de nebulosa que se movem devagar e estrelas cadentes ocasionais.
// No tema claro: brilhos em forma de estrela, linhas de constelação e estrelas cadentes coloridas. Com "reduzir movimento" ligado, desenha um quadro estático.
import { useEffect, useRef } from 'react';
import { usePrefs } from '../prefs.jsx';

const DARK = {
  stars: ['#f1f5ff', '#f1f5ff', '#f1f5ff', '#93c5fd', '#c4b5fd', '#5eead4'],
  clouds: [
    [139, 92, 246, 0.16],
    [59, 130, 246, 0.13],
    [45, 212, 191, 0.08],
  ],
};
const LIGHT = {
  stars: ['#2563eb', '#4f46e5', '#7c3aed', '#0d9488', '#60a5fa', '#8b5cf6'],
  clouds: [
    [139, 92, 246, 0.1],
    [59, 130, 246, 0.1],
    [45, 212, 191, 0.08],
  ],
};

export default function Starfield() {
  const ref = useRef(null);
  const { resolvedTheme } = usePrefs();

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas.getContext('2d');
    const light = resolvedTheme === 'light';
    const pal = light ? LIGHT : DARK;
    const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    let w = 0;
    let h = 0;
    let dpr = 1;
    let stars = [];
    let clouds = [];
    let meteors = [];
    let raf = 0;
    let nextMeteor = performance.now() + 2500;
    const mouse = { x: 0, y: 0, tx: 0, ty: 0 };

    const rand = (a, b) => a + Math.random() * (b - a);

    function resize() {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      w = window.innerWidth;
      h = window.innerHeight;
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const count = Math.round((w * h) / (light ? 7000 : 4200));
      stars = Array.from({ length: count }, () => {
        const layer = Math.random() < 0.6 ? 0 : Math.random() < 0.75 ? 1 : 2;
        return {
          x: Math.random() * w,
          y: Math.random() * h,
          r: light ? [1.2, 2, 3.2][layer] * rand(0.8, 1.2) : [0.5, 0.9, 1.4][layer] * rand(0.8, 1.2),
          vx: light ? rand(-0.06, 0.06) : 0,
          vy: light ? rand(-0.04, 0.04) : 0,
          layer,
          color: pal.stars[(Math.random() * pal.stars.length) | 0],
          phase: Math.random() * Math.PI * 2,
          speed: rand(0.6, 2.2),
          base: light ? rand(0.35, 0.75) : rand(0.35, 0.95),
        };
      });
      clouds = pal.clouds.map(([r, g, b, a], i) => ({
        r,
        g,
        b,
        a,
        x: [0.82, 0.12, 0.55][i] * w,
        y: [0.12, 0.85, 0.55][i] * h,
        rad: Math.max(w, h) * [0.45, 0.4, 0.3][i],
        t: Math.random() * 1000,
      }));
    }

    function spawnMeteor() {
      const fromLeft = Math.random() < 0.5;
      meteors.push({
        x: fromLeft ? rand(-50, w * 0.5) : rand(w * 0.5, w + 50),
        y: rand(-40, h * 0.35),
        vx: (fromLeft ? 1 : -1) * rand(9, 14),
        vy: rand(3.5, 6),
        life: 0,
        max: rand(45, 70),
      });
    }

    function draw(now) {
      ctx.clearRect(0, 0, w, h);
      mouse.x += (mouse.tx - mouse.x) * 0.04;
      mouse.y += (mouse.ty - mouse.y) * 0.04;

      // nebulosas
      for (const c of clouds) {
        const t = reduce ? 0 : (now / 1000 + c.t) * 0.05;
        const cx = c.x + Math.cos(t) * 60 - mouse.x * 8;
        const cy = c.y + Math.sin(t * 1.3) * 40 - mouse.y * 8;
        const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, c.rad);
        g.addColorStop(0, `rgba(${c.r},${c.g},${c.b},${c.a})`);
        g.addColorStop(1, `rgba(${c.r},${c.g},${c.b},0)`);
        ctx.fillStyle = g;
        ctx.fillRect(0, 0, w, h);
      }

      // estrelas
      const pts = light ? [] : null;
      for (const s of stars) {
        const tw = reduce ? 1 : 0.55 + 0.45 * Math.sin(now / 1000 * s.speed + s.phase);
        if (light && !reduce) {
          s.x += s.vx;
          s.y += s.vy;
        }
        const par = (s.layer + 1) * 6;
        let x = s.x - mouse.x * par;
        let y = s.y - mouse.y * par;
        x = ((x % w) + w) % w;
        y = ((y % h) + h) % h;
        ctx.globalAlpha = s.base * tw;
        ctx.fillStyle = s.color;
        if (light) {
          // brilho de 4 pontas (estrelinha desenhada)
          const k = s.r * (0.8 + tw * 0.9);
          ctx.beginPath();
          ctx.moveTo(x, y - k * 2.2);
          ctx.quadraticCurveTo(x, y, x + k * 2.2, y);
          ctx.quadraticCurveTo(x, y, x, y + k * 2.2);
          ctx.quadraticCurveTo(x, y, x - k * 2.2, y);
          ctx.quadraticCurveTo(x, y, x, y - k * 2.2);
          ctx.fill();
          if (s.layer > 0) pts.push([x, y]);
          continue;
        }
        ctx.beginPath();
        ctx.arc(x, y, s.r, 0, Math.PI * 2);
        ctx.fill();
        if (s.layer === 2 && tw > 0.9) {
          // brilho em cruz nas estrelas mais próximas
          ctx.globalAlpha = s.base * (tw - 0.9) * 6;
          ctx.fillRect(x - s.r * 4, y - 0.4, s.r * 8, 0.8);
          ctx.fillRect(x - 0.4, y - s.r * 4, 0.8, s.r * 8);
        }
      }
      // linhas de constelação ligando estrelinhas próximas (tema claro)
      if (light) {
        const maxD = 150;
        ctx.lineWidth = 1;
        for (let i = 0; i < pts.length; i++) {
          for (let j = i + 1; j < pts.length; j++) {
            const dx = pts[i][0] - pts[j][0];
            const dy = pts[i][1] - pts[j][1];
            const d = Math.hypot(dx, dy);
            if (d < maxD) {
              ctx.globalAlpha = 0.22 * (1 - d / maxD);
              ctx.strokeStyle = '#6366f1';
              ctx.beginPath();
              ctx.moveTo(pts[i][0], pts[i][1]);
              ctx.lineTo(pts[j][0], pts[j][1]);
              ctx.stroke();
            }
          }
        }
      }
      ctx.globalAlpha = 1;

      // estrelas cadentes (nos dois temas, com cores próprias)
      if (!reduce) {
        if (now > nextMeteor) {
          spawnMeteor();
          nextMeteor = now + rand(3500, 8000);
        }
        meteors = meteors.filter((m) => m.life < m.max);
        for (const m of meteors) {
          m.life += 1;
          m.x += m.vx;
          m.y += m.vy;
          const fade = 1 - m.life / m.max;
          const tail = ctx.createLinearGradient(m.x, m.y, m.x - m.vx * 10, m.y - m.vy * 10);
          tail.addColorStop(0, light ? `rgba(79,70,229,${0.85 * fade})` : `rgba(241,245,255,${0.9 * fade})`);
          tail.addColorStop(0.3, light ? `rgba(59,130,246,${0.45 * fade})` : `rgba(94,234,212,${0.4 * fade})`);
          tail.addColorStop(1, 'rgba(139,92,246,0)');
          ctx.strokeStyle = tail;
          ctx.lineWidth = 1.6;
          ctx.lineCap = 'round';
          ctx.beginPath();
          ctx.moveTo(m.x, m.y);
          ctx.lineTo(m.x - m.vx * 10, m.y - m.vy * 10);
          ctx.stroke();
        }
      }

      if (!reduce) raf = requestAnimationFrame(draw);
    }

    const onMove = (e) => {
      mouse.tx = e.clientX / w - 0.5;
      mouse.ty = e.clientY / h - 0.5;
    };
    const onVisibility = () => {
      cancelAnimationFrame(raf);
      if (!document.hidden && !reduce) raf = requestAnimationFrame(draw);
    };

    const onResize = () => {
      resize();
      if (reduce) draw(performance.now());
    };

    resize();
    draw(performance.now());
    window.addEventListener('resize', onResize);
    window.addEventListener('pointermove', onMove, { passive: true });
    document.addEventListener('visibilitychange', onVisibility);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', onResize);
      window.removeEventListener('pointermove', onMove);
      document.removeEventListener('visibilitychange', onVisibility);
    };
  }, [resolvedTheme]);

  return <canvas ref={ref} className="starfield" aria-hidden="true" />;
}
