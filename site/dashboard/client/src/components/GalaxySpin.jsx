// Galáxia espiral animada em partículas (canvas). Gira com rotação diferencial:
// o centro gira mais rápido que as bordas, como nas galáxias reais.
import { useEffect, useRef } from 'react';
import { usePrefs } from '../prefs.jsx';

export default function GalaxySpin({ size = 260, arms = 2, className = '' }) {
  const ref = useRef(null);
  const { resolvedTheme } = usePrefs();

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas.getContext('2d');
    const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    const light = resolvedTheme === 'light';
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const R = size * 0.46;
    const colors = light
      ? ['#1d4ed8', '#4338ca', '#7c3aed', '#0d9488']
      : ['#93c5fd', '#3b82f6', '#8b5cf6', '#c4b5fd', '#5eead4'];
    const parts = [];
    const N = Math.round(size * 11);
    const gauss = () => (Math.random() + Math.random() + Math.random() - 1.5) / 1.5;
    for (let i = 0; i < N; i++) {
      const arm = i % arms;
      const t = Math.pow(Math.random(), 0.7); // mais partículas perto do centro
      const r = t * R;
      const spread = 0.28 + (1 - t) * 0.5;
      const halo = i % 5 === 0; // estrelas espalhadas no disco, fora dos braços
      const angle = halo ? Math.random() * Math.PI * 2 : arm * ((Math.PI * 2) / arms) + t * 4.2 + gauss() * spread;
      parts.push({
        r,
        a: angle,
        size: Math.random() < 0.05 ? 2.2 : Math.random() * 1.3 + 0.5,
        color: t < 0.15 ? (light ? '#1e1b4b' : '#f1f5ff') : colors[(Math.random() * colors.length) | 0],
        alpha: (light ? 0.3 + Math.random() * 0.5 : 0.25 + Math.random() * 0.6) * (halo ? 0.45 : 1),
        z: (Math.random() - 0.5) * (1 - t) * 10,
      });
    }

    let raf = 0;
    const tilt = 0.58;
    function draw(now) {
      ctx.clearRect(0, 0, size, size);
      const cx = size / 2;
      const cy = size / 2;
      // brilho do núcleo
      const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, R * 0.45);
      g.addColorStop(0, light ? 'rgba(79,70,229,0.35)' : 'rgba(241,245,255,0.55)');
      g.addColorStop(0.35, light ? 'rgba(59,130,246,0.15)' : 'rgba(94,234,212,0.18)');
      g.addColorStop(1, 'rgba(139,92,246,0)');
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, size, size);

      const time = reduce ? 0 : now / 1000;
      ctx.globalCompositeOperation = light ? 'source-over' : 'lighter';
      for (const p of parts) {
        const omega = 0.35 / (0.25 + p.r / R); // rotação diferencial
        const a = p.a + time * omega * 0.35;
        const x = cx + Math.cos(a) * p.r;
        const y = cy + Math.sin(a) * p.r * tilt + p.z;
        ctx.globalAlpha = p.alpha;
        ctx.fillStyle = p.color;
        ctx.fillRect(x, y, p.size, p.size);
      }
      ctx.globalAlpha = 1;
      ctx.globalCompositeOperation = 'source-over';
      if (!reduce) raf = requestAnimationFrame(draw);
    }
    draw(performance.now());
    return () => cancelAnimationFrame(raf);
  }, [size, arms, resolvedTheme]);

  return <canvas ref={ref} className={`galaxy-spin ${className}`} style={{ width: size, height: size }} aria-hidden="true" />;
}
