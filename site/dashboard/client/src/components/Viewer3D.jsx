// Janela com o modelo 3D de uma galáxia, nebulosa ou planeta.
// A biblioteca three.js só é carregada quando a janela abre (o site continua leve).
import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { usePrefs } from '../prefs.jsx';
import { OBJECTS, VIEW_TILT } from '../three/objects3d.js';
import { IconInfo } from './Icons.jsx';

const HD_KEY = 'pg-3d-hd';
// HD por padrão em computadores; em celulares começa no modo leve (pode ligar no botão)
function initialHD() {
  try {
    const saved = window.localStorage.getItem(HD_KEY);
    if (saved === '1' || saved === '0') return saved === '1';
  } catch {
    /* sem armazenamento */
  }
  const small = window.matchMedia?.('(max-width: 820px), (pointer: coarse)').matches;
  return !small;
}

/**
 * Partículas ao dar zoom bem de perto: limita o tamanho na tela (evita "bolhas" gigantes e deixa leve)
 * e apaga as que ficam quase encostadas na câmera.
 */
function tamePoints(material, maxPx) {
  if (material.userData.tamed) return;
  material.userData.tamed = true;
  material.onBeforeCompile = (shader) => {
    shader.uniforms.pgMaxPx = { value: maxPx };
    shader.vertexShader = shader.vertexShader
      .replace('void main() {', 'uniform float pgMaxPx;\nvarying float vPgNear;\nvoid main() {')
      .replace('#include <fog_vertex>', '#include <fog_vertex>\n  vPgNear = smoothstep(0.05, 0.6, -mvPosition.z);\n  gl_PointSize = min(gl_PointSize, pgMaxPx);');
    shader.fragmentShader = shader.fragmentShader
      .replace('void main() {', 'varying float vPgNear;\nvoid main() {')
      .replace('#include <opaque_fragment>', 'diffuseColor.a *= vPgNear;\n  #include <opaque_fragment>');
  };
  material.customProgramCacheKey = () => `pg-tame-${maxPx}`;
  material.needsUpdate = true;
}

export default function Viewer3D({ id, onClose }) {
  const { t, lang } = usePrefs();
  const obj = OBJECTS[id];
  const info = obj?.[lang] ?? obj?.pt;
  const mountRef = useRef(null);
  const closeRef = useRef(null);
  const apiRef = useRef({});
  const [status, setStatus] = useState('loading'); // loading | ready | error
  const [auto, setAuto] = useState(true);
  const [hd, setHD] = useState(initialHD);
  const [touch] = useState(() => Boolean(window.matchMedia?.('(pointer: coarse)').matches));

  const toggleHD = () => {
    const v = !hd;
    setHD(v);
    try {
      window.localStorage.setItem(HD_KEY, v ? '1' : '0');
    } catch {
      /* sem armazenamento */
    }
  };

  const fullscreen = () => {
    const el = mountRef.current;
    if (document.fullscreenElement) document.exitFullscreen?.().catch(() => {});
    else el?.requestFullscreen?.().catch(() => {});
  };

  // fechar com Esc e travar a rolagem da página
  useEffect(() => {
    const onKey = (e) => e.key === 'Escape' && onClose();
    document.addEventListener('keydown', onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    closeRef.current?.focus();
    return () => {
      document.removeEventListener('keydown', onKey);
      document.body.style.overflow = prev;
    };
  }, [onClose]);

  useEffect(() => {
    if (!obj) return undefined;
    let disposed = false;
    let raf = 0;
    let cleanup = () => {};
    const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

    (async () => {
      try {
        const THREE = await import('three');
        const { OrbitControls } = await import('three/examples/jsm/controls/OrbitControls.js');
        const isPlanet = obj.kind === 'planet';
        // planetas usam esferas com textura (three/planets.js); galáxias e nebulosas, partículas (three/scenes.js)
        const builder = isPlanet ? await import('../three/planets.js') : await import('../three/scenes.js');
        // efeitos de brilho (bloom) só no HD
        const post = hd
          ? await Promise.all([
              import('three/examples/jsm/postprocessing/EffectComposer.js'),
              import('three/examples/jsm/postprocessing/RenderPass.js'),
              import('three/examples/jsm/postprocessing/UnrealBloomPass.js'),
              import('three/examples/jsm/postprocessing/OutputPass.js'),
            ])
          : null;
        if (disposed) return;
        setStatus('loading');
        const mount = mountRef.current;
        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' });
        // HD: resolução nativa da tela (até 2,5x); modo leve: até 1,25x
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, hd ? 2.5 : 1.25));
        renderer.setClearColor('#02040d', 1);
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = hd ? 1.15 : 1.3;
        mount.appendChild(renderer.domElement);

        const scene = new THREE.Scene();
        // fundo definido na cena (com gestão de cor), para o HD não clarear o preto do espaço
        scene.background = new THREE.Color('#02040d');
        // "near" pequeno para dar zoom bem perto de um detalhe sem cortar a imagem
        const camera = new THREE.PerspectiveCamera(50, 1, 0.02, 400);
        const built = isPlanet ? await builder.buildPlanet(THREE, obj, hd ? 3 : 1) : builder.buildScene(THREE, obj, hd ? 3 : 1);
        if (disposed) return;
        const tilt = isPlanet ? built.initialTilt : (obj.tiltOverride ?? VIEW_TILT[obj.view] ?? 0.8);
        built.group.rotation.x = tilt;
        scene.add(built.group);
        for (const extra of built.extras ?? []) scene.add(extra);

        const home = new THREE.Vector3(0, 0, isPlanet ? built.camDist : 26);
        camera.position.copy(home);
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        // zoom em qualquer ponto: a roda do mouse / a pinça aproximam do ponto indicado (não só do centro)
        controls.zoomToCursor = true;
        controls.enablePan = true; // botão direito ou dois dedos: mover para os lados
        controls.screenSpacePanning = true;
        controls.minDistance = 0.12;
        controls.maxDistance = isPlanet ? 60 : 55;
        controls.zoomSpeed = 1.1;
        // planetas giram no próprio eixo (em vez de a câmera girar em volta)
        controls.autoRotate = !reduce && !obj.flat && !isPlanet;
        controls.autoRotateSpeed = 0.6;
        // sessão 11: giro livre de 360° em todos os modelos (antes as nebulosas "de frente" tinham giro limitado)

        // ---------------- zoom no ponto escolhido
        const pixelRatio = renderer.getPixelRatio();
        scene.traverse((o) => {
          // limite de tamanho na tela: 40 px (x densidade da tela) — com dezenas de milhares de partículas,
          // pontos enormes ao entrar numa nuvem deixariam o desenho muito pesado (celulares travariam)
          if (o.isPoints && o.material?.isPointsMaterial) tamePoints(o.material, Math.round((o.material.userData.maxPx ?? 40) * pixelRatio));
        });
        const pickables = [];
        scene.traverse((o) => {
          if ((o.isPoints || o.isMesh) && !o.userData.noPick) pickables.push(o);
        });
        const pickRadius = isPlanet ? built.pickRadius : 30;
        const minCamR = isPlanet ? built.radius * 1.025 : 0; // não deixa a câmera entrar no planeta
        const raycaster = new THREE.Raycaster();
        raycaster.camera = camera;
        const ndc = new THREE.Vector2();
        const fwd = new THREE.Vector3();
        const tmp = new THREE.Vector3();

        /** acha o ponto do modelo que está embaixo do mouse/dedo */
        const pickAt = (clientX, clientY) => {
          const r = renderer.domElement.getBoundingClientRect();
          ndc.set(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1);
          raycaster.setFromCamera(ndc, camera);
          // tolerância para acertar partículas: cresce com a distância
          raycaster.params.Points.threshold = Math.max(0.02, camera.position.distanceTo(controls.target) * 0.012);
          const hits = raycaster.intersectObjects(pickables, false);
          return hits.find((h) => h.point.length() < pickRadius) ?? null;
        };

        const stopAuto = () => {
          if (apiRef.current.auto) {
            apiRef.current.setAuto(false);
            setAuto(false);
          }
        };

        // Antes de cada zoom, o centro do zoom vai para a profundidade do ponto indicado.
        // Assim a aproximação chega perto desse ponto, sem atravessar o modelo.
        let lastPick = 0;
        const anchorZoom = (clientX, clientY) => {
          const now = performance.now();
          if (now - lastPick < 80) return;
          lastPick = now;
          const hit = pickAt(clientX, clientY);
          // nada embaixo do mouse/dedo (espaço vazio): o zoom vai para o centro atual, como antes,
          // para a pessoa não "se perder" voando para o vazio
          controls.zoomToCursor = Boolean(hit);
          if (!hit) return;
          camera.getWorldDirection(fwd);
          const depth = tmp.copy(hit.point).sub(camera.position).dot(fwd);
          if (depth > controls.minDistance * 1.5) controls.target.copy(camera.position).addScaledVector(fwd, depth);
        };

        // voo suave até um ponto (duplo clique / dois toques)
        let fly = null;
        const focusAt = (clientX, clientY) => {
          const hit = pickAt(clientX, clientY);
          if (!hit) return;
          stopAuto();
          const toCam = hit.point.clone().add(camera.position.clone().sub(hit.point).multiplyScalar(0.35));
          if (toCam.distanceTo(hit.point) < 0.6) toCam.sub(hit.point).setLength(0.6).add(hit.point);
          if (minCamR && toCam.length() < minCamR * 1.02) toCam.setLength(minCamR * 1.02);
          fly = { t0: performance.now(), dur: 700, fromCam: camera.position.clone(), fromT: controls.target.clone(), toCam, toT: hit.point.clone() };
        };

        // já deu zoom (ou moveu o centro)? então o clique duplo volta para a vista inicial
        const isZoomed = () => controls.target.length() > 0.05 || Math.abs(camera.position.length() - home.length()) > home.length() * 0.06;
        const flyHome = () => {
          fly = { t0: performance.now(), dur: 800, fromCam: camera.position.clone(), fromT: controls.target.clone(), toCam: home.clone(), toT: new THREE.Vector3() };
        };

        const stage = mount;
        const onWheel = (e) => {
          fly = null;
          anchorZoom(e.clientX, e.clientY);
          stopAuto();
        };
        const pointers = new Map();
        // Correção para a pinça: o three.js calcula o centro da pinça com a posição na PÁGINA (pageX/pageY),
        // que inclui a rolagem da página por trás da janela; aqui o centro volta a ser o ponto da tela.
        const origZoomParams = controls._updateZoomParameters?.bind(controls);
        if (origZoomParams) {
          controls._updateZoomParameters = (x, y) =>
            pointers.size >= 2 ? origZoomParams(x - window.scrollX, y - window.scrollY) : origZoomParams(x, y);
        }
        let lastTap = null;
        const onPointerDown = (e) => {
          fly = null;
          pointers.set(e.pointerId, { x: e.clientX, y: e.clientY, t: performance.now() });
          if (pointers.size === 2) {
            // começo da pinça: mira no ponto entre os dois dedos
            const [a, b] = [...pointers.values()];
            anchorZoom((a.x + b.x) / 2, (a.y + b.y) / 2);
            stopAuto();
          }
        };
        const onPointerUp = (e) => {
          const d = pointers.get(e.pointerId);
          pointers.delete(e.pointerId);
          if (!d || e.button > 0) return;
          const now = performance.now();
          const moved = Math.hypot(e.clientX - d.x, e.clientY - d.y);
          if (moved > 8 || now - d.t > 300) return; // não foi um toque rápido
          if (lastTap && now - lastTap.t < 380 && Math.hypot(e.clientX - lastTap.x, e.clientY - lastTap.y) < 30) {
            lastTap = null;
            if (isZoomed()) flyHome();
            else focusAt(e.clientX, e.clientY);
          } else {
            lastTap = { x: e.clientX, y: e.clientY, t: now };
          }
        };
        const onPointerCancel = (e) => pointers.delete(e.pointerId);
        // fase de captura no elemento pai: roda antes dos controles, que ficam no <canvas>
        stage.addEventListener('wheel', onWheel, { capture: true, passive: true });
        stage.addEventListener('pointerdown', onPointerDown, { capture: true });
        stage.addEventListener('pointerup', onPointerUp, { capture: true });
        stage.addEventListener('pointercancel', onPointerCancel, { capture: true });

        let composer = null;
        let bloom = null;
        if (post) {
          const [{ EffectComposer }, { RenderPass }, { UnrealBloomPass }, { OutputPass }] = post;
          composer = new EffectComposer(renderer);
          composer.addPass(new RenderPass(scene, camera));
          // força, raio e limite: só o que já é bem brilhante ganha halo (evita "neblina" no fundo)
          // planetas: brilho bem mais fraco, só na borda iluminada e na atmosfera
          bloom = isPlanet ? new UnrealBloomPass(new THREE.Vector2(1, 1), 0.3, 0.4, 0.75) : new UnrealBloomPass(new THREE.Vector2(1, 1), 0.7, 0.45, 0.35);
          composer.addPass(bloom);
          composer.addPass(new OutputPass());
        }

        const resize = () => {
          const w = mount.clientWidth;
          const h = mount.clientHeight;
          renderer.setSize(w, h, false);
          composer?.setSize(w, h);
          camera.aspect = w / h;
          camera.updateProjectionMatrix();
        };
        const ro = new ResizeObserver(resize);
        ro.observe(mount);
        resize();

        const t0 = performance.now();
        const loop = () => {
          const time = (performance.now() - t0) / 1000;
          built.animate(time, reduce, apiRef.current.auto !== false, camera);
          if (obj.flat && !reduce && apiRef.current.auto) built.group.rotation.y = Math.sin(time * 0.25) * 0.22;
          if (fly) {
            const k = Math.min(1, (performance.now() - fly.t0) / fly.dur);
            const e = k < 0.5 ? 4 * k * k * k : 1 - (-2 * k + 2) ** 3 / 2; // começa e termina devagar
            camera.position.lerpVectors(fly.fromCam, fly.toCam, e);
            controls.target.lerpVectors(fly.fromT, fly.toT, e);
            if (k >= 1) fly = null;
          }
          controls.update();
          if (minCamR && camera.position.length() < minCamR) camera.position.setLength(minCamR);
          if (camera.position.length() > 90) camera.position.setLength(90);
          if (composer) composer.render();
          else renderer.render(scene, camera);
          raf = requestAnimationFrame(loop);
        };
        loop();

        apiRef.current = {
          auto: !reduce,
          setAuto(v) {
            this.auto = v;
            if (!obj.flat && !isPlanet) controls.autoRotate = v;
          },
          reset() {
            fly = null;
            camera.position.copy(home);
            controls.target.set(0, 0, 0);
            controls.update();
          },
        };
        setAuto(!reduce);
        setStatus('ready');

        cleanup = () => {
          cancelAnimationFrame(raf);
          stage.removeEventListener('wheel', onWheel, { capture: true });
          stage.removeEventListener('pointerdown', onPointerDown, { capture: true });
          stage.removeEventListener('pointerup', onPointerUp, { capture: true });
          stage.removeEventListener('pointercancel', onPointerCancel, { capture: true });
          ro.disconnect();
          controls.dispose();
          composer?.dispose?.();
          bloom?.dispose?.();
          scene.traverse((o) => {
            o.geometry?.dispose?.();
            o.material?.map?.dispose?.();
            o.material?.alphaMap?.dispose?.();
            o.material?.dispose?.();
          });
          renderer.dispose();
          renderer.domElement.remove();
        };
      } catch (err) {
        console.error('[3D]', err);
        if (!disposed) setStatus('error');
      }
    })();

    return () => {
      disposed = true;
      cleanup();
    };
  }, [id, hd]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!obj) return null;

  // renderizado direto no <body> para ficar acima de tudo (inclusive do cabeçalho fixo no celular)
  return createPortal(
    <div className="viewer-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="viewer" role="dialog" aria-modal="true" aria-labelledby="viewer-title">
        <div className="viewer-stage" ref={mountRef}>
          {status === 'loading' && <div className="viewer-msg">{t('v3d.loading')}</div>}
          {status === 'error' && <div className="viewer-msg">{t('v3d.error')}</div>}
          <div className="viewer-hint">{touch ? t('v3d.hintTouch') : t('v3d.hint')}</div>
        </div>
        <aside className="viewer-info">
          <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'start' }}>
            <h2 id="viewer-title">{info.name}</h2>
            <button ref={closeRef} type="button" className="viewer-close" onClick={onClose} aria-label={t('v3d.close')}>
              ×
            </button>
          </div>
          <dl className="viewer-facts">
            <div>
              <dt className="eyebrow">{obj.kind === 'planet' ? t('v3d.distanceSun') : t('v3d.distance')}</dt>
              <dd>{info.dist}</dd>
            </div>
            {info.size && (
              <div>
                <dt className="eyebrow">{t('v3d.size')}</dt>
                <dd>{info.size}</dd>
              </div>
            )}
          </dl>
          <p>{info.fact}</p>
          <div className="viewer-actions">
            <button
              type="button"
              className="btn ghost"
              onClick={() => {
                const v = !auto;
                setAuto(v);
                apiRef.current.setAuto?.(v);
              }}
            >
              {auto ? t('v3d.stop') : t('v3d.play')}
            </button>
            <button type="button" className="btn ghost" onClick={() => apiRef.current.reset?.()}>
              {t('v3d.reset')}
            </button>
            <button type="button" className="btn ghost" onClick={fullscreen}>
              {t('v3d.full')}
            </button>
          </div>
          <div className="quality" role="group" aria-label={t('v3d.quality')}>
            <span className="eyebrow">{t('v3d.quality')}</span>
            <div className="seg">
              <button type="button" aria-pressed={hd} onClick={() => !hd && toggleHD()}>
                HD
              </button>
              <button type="button" aria-pressed={!hd} onClick={() => hd && toggleHD()}>
                {t('v3d.light')}
              </button>
            </div>
            <span className="quality-hint">{hd ? t('v3d.hdOn') : t('v3d.hdOff')}</span>
          </div>
          <div className="notice" style={{ marginTop: 'auto' }}>
            <IconInfo />
            <span>{obj.kind === 'planet' ? t('v3d.notePlanet') : t('v3d.note')}</span>
          </div>
        </aside>
      </div>
    </div>,
    document.body,
  );
}
