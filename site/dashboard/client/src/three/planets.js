// Planetas do Sistema Solar em 3D: esferas com textura, luz do Sol, atmosfera, nuvens (Terra)
// e anéis (Saturno e Urano). Texturas ILUSTRATIVAS geradas por código em
// ferramentas/texturas-planetas/gerar_texturas.py (continentes da Terra: Natural Earth, domínio público).
import { PLANET_TEXTURES, PLANET_TEXTURES_HD } from './planetTextures.js';
import { fbm } from './scenes.js';

// posição do Sol na cena (a mesma da luz direcional)
const SUN_POS = [-30, 8, 18];

/** halo da atmosfera: esfera um pouco maior vista por dentro; o brilho é forte junto da borda do planeta
 *  e some suavemente para fora. Só aparece do lado iluminado pelo Sol. */
function fresnelMaterial(THREE, color, power = 2.5, strength = 1) {
  return new THREE.ShaderMaterial({
    uniforms: {
      glowColor: { value: new THREE.Color(color) },
      power: { value: power },
      strength: { value: strength },
      sunDir: { value: new THREE.Vector3(...SUN_POS).normalize() },
    },
    vertexShader: `
      varying vec3 vNormal;
      varying vec3 vView;
      varying vec3 vWorldNormal;
      void main() {
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        vNormal = normalize(normalMatrix * normal);
        vWorldNormal = normalize(mat3(modelMatrix) * normal);
        vView = normalize(-mv.xyz);
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: `
      uniform vec3 glowColor;
      uniform float power;
      uniform float strength;
      uniform vec3 sunDir;
      varying vec3 vNormal;
      varying vec3 vView;
      varying vec3 vWorldNormal;
      void main() {
        float lit = smoothstep(-0.25, 0.55, dot(vWorldNormal, sunDir));
        float f = pow(abs(dot(vNormal, vView)), power) * strength * lit;
        gl_FragColor = vec4(glowColor * f, f);
      }`,
    transparent: true,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    side: THREE.BackSide,
  });
}

function loadTex(THREE, url, srgb = true) {
  return new Promise((resolve, reject) => {
    new THREE.TextureLoader().load(
      url,
      (t) => {
        if (srgb) t.colorSpace = THREE.SRGBColorSpace;
        t.anisotropy = 16; // o renderizador limita ao máximo que a placa de vídeo aceita
        resolve(t);
      },
      undefined,
      reject,
    );
  });
}

function starBackground(THREE) {
  const pos = [];
  const col = [];
  const c = new THREE.Color();
  for (let i = 0; i < 2200; i++) {
    const r = 80 + Math.random() * 40;
    const th = Math.random() * Math.PI * 2;
    const ph = Math.acos(Math.random() * 2 - 1);
    pos.push(r * Math.sin(ph) * Math.cos(th), r * Math.cos(ph), r * Math.sin(ph) * Math.sin(th));
    c.set(['#ffffff', '#cfe3ff', '#ffe6c2'][i % 3]).multiplyScalar(0.3 + Math.random() * 0.6);
    col.push(c.r, c.g, c.b);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  return new THREE.Points(geo, new THREE.PointsMaterial({ size: 0.35, vertexColors: true, sizeAttenuation: true, depthWrite: false }));
}

/** anel plano no equador; o mapa de textura acompanha a distância ao centro (raio) */
function ringMesh(THREE, inner, outer, material) {
  const geo = new THREE.RingGeometry(inner, outer, 256, 1);
  const pos = geo.attributes.position;
  const uv = geo.attributes.uv;
  const v = new THREE.Vector3();
  for (let i = 0; i < pos.count; i++) {
    v.fromBufferAttribute(pos, i);
    uv.setXY(i, (v.length() - inner) / (outer - inner), 0.5);
  }
  const mesh = new THREE.Mesh(geo, material);
  mesh.rotation.x = -Math.PI / 2;
  return mesh;
}

/** ponto redondo e macio (as partículas dos anéis viram "fumaça" ao chegar perto) */
function softDot(THREE, hard) {
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const g = c.getContext('2d');
  const grd = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  if (hard) {
    grd.addColorStop(0, 'rgba(255,255,255,1)');
    grd.addColorStop(0.45, 'rgba(255,255,255,0.8)');
    grd.addColorStop(1, 'rgba(255,255,255,0)');
  } else {
    grd.addColorStop(0, 'rgba(255,255,255,0.9)');
    grd.addColorStop(0.35, 'rgba(255,255,255,0.35)');
    grd.addColorStop(1, 'rgba(255,255,255,0)');
  }
  g.fillStyle = grd;
  g.fillRect(0, 0, 64, 64);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

/** lê o perfil (cor e densidade) da textura dos anéis, do raio interno ao externo */
function ringProfile(image) {
  const n = 512;
  const c = document.createElement('canvas');
  c.width = n;
  c.height = 1;
  const g = c.getContext('2d', { willReadFrequently: true });
  g.drawImage(image, 0, 0, n, 1);
  const d = g.getImageData(0, 0, n, 1).data;
  const col = [];
  const alpha = [];
  for (let i = 0; i < n; i++) {
    col.push([d[i * 4] / 255, d[i * 4 + 1] / 255, d[i * 4 + 2] / 255]);
    alpha.push(d[i * 4 + 3] / 255);
  }
  return { n, col, alpha };
}

// gerador pseudoaleatório com semente (o mesmo anel toda vez)
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Anéis de Saturno em partículas, por cima da textura lisa:
 * - "poeira": muitos grãos pequenos (de longe somem na textura; de perto aparecem os grãos);
 * - "gás": nuvens macias e irregulares (de perto o anel parece uma névoa, não uma placa).
 */
function saturnRingParticles(THREE, inner, outer, profile, quality) {
  const rand = rng(1610);
  const { n, col, alpha } = profile;
  // sorteio do raio proporcional à densidade × área (anel B cheio, divisão de Cassini quase vazia)
  const cdf = [];
  let acc = 0;
  for (let i = 0; i < n; i++) {
    const r = inner + ((i + 0.5) / n) * (outer - inner);
    acc += alpha[i] ** 1.4 * r;
    cdf.push(acc);
  }
  const sampleR = () => {
    const x = rand() * acc;
    let lo = 0;
    let hi = n - 1;
    while (lo < hi) {
      const m = (lo + hi) >> 1;
      if (cdf[m] < x) lo = m + 1;
      else hi = m;
    }
    return { i: lo, r: inner + ((lo + rand()) / n) * (outer - inner) };
  };
  const gauss = () => (rand() + rand() + rand() - 1.5) * 0.8;

  const make = (count, size, thick, clumpy, bright) => {
    const pos = new Float32Array(count * 3);
    const cols = new Float32Array(count * 3);
    let k = 0;
    let guard = 0;
    while (k < count && guard++ < count * 12) {
      const { i, r } = sampleR();
      const th = rand() * Math.PI * 2;
      const x = Math.cos(th) * r;
      const z = Math.sin(th) * r;
      // nuvens irregulares: o ruído aceita mais partículas em umas regiões que em outras
      const w = fbm(x * 0.55, r * 1.7, z * 0.55, 4);
      if (clumpy && rand() > Math.max(0.05, 0.25 + (w - 0.45) * 4)) continue; // fbm vai de 0 a 1: nuvens com buracos
      pos[k * 3] = x;
      pos[k * 3 + 1] = gauss() * thick;
      pos[k * 3 + 2] = z;
      // nuvens com tons variados: umas mais claras e frias, outras mais escuras e puxadas para o marrom
      const v = bright * (0.55 + rand() * 0.65) * (0.75 + 0.5 * alpha[i]) * (clumpy ? 0.75 + 0.5 * w : 1);
      const warm = clumpy ? (rand() - 0.5) * 0.18 : 0;
      cols[k * 3] = Math.min(1, col[i][0] * v * (1 + warm));
      cols[k * 3 + 1] = Math.min(1, col[i][1] * v);
      cols[k * 3 + 2] = Math.min(1, col[i][2] * v * (1 - warm) * 0.97);
      k++;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos.subarray(0, k * 3), 3));
    geo.setAttribute('color', new THREE.BufferAttribute(cols.subarray(0, k * 3), 3));
    return { geo, size };
  };

  const hd = quality > 1;
  // névoa: muitas nuvens macias e grandes (o principal de perto); grãos: só um brilho fino por cima
  const gas = make(hd ? 50000 : 18000, hd ? 1.1 : 1.3, 0.1, true, 0.72);
  const dust = make(hd ? 60000 : 22000, hd ? 0.035 : 0.05, 0.03, false, 0.9);
  const dustMat = new THREE.PointsMaterial({ size: dust.size, map: softDot(THREE, true), vertexColors: true, transparent: true, opacity: 0.1, depthWrite: false, sizeAttenuation: true });
  const gasMat = new THREE.PointsMaterial({ size: gas.size, map: softDot(THREE, false), vertexColors: true, transparent: true, opacity: 0.0, depthWrite: false, sizeAttenuation: true });
  gasMat.userData.maxPx = 180; // só 3–6 mil nuvens: podem ficar grandes na tela sem pesar
  const group = new THREE.Group();
  group.add(new THREE.Points(gas.geo, gasMat));
  group.add(new THREE.Points(dust.geo, dustMat));
  return { group, dustMat, gasMat };
}

const smooth = (a, b, x) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

export async function buildPlanet(THREE, spec, quality = 1) {
  const p = spec.planet;
  const R = 5;
  const seg = quality > 1 ? 160 : 72;
  const tex = quality > 1 ? PLANET_TEXTURES_HD : PLANET_TEXTURES;

  const group = new THREE.Group(); // inclinação da vista
  const yaw = new THREE.Group(); // gira o eixo inclinado em direção à câmera (ex.: Urano, para os anéis não ficarem de perfil)
  yaw.rotation.y = ((p.yawDeg ?? 0) * Math.PI) / 180;
  const pivot = new THREE.Group(); // inclinação do eixo do planeta
  pivot.rotation.z = -((p.tiltDeg ?? 0) * Math.PI) / 180;
  yaw.add(pivot);
  group.add(yaw);

  const map = await loadTex(THREE, tex[p.tex]);
  const surface = new THREE.MeshStandardMaterial({ map, roughness: p.rough ?? 0.95, metalness: 0 });
  // relevo: crateras, montanhas e cânions reagem à luz do Sol (sombra de um lado, brilho do outro)
  if (p.normal) {
    surface.normalMap = await loadTex(THREE, tex[p.normal], false);
    surface.normalScale.set(p.normalScale ?? 1, p.normalScale ?? 1);
  }
  // Terra: oceano liso (reflete o Sol), terra firme fosca
  if (p.roughMap) {
    surface.roughnessMap = await loadTex(THREE, tex[p.roughMap], false);
    surface.roughness = 1;
  }
  const planet = new THREE.Mesh(new THREE.SphereGeometry(R, seg, seg / 2), surface);
  pivot.add(planet);

  let clouds = null;
  if (p.clouds) {
    const alpha = await loadTex(THREE, tex[p.clouds], false);
    clouds = new THREE.Mesh(
      new THREE.SphereGeometry(R * 1.012, seg, seg / 2),
      new THREE.MeshStandardMaterial({ color: '#ffffff', alphaMap: alpha, transparent: true, depthWrite: false, roughness: 1 }),
    );
    pivot.add(clouds);
  }

  if (p.atmo) {
    const glow = new THREE.Mesh(new THREE.SphereGeometry(R * (p.atmoSize ?? 1.07), seg, seg / 2), fresnelMaterial(THREE, p.atmo, p.atmoPower ?? 1.6, (p.atmoStrength ?? 1.1) * 1.6));
    glow.userData.noPick = true; // o halo não é alvo do zoom (o alvo é a superfície)
    pivot.add(glow);
  }

  let ringFx = null;
  if (p.rings === 'saturn') {
    const ringTex = await loadTex(THREE, tex.saturnoAneis);
    // material sem sombreamento: a luz do Sol chega quase de lado nos anéis e os deixaria escuros demais
    const mat = new THREE.MeshBasicMaterial({ map: ringTex, color: '#d9d2c4', transparent: true, side: THREE.DoubleSide, depthWrite: false });
    pivot.add(ringMesh(THREE, R * 1.24, R * 2.27, mat));
    // de longe: parece uma placa lisa; de perto: a placa some e ficam grãos e névoa
    const parts = saturnRingParticles(THREE, R * 1.24, R * 2.27, ringProfile(ringTex.image), quality);
    pivot.add(parts.group);
    // no HD o halo de brilho (bloom) clareia a névoa, então ela fica um pouco mais transparente
    ringFx = { mat, ...parts, far: spec.camDist ?? 24, gasMax: quality > 1 ? 0.1 : 0.13 };
  } else if (p.rings === 'uranus') {
    // anéis finos e escuros (proporções aproximadas)
    for (const [r, w, o] of [[1.64, 0.012, 0.14], [1.74, 0.01, 0.12], [2.0, 0.03, 0.22]]) {
      const mat = new THREE.MeshBasicMaterial({ color: '#b9c7d6', transparent: true, opacity: o, side: THREE.DoubleSide, depthWrite: false });
      pivot.add(ringMesh(THREE, R * r, R * (r + w), mat));
    }
  }

  // luz do Sol vindo da esquerda (um pouco de frente) + luz ambiente fraca no lado da noite
  const sun = new THREE.DirectionalLight('#fff6e8', 3.2);
  sun.position.set(...SUN_POS);
  const ambient = new THREE.AmbientLight('#8fa6d8', 0.12);
  const lights = [sun, ambient];

  const bg = starBackground(THREE);
  bg.userData.noPick = true;

  // velocidade de giro proporcional à duração do dia (mínimo visível para Mercúrio e Vênus)
  const spin = Math.max(0.03, Math.min(0.35, 24 / (p.dayHours ?? 24) * 0.12)) * (p.retro ? -1 : 1);

  // o giro acumula o tempo só enquanto está ligado (pausar e voltar não dá "pulo")
  let last = null;
  let angle = 0;
  return {
    group,
    extras: [bg, ...lights],
    initialTilt: spec.tiltOverride ?? 0.12,
    radius: R, // a câmera não pode entrar no planeta
    pickRadius: R * 2.4, // alvos do zoom: planeta e anéis
    camDist: spec.camDist ?? 17,
    animate(t, reduce, running = true, camera = null) {
      const dt = last === null ? 0 : Math.min(0.1, t - last);
      last = t;
      if (ringFx && camera) {
        // 0 = bem perto, 1 = na distância inicial ou mais longe
        const f = smooth(8, ringFx.far * 0.92, camera.position.length());
        ringFx.mat.opacity = 0.03 + 0.97 * f ** 2.2; // a placa lisa some rápido ao aproximar
        ringFx.dustMat.opacity = 0.45 - 0.37 * f;
        ringFx.gasMat.opacity = ringFx.gasMax * (1 - f) ** 0.6;
        ringFx.gasMat.visible = f < 0.995;
      }
      if (reduce || !running) return;
      angle += dt * spin;
      planet.rotation.y = angle;
      if (clouds) clouds.rotation.y = angle * 1.15;
      // as partículas dos anéis dão a volta devagar em torno do planeta
      if (ringFx) ringFx.group.rotation.y = angle * 0.25;
    },
  };
}
