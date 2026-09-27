// Modelos 3D ILUSTRATIVOS de galáxias e nebulosas, feitos de partículas (three.js).
// Não são fotos nem medições exatas em 3D: o formato é aproximado a partir de fotos
// públicas e do tipo de cada objeto. Cada modelo é gerado por código (sem arquivos externos).
//
// buildScene(THREE, spec, qualidade) -> { group, animate(t), initialTilt }
// qualidade: 1 = normal (leve), 3 = HD (3x mais partículas, textura de ruído mais fina)

// ---------------------------------------------------------------- qualidade
// Q multiplica a quantidade de partículas (1 = normal, 3 = HD). O brilho de cada partícula
// é compensado para a imagem não "estourar" quando há mais pontos.
let Q = 1;
const N = (n) => Math.max(1, Math.round(n * Q));

// ---------------------------------------------------------------- ruído (para dar textura de nuvem)
function hash3(x, y, z) {
  let h = (x * 374761393 + y * 668265263 + z * 2147483647) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967295;
}
function valueNoise(x, y, z) {
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const zi = Math.floor(z);
  const xf = x - xi;
  const yf = y - yi;
  const zf = z - zi;
  const s = (t) => t * t * (3 - 2 * t);
  const u = s(xf);
  const v = s(yf);
  const w = s(zf);
  const L = (a, b, t) => a + (b - a) * t;
  const c = (dx, dy, dz) => hash3(xi + dx, yi + dy, zi + dz);
  return L(
    L(L(c(0, 0, 0), c(1, 0, 0), u), L(c(0, 1, 0), c(1, 1, 0), u), v),
    L(L(c(0, 0, 1), c(1, 0, 1), u), L(c(0, 1, 1), c(1, 1, 1), u), v),
    w,
  );
}
/** ruído fractal 0..1 (várias escalas somadas) */
export function fbm(x, y, z, oct = 4) {
  let a = 0.5;
  let f = 1;
  let sum = 0;
  let norm = 0;
  for (let i = 0; i < N(oct); i++) {
    sum += a * valueNoise(x * f, y * f, z * f);
    norm += a;
    a *= 0.5;
    f *= 2.03;
  }
  return sum / norm;
}

// ---------------------------------------------------------------- utilidades
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeTools(THREE, seed) {
  const r = rng(seed);
  const rand = (a = 0, b = 1) => a + r() * (b - a);
  const gauss = () => {
    let u = 0;
    let v = 0;
    while (u === 0) u = r();
    while (v === 0) v = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  const pick = (arr) => arr[Math.floor(r() * arr.length)];
  const color = new THREE.Color();
  return { rand, gauss, pick, color, Color: THREE.Color };
}

const TEX = {};
function sprite(THREE, kind = 'star') {
  if (TEX[kind]) return TEX[kind];
  const c = document.createElement('canvas');
  c.width = c.height = 128;
  const g = c.getContext('2d');
  const grd = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  if (kind === 'gas') {
    grd.addColorStop(0, 'rgba(255,255,255,0.9)');
    grd.addColorStop(0.35, 'rgba(255,255,255,0.35)');
    grd.addColorStop(0.7, 'rgba(255,255,255,0.08)');
  } else {
    grd.addColorStop(0, 'rgba(255,255,255,1)');
    grd.addColorStop(0.12, 'rgba(255,255,255,0.95)');
    grd.addColorStop(0.3, 'rgba(255,255,255,0.35)');
    grd.addColorStop(0.6, 'rgba(255,255,255,0.06)');
  }
  grd.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grd;
  g.fillRect(0, 0, 128, 128);
  TEX[kind] = new THREE.CanvasTexture(c);
  TEX[kind].colorSpace = THREE.SRGBColorSpace;
  return TEX[kind];
}

/** Acumulador de partículas: cada camada vira um THREE.Points. */
class Layer {
  constructor(THREE, { size = 0.12, opacity = 1, dark = false, tex = 'star' } = {}) {
    this.tex = tex;
    this.THREE = THREE;
    this.pos = [];
    this.col = [];
    this.size = size;
    this.opacity = opacity;
    this.dark = dark;
    // mais partículas no HD -> cada uma um pouco menor e mais fraca
    // gás compensa quase tudo (senão a nuvem fica branca demais); estrelas ficam um pouco mais vivas
    this.kScale = 1 / Math.pow(Q, tex === 'gas' ? 0.9 : 0.5);
    this.size = size / Math.pow(Q, 0.22);
    this.opacity = dark ? opacity / Math.pow(Q, 0.35) : opacity;
  }
  add(x, y, z, c, k = 1) {
    this.pos.push(x, y, z);
    const kk = this.dark ? 1 : k * this.kScale;
    this.col.push(c.r * kk, c.g * kk, c.b * kk);
  }
  build() {
    const THREE = this.THREE;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(this.pos, 3));
    geo.setAttribute('color', new THREE.Float32BufferAttribute(this.col, 3));
    const mat = new THREE.PointsMaterial({
      size: this.size,
      map: sprite(THREE, this.tex),
      vertexColors: true,
      transparent: true,
      opacity: this.opacity,
      depthWrite: false,
      sizeAttenuation: true,
      blending: this.dark ? THREE.NormalBlending : THREE.AdditiveBlending,
    });
    return new THREE.Points(geo, mat);
  }
}

// ---------------------------------------------------------------- galáxias
function addSpiral(T, L, o) {
  const {
    arms = 2,
    R = 10,
    turns = 0.85,
    spread = 0.38,
    bulge = 1.6,
    bar = 0,
    thick = 0.25,
    count = 26000,
    armColors = ['#9cc3ff', '#b9a6ff', '#e8f0ff', '#7fb2ff'],
    core = '#ffe2b0',
    knots = 90,
    dust = 0.55,
    cx = 0,
    cz = 0,
    rot = 0,
    scale = 1,
    tiltX = 0,
  } = o;
  const { rand, gauss, pick, color } = T;
  const stars = L.stars;
  const put = (layer, x, y, z, c, k) => {
    // aplica escala, rotação no plano, inclinação e deslocamento
    let X = x * scale;
    let Y = y * scale;
    let Z = z * scale;
    const cr = Math.cos(rot);
    const sr = Math.sin(rot);
    [X, Z] = [X * cr - Z * sr, X * sr + Z * cr];
    if (tiltX) {
      const ct = Math.cos(tiltX);
      const st = Math.sin(tiltX);
      [Y, Z] = [Y * ct - Z * st, Y * st + Z * ct];
    }
    layer.add(X + cx, Y, Z + cz, c, k);
  };
  // braços
  for (let i = 0; i < N(count); i++) {
    const arm = i % arms;
    const t = Math.pow(rand(), 0.75);
    const r = bar + t * (R - bar) + gauss() * 0.2;
    const a = arm * ((Math.PI * 2) / arms) + t * turns * Math.PI * 2 + gauss() * spread * (1 - t * 0.4);
    const y = gauss() * thick * (1 - t * 0.5);
    // centro amarelado, braços azulados; brilho com manchas (aglomerados de estrelas)
    const ax = Math.cos(a) * r;
    const az = Math.sin(a) * r;
    color.set(t < 0.12 ? core : pick(armColors));
    if (t > 0.12 && t < 0.3) color.lerp(new T.Color(core), 0.45);
    const clump = 0.45 + 1.2 * fbm(ax * 0.45 + 3, y, az * 0.45, 3);
    put(stars, ax, y, az, color, rand(0.35, 1) * clump);
  }
  // disco difuso entre os braços
  for (let i = 0; i < N(count * 0.35); i++) {
    const r = Math.abs(gauss()) * R * 0.45;
    const a = rand(0, Math.PI * 2);
    color.set(pick(['#8ea6d8', '#c9c2ff', '#fff1d6']));
    put(stars, Math.cos(a) * r, gauss() * thick, Math.sin(a) * r, color, rand(0.12, 0.35));
  }
  // barra
  if (bar > 0) {
    for (let i = 0; i < N(count * 0.18); i++) {
      const x = rand(-bar, bar);
      color.set(pick(['#ffe2b0', '#fff4e0', '#ffd08a']));
      put(stars, x, gauss() * thick * 0.8, gauss() * bar * 0.14, color, rand(0.4, 0.9));
    }
  }
  // bojo central
  for (let i = 0; i < N(count * 0.22); i++) {
    const rr = Math.abs(gauss()) * bulge;
    const th = rand(0, Math.PI * 2);
    const ph = Math.acos(rand(-1, 1));
    color.set(pick(['#ffe2b0', '#fff4e0', '#ffd08a', '#ffffff']));
    put(stars, rr * Math.sin(ph) * Math.cos(th), rr * Math.cos(ph) * 0.75, rr * Math.sin(ph) * Math.sin(th), color, rand(0.5, 1));
  }
  // regiões rosadas de estrelas nascendo
  for (let k = 0; k < N(knots); k++) {
    const arm = k % arms;
    const t = rand(0.3, 1);
    const r = bar + t * (R - bar);
    const a = arm * ((Math.PI * 2) / arms) + t * turns * Math.PI * 2 + gauss() * spread * 0.4;
    const n = Math.floor(rand(6, 16));
    for (let j = 0; j < N(n); j++) {
      color.set(pick(['#ff7ab8', '#ff9fd0', '#ff6f91']));
      put(L.knots, Math.cos(a) * r + gauss() * 0.12, gauss() * 0.06, Math.sin(a) * r + gauss() * 0.12, color, rand(0.5, 1));
    }
  }
  // faixas de poeira (escuras) na borda interna dos braços
  if (dust > 0) {
    for (let i = 0; i < N(count * 0.3 * dust); i++) {
      const arm = i % arms;
      const t = Math.pow(rand(), 0.8);
      const r = bar + t * (R - bar);
      const a = arm * ((Math.PI * 2) / arms) + t * turns * Math.PI * 2 - 0.22 + gauss() * 0.08;
      color.set('#07060d');
      put(L.dust, Math.cos(a) * r, gauss() * thick * 0.4, Math.sin(a) * r, color, 1);
    }
  }
}

function addElliptical(T, L, { R = 6, axis = [1, 0.85, 0.75], count = 30000, jet = false, globs = 250 } = {}) {
  const { rand, gauss, pick, color } = T;
  for (let i = 0; i < N(count); i++) {
    const s = Math.abs(gauss()) * R * 0.45 * (rand() < 0.2 ? 1.8 : 1);
    const th = rand(0, Math.PI * 2);
    const ph = Math.acos(rand(-1, 1));
    color.set(pick(['#ffd9a0', '#ffe6c2', '#ffcf8a', '#fff3e0', '#ffb877']));
    L.stars.add(s * Math.sin(ph) * Math.cos(th) * axis[0], s * Math.cos(ph) * axis[1], s * Math.sin(ph) * Math.sin(th) * axis[2], color, rand(0.3, 0.9));
  }
  // aglomerados globulares (pontinhos em volta)
  for (let i = 0; i < N(globs); i++) {
    const s = rand(0.6, 1.6) * R;
    const th = rand(0, Math.PI * 2);
    const ph = Math.acos(rand(-1, 1));
    color.set('#fff2d8');
    L.bright.add(s * Math.sin(ph) * Math.cos(th), s * Math.cos(ph), s * Math.sin(ph) * Math.sin(th), color, rand(0.4, 0.9));
  }
  if (jet) {
    // jato de matéria saindo do buraco negro central (M87)
    for (let i = 0; i < N(2500); i++) {
      const x = Math.pow(rand(), 0.8) * R * 0.9;
      color.set(pick(['#5b8cff', '#8fd3ff', '#b7e6ff']));
      L.knots.add(x, gauss() * 0.06 * (1 + x * 0.15), gauss() * 0.06 * (1 + x * 0.15), color, rand(0.2, 0.55));
    }
  }
}

function addDustRing(T, L, { r0, r1, thick = 0.06, count = 7000, y = 0 }) {
  const { rand, gauss, color } = T;
  color.set('#07060d');
  for (let i = 0; i < N(count); i++) {
    const r = rand(r0, r1);
    const a = rand(0, Math.PI * 2);
    L.dust.add(Math.cos(a) * r, y + gauss() * thick, Math.sin(a) * r, color, 1);
  }
}

function addTail(T, L, { from, ctrl, to, count = 5000, width = 0.35, colors = ['#9cc3ff', '#c4b5fd', '#e8f0ff'] }) {
  const { rand, gauss, pick, color } = T;
  for (let i = 0; i < N(count); i++) {
    const t = rand();
    const u = 1 - t;
    const x = u * u * from[0] + 2 * u * t * ctrl[0] + t * t * to[0];
    const y = u * u * from[1] + 2 * u * t * ctrl[1] + t * t * to[1];
    const z = u * u * from[2] + 2 * u * t * ctrl[2] + t * t * to[2];
    const w = width * (0.4 + t);
    color.set(pick(colors));
    L.stars.add(x + gauss() * w, y + gauss() * w * 0.5, z + gauss() * w, color, rand(0.2, 0.7) * (1 - t * 0.5));
  }
}

function addClumps(T, L, { n = 14, R = 6, flat = 0.4, colors, pinks = true }) {
  const { rand, gauss, pick, color } = T;
  for (let k = 0; k < n; k++) {
    const cx = gauss() * R * 0.45;
    const cy = gauss() * R * 0.45 * flat;
    const cz = gauss() * R * 0.45;
    const size = rand(0.5, 1.6);
    const cnt = Math.floor(rand(600, 1800));
    for (let i = 0; i < N(cnt); i++) {
      color.set(pick(colors));
      L.stars.add(cx + gauss() * size, cy + gauss() * size * flat, cz + gauss() * size, color, rand(0.3, 0.9));
    }
    if (pinks && rand() < 0.6) {
      for (let i = 0; i < N(40); i++) {
        color.set(pick(['#ff7ab8', '#ff9fd0']));
        L.knots.add(cx + gauss() * 0.3, cy + gauss() * 0.2, cz + gauss() * 0.3, color, 1);
      }
    }
  }
}

// ---------------------------------------------------------------- nebulosas
function addCloud(T, L, { blobs = 10, R = 6, count = 30000, colors, flat = 0.6, seedShift = 0 }) {
  const { rand, gauss, pick, color } = T;
  const centers = Array.from({ length: blobs }, () => [gauss() * R * 0.5, gauss() * R * 0.5 * flat, gauss() * R * 0.4, rand(0.6, 1.8) * (R / 6)]);
  const off = rand(0, 100);
  const total = N(count);
  let placed = 0;
  let tries = 0;
  while (placed < total && tries < total * 6) {
    tries++;
    const c = centers[tries % blobs];
    const sc = c[3] * (1 + seedShift);
    const x = c[0] + gauss() * sc * 1.4;
    const y = c[1] + gauss() * sc;
    const z = c[2] + gauss() * sc;
    // mantém mais pontos onde o ruído é alto: cria fios, bordas e vazios, como nas fotos
    const n = fbm(x * 0.32 + off, y * 0.32, z * 0.32, 5);
    if (rand() > Math.min(1, Math.max(0, (n - 0.38) * 3.2))) continue;
    color.set(pick(colors));
    L.gas.add(x, y, z, color, rand(0.12, 0.45) * (0.35 + n));
    placed++;
  }
}

function addStars(T, L, { n, R, colors = ['#ffffff', '#cfe3ff', '#a8c8ff'], k = [0.7, 1], layer = 'bright', center = [0, 0, 0] }) {
  const { rand, gauss, pick, color } = T;
  for (let i = 0; i < N(n); i++) {
    color.set(pick(colors));
    L[layer].add(center[0] + gauss() * R, center[1] + gauss() * R, center[2] + gauss() * R, color, rand(k[0], k[1]));
  }
}

function addShell(T, L, { R = 5, ax = [1, 1, 1], thick = 0.12, count = 20000, colors, arc = Math.PI * 2, filaments = false, layer = 'gas', k = [0.3, 0.9], offset = [0, 0, 0] }) {
  const { rand, gauss, pick, color } = T;
  if (!filaments) {
    for (let i = 0; i < N(count); i++) {
      const th = rand(0, arc);
      const ph = Math.acos(rand(-1, 1));
      const r = R * (1 + gauss() * thick);
      color.set(pick(colors));
      const px = r * Math.sin(ph) * Math.cos(th);
      const py = r * Math.cos(ph);
      const pz = r * Math.sin(ph) * Math.sin(th);
      const patch = 0.35 + 1.1 * fbm(px * 0.5 + 11, py * 0.5, pz * 0.5, 4);
      L[layer].add(offset[0] + px * ax[0], offset[1] + py * ax[1], offset[2] + pz * ax[2], color, rand(k[0], k[1]) * patch);
    }
    return;
  }
  // fios: pedaços de arcos sobre a casca
  const nFil = Math.floor(count / 300);
  for (let f = 0; f < N(nFil); f++) {
    let th = rand(0, arc);
    let ph = Math.acos(rand(-1, 1));
    const dth = gauss() * 0.02;
    const dph = gauss() * 0.02;
    color.set(pick(colors));
    const c = color.clone();
    for (let i = 0; i < 300; i++) {
      th += dth + gauss() * 0.004;
      ph += dph + gauss() * 0.004;
      const r = R * (1 + gauss() * thick * 0.4);
      L[layer].add(offset[0] + r * Math.sin(ph) * Math.cos(th) * ax[0], offset[1] + r * Math.cos(ph) * ax[1], offset[2] + r * Math.sin(ph) * Math.sin(th) * ax[2], c, rand(k[0], k[1]));
    }
  }
}

function addTorus(T, L, { R = 4, tube = 1, count = 20000, inner = ['#5eead4', '#8fe3ff'], outer = ['#ff6f91', '#ff9f7a'], tilt = 0 }) {
  const { rand, gauss, pick, color } = T;
  const ct = Math.cos(tilt);
  const st = Math.sin(tilt);
  for (let i = 0; i < N(count); i++) {
    const u = rand(0, Math.PI * 2);
    const v = rand(0, Math.PI * 2);
    const tr = tube * Math.sqrt(rand());
    const rr = R + tr * Math.cos(v);
    const x = rr * Math.cos(u);
    const z = rr * Math.sin(u);
    const y = tr * Math.sin(v) * 1.4;
    const outerPart = tr * Math.cos(v) > 0;
    color.set(pick(outerPart ? outer : inner));
    L.gas.add(x, y * ct - z * st, y * st + z * ct, color, rand(0.1, 0.45));
  }
}

function addPillar(T, L, { x, z, h, w, tip = 1 }) {
  const { rand, gauss, pick, color } = T;
  for (let i = 0; i < N(9000 * (h / 8)); i++) {
    const t = rand();
    const y = -h / 2 + t * h;
    const ww = w * (1 - t * 0.55) * tip;
    const px = x + gauss() * ww * 0.5 + Math.sin(t * 3 + x) * 0.3;
    const pz = z + gauss() * ww * 0.5;
    color.set('#1a0f0c');
    L.dust.add(px, y, pz, color, 1);
    if (rand() < 0.14) {
      // bordas iluminadas pelas estrelas
      color.set(pick(['#ffb36b', '#ffd39a', '#ff8a5c']));
      L.gas.add(px + ww * 0.55, y + rand(0, 0.4), pz + ww * 0.3, color, rand(0.4, 0.9));
    }
  }
}

// Pilares da Criação em alta definição: colunas de poeira com a superfície moldada por ruído,
// borda iluminada do lado das estrelas (em cima, à direita), fios de gás "evaporando" nas pontas.
function addPillarHD(T, L, { x0, z0, h, w, lean = 0, phase = 0 }) {
  const { rand, gauss, pick, color } = T;
  const light = [0.62, 0.35, 0.7]; // direção de onde vem a luz das estrelas jovens
  const total = N(26000 * (h / 10));
  for (let i = 0; i < total; i++) {
    const t = Math.pow(rand(), 0.85); // 0 = base, 1 = ponta
    const y = -h / 2 + t * h;
    const cx = x0 + lean * t * t + Math.sin(t * 4 + phase) * 0.35;
    const cz = z0 + Math.cos(t * 3 + phase) * 0.2;
    const a = rand(0, Math.PI * 2);
    // superfície irregular (calombos e reentrâncias), como nas fotos
    const bump = 0.6 + 0.8 * fbm(Math.cos(a) * 1.8 + phase, y * 0.9, Math.sin(a) * 1.8, 5);
    const tipRound = t > 0.86 ? Math.sqrt(Math.max(0, 1 - ((t - 0.86) / 0.14) ** 2)) : 1;
    const R = w * (1 - 0.5 * t) * bump * (0.35 + 0.65 * tipRound);
    const rr = R * Math.sqrt(rand());
    const nx = Math.cos(a);
    const nz = Math.sin(a);
    const x = cx + nx * rr;
    const z = cz + nz * rr;
    const surface = rr > R * 0.78;
    const lit = Math.max(0, nx * light[0] + nz * light[2]) + (t > 0.8 ? (t - 0.8) * 2.5 : 0) * light[1];
    if (surface && lit > 0.08) {
      color.set(pick(t > 0.75 ? ['#ffe0a8', '#ffd08a', '#ffb36b', '#fff1d6'] : ['#ff9a5c', '#ffb36b', '#e8844f', '#ffc98a']));
      L.gas.add(x, y, z, color, Math.min(1, lit) * rand(0.3, 0.8));
    } else {
      // corpo escuro da coluna (duas camadas deixam a silhueta bem marcada contra o fundo)
      color.set(rand() < 0.5 ? '#140a07' : '#2a150c');
      L.dust.add(x, y, z, color, 1);
      if (!surface) L.dust.add(x, y, z + 0.3, color, 1);
    }
  }
  // fios de gás saindo das pontas, empurrados pela luz
  const tipX = x0 + lean;
  for (let i = 0; i < N(3500 * (h / 10)); i++) {
    const up = Math.pow(rand(), 1.8) * h * 0.16;
    color.set(pick(['#9ff3e6', '#c8fff6', '#8fe3ff']));
    L.gas.add(tipX + gauss() * w * 0.22 + up * 0.35, h / 2 + up - w * 0.35, z0 + gauss() * w * 0.2, color, rand(0.04, 0.18) * (1 - up / (h * 0.16)));
  }
  // pequenas estrelas nascendo nas pontas
  for (let i = 0; i < 3; i++) {
    color.set('#fff6e0');
    L.bright.add(tipX + gauss() * w * 0.3, h / 2 - rand(0, h * 0.15), z0 + gauss() * 0.3, color, rand(0.5, 0.9));
  }
}

// ---------------------------------------------------------------- montagem
export function buildScene(THREE, spec, quality = 1) {
  Q = quality;
  const T = makeTools(THREE, spec.seed ?? 7);
  const L = {
    stars: new Layer(THREE, { size: 0.12 }),
    bright: new Layer(THREE, { size: 0.32 }),
    knots: new Layer(THREE, { size: 0.16 }),
    gas: new Layer(THREE, { size: 0.6, opacity: 0.6, tex: 'gas' }),
    dust: new Layer(THREE, { size: 0.28, opacity: 0.42, dark: true }),
    bg: new Layer(THREE, { size: 0.35 }),
  };
  const glows = []; // estrelas brilhantes com halo: { pos, color, size, kind }
  let marker = null;
  const { rand, gauss, pick, color } = T;

  switch (spec.kind) {
    case 'spiral':
      addSpiral(T, L, spec.p ?? {});
      break;
    case 'barred':
      addSpiral(T, L, { bar: 3, turns: 0.55, spread: 0.3, ...(spec.p ?? {}) });
      break;
    case 'milkyway':
      addSpiral(T, L, { arms: 4, bar: 2.4, turns: 0.7, spread: 0.32, R: 11, bulge: 1.4, ...(spec.p ?? {}) });
      // o Sol fica a ~26 mil anos-luz do centro (galáxia com ~100 mil de diâmetro)
      marker = new THREE.Mesh(new THREE.SphereGeometry(0.16, 16, 16), new THREE.MeshBasicMaterial({ color: '#ffe14d' }));
      marker.position.set(Math.cos(2.2) * 5.7, 0.05, Math.sin(2.2) * 5.7);
      break;
    case 'elliptical':
      addElliptical(T, L, spec.p ?? {});
      break;
    case 'lenticular': {
      addSpiral(T, L, { arms: 6, turns: 0.2, spread: 1.4, knots: 0, dust: 0, thick: 0.18, bulge: 2, armColors: ['#ffe6c2', '#fff3e0', '#e8ecff'], ...(spec.p ?? {}) });
      addDustRing(T, L, { r0: 1.5, r1: 7.5, thick: 0.05, count: 9000 });
      break;
    }
    case 'sombrero':
      addSpiral(T, L, { arms: 2, turns: 1.6, spread: 0.6, R: 9, bulge: 3.4, thick: 0.2, knots: 20, dust: 0.3, armColors: ['#ffe6c2', '#f6e3cf', '#e8ecff'] });
      addDustRing(T, L, { r0: 6.2, r1: 8.2, thick: 0.1, count: 16000 });
      break;
    case 'needle':
      addSpiral(T, L, { arms: 2, turns: 0.9, R: 11, bulge: 1.3, thick: 0.18, ...(spec.p ?? {}) });
      addDustRing(T, L, { r0: 1.2, r1: 10.5, thick: 0.07, count: 14000 });
      break;
    case 'whirlpool':
      addSpiral(T, L, { arms: 2, turns: 0.9, spread: 0.3, R: 9, knots: 160 });
      addSpiral(T, L, { arms: 2, turns: 0.3, spread: 1, R: 2.4, bulge: 1, count: 6000, knots: 0, dust: 0.2, cx: 9, cz: 5.5, armColors: ['#ffe2b0', '#ffd08a'] });
      addTail(T, L, { from: [3.5, 0, 8], ctrl: [8, 0, 9], to: [9, 0, 5.5], count: 3000, width: 0.3 });
      break;
    case 'antennae':
      addSpiral(T, L, { R: 4.5, count: 12000, turns: 0.5, cx: -2.2, cz: 0.5, rot: 0.3, knots: 120 });
      addSpiral(T, L, { R: 4, count: 11000, turns: 0.5, cx: 2.4, cz: -0.4, rot: 2.4, tiltX: 0.6, knots: 120 });
      addTail(T, L, { from: [-3, 0, 2], ctrl: [-9, 1, 9], to: [-2, 2, 16], count: 7000 });
      addTail(T, L, { from: [3, 0, -1], ctrl: [10, -1, -8], to: [3, -2, -15], count: 7000 });
      break;
    case 'mice':
      addSpiral(T, L, { R: 3.2, count: 9000, cx: -2.5, cz: 0, tiltX: 1.1, knots: 60 });
      addSpiral(T, L, { R: 3, count: 9000, cx: 2.5, cz: 0.5, rot: 1.2, knots: 60 });
      addTail(T, L, { from: [-2.5, 1, 1], ctrl: [-4, 5, 7], to: [-3, 1, 15], count: 7000, width: 0.25 });
      addTail(T, L, { from: [2.5, 0, 0.5], ctrl: [6, -2, -4], to: [9, -1, -6], count: 4000, width: 0.3 });
      break;
    case 'lmc':
      addSpiral(T, L, { arms: 1, bar: 3, turns: 0.35, spread: 0.9, R: 7, bulge: 0.2, count: 14000, knots: 60, dust: 0.1 });
      addClumps(T, L, { n: 8, R: 7, flat: 0.25, colors: ['#9cc3ff', '#e8f0ff', '#c4b5fd'] });
      // região da Tarântula (brilho rosa forte)
      for (let i = 0; i < N(1500); i++) {
        color.set(pick(['#ff7ab8', '#ff9fd0', '#ffffff']));
        L.knots.add(2.6 + gauss() * 0.5, gauss() * 0.2, 1.5 + gauss() * 0.5, color, rand(0.5, 1));
      }
      break;
    case 'cigar':
      // disco visto de lado com ventos avermelhados saindo do centro
      addElliptical(T, L, { R: 7, axis: [1.5, 0.28, 0.5], count: 22000, globs: 80 });
      addDustRing(T, L, { r0: 0.5, r1: 7, thick: 0.1, count: 6000 });
      for (let i = 0; i < N(12000); i++) {
        const y = (rand() < 0.5 ? -1 : 1) * Math.pow(rand(), 0.7) * 6;
        const w = 0.4 + Math.abs(y) * 0.28;
        color.set(pick(['#ff4d6d', '#ff6f91', '#ff9f7a']));
        L.gas.add(gauss() * w, y, gauss() * w * 0.6, color, rand(0.2, 0.6) * (1 - Math.abs(y) / 7));
      }
      break;

    case 'emission': {
      const p = spec.p ?? {};
      addCloud(T, L, { colors: p.colors ?? ['#ff4d6d', '#ff6f91', '#ff9fb8', '#ffb3c7', '#5eead4'], blobs: p.blobs ?? 12, R: p.R ?? 7, count: p.count ?? 34000 });
      addStars(T, L, { n: p.youngStars ?? 30, R: 0.8, colors: ['#ffffff', '#cfe3ff'] });
      if (p.trapezium) for (const [x, y] of [[-0.3, 0.2], [0.25, 0.35], [0.1, -0.25], [-0.15, -0.1]]) glows.push({ kind: 'twinkle', pos: [x, y, 0.6], color: '#e8f0ff', size: 1.1 });
      if (p.darkLane) {
        for (let i = 0; i < N(8000); i++) {
          color.set('#120a10');
          L.dust.add(rand(-7, 7), gauss() * 0.5 + Math.sin(i) * 0.1, gauss() * 1.2 + 1, color, 1);
        }
      }
      break;
    }
    case 'eagle':
      addCloud(T, L, { colors: ['#5eead4', '#8fe3ff', '#ffd39a', '#ffb36b'], blobs: 10, R: 9, count: 26000, flat: 0.9 });
      addPillar(T, L, { x: -2.6, z: 0, h: 9, w: 1.7 });
      addPillar(T, L, { x: 0.8, z: 0.8, h: 6.5, w: 1.2 });
      addPillar(T, L, { x: 3, z: -0.4, h: 4.5, w: 0.9 });
      addStars(T, L, { n: 12, R: 4, center: [4, 5, 2] });
      break;
    case 'pillars': {
      // fundo: gás azul-esverdeado em cima e dourado embaixo, com textura de nuvem e bordas suaves
      {
        const teal = new T.Color('#3fd9c4');
        const blue = new T.Color('#6fa8ff');
        const gold = new T.Color('#e0a95c');
        const amber = new T.Color('#b9783f');
        for (let i = 0; i < N(70000); i++) {
          const x = gauss() * 7.5;
          const y = gauss() * 6 + 1;
          const z = -3.2 + gauss() * 0.9;
          const n = fbm(x * 0.2 + 5, y * 0.2, z * 0.3, 5);
          if (rand() > Math.min(1, Math.max(0, (n - 0.42) * 4))) continue;
          const mix = Math.min(1, Math.max(0, (y + 4) / 11)); // 0 = embaixo (dourado), 1 = em cima (azul-esverdeado)
          color.copy(gold).lerp(amber, rand()).lerp(rand() < 0.5 ? teal : blue, mix);
          const nearCenter = Math.exp(-(x * x + (y - 1) * (y - 1)) / 60); // mais brilho atrás das colunas
          L.gas.add(x, y, z, color, rand(0.12, 0.4) * (0.4 + n) * (0.6 + nearCenter));
        }
      }
      addPillarHD(T, L, { x0: -3.8, z0: 0.6, h: 13, w: 2.3, lean: 1.2, phase: 0.3 });
      addPillarHD(T, L, { x0: 1.6, z0: -0.2, h: 8.5, w: 1.6, lean: 0.4, phase: 2.1 });
      addPillarHD(T, L, { x0: 5, z0: -0.8, h: 6, w: 1.15, lean: -0.3, phase: 4.2 });
      // base escura que une as colunas
      for (let i = 0; i < N(16000); i++) {
        color.set('#140a07');
        L.dust.add(rand(-9, 9) + gauss() * 0.5, -6.8 + gauss() * 0.9 - Math.abs(gauss()) * 0.6, gauss() * 1.2, color, 1);
      }
      // estrelas jovens que iluminam tudo (fora da imagem, em cima à direita) e estrelas soltas
      glows.push({ kind: 'twinkle', pos: [10, 9, 6], color: '#dbe7ff', size: 2.6 });
      glows.push({ kind: 'twinkle', pos: [7.5, 6.5, 3], color: '#fff1d6', size: 1.6 });
      glows.push({ kind: 'twinkle', pos: [-9, -3, 2], color: '#cfe0ff', size: 1.2 });
      addStars(T, L, { n: 90, R: 7, colors: ['#ffffff', '#ffe6c2', '#cfe3ff'], k: [0.3, 0.8], layer: 'stars', center: [0, 1, 1] });
      break;
    }
    case 'pleiades':
      for (const [x, y, z, sz] of [[-2.6, 1.2, 0, 2.4], [-0.6, 0.4, 0.5, 2], [0.8, 1.6, -0.3, 1.8], [2.2, -0.2, 0.2, 2.6], [1.4, -1.6, 0.4, 1.6], [-1.4, -1.2, -0.2, 1.7], [3.4, 1.1, 0, 1.5], [-3.6, -0.6, 0.3, 1.4]]) {
        glows.push({ kind: 'twinkle', pos: [x, y, z], color: '#cfe0ff', size: sz });
      }
      addStars(T, L, { n: 140, R: 3.5, colors: ['#cfe3ff', '#ffffff', '#9cc3ff'], k: [0.2, 0.6], layer: 'stars' });
      for (let i = 0; i < N(24000); i++) {
        // fiapos de poeira azul, todos na mesma direção
        const s = rand(-6, 6);
        color.set(pick(['#5b8cff', '#7fb2ff', '#9cc3ff', '#b9a6ff']));
        L.gas.add(s + gauss() * 0.5, s * 0.35 + gauss() * 1.6, gauss() * 1.2, color, rand(0.04, 0.2));
      }
      break;
    case 'witchhead':
      for (let i = 0; i < N(26000); i++) {
        // perfil curvo (queixo, nariz, testa)
        const t = rand(0, 1);
        const x = -6 + t * 10;
        const y = Math.sin(t * Math.PI * 1.6) * 2 + (t > 0.6 ? (t - 0.6) * 5 : 0);
        color.set(pick(['#5b8cff', '#7fb2ff', '#9cc3ff', '#dbe7ff']));
        L.gas.add(x + gauss() * 0.4, y + gauss() * 0.9, gauss() * 0.8, color, rand(0.1, 0.45));
      }
      addStars(T, L, { n: 1, R: 0.01, center: [8, 5, 1], colors: ['#dbe7ff'], k: [1, 1] });
      glows.push({ kind: 'glow', pos: [8, 5, 1], color: '#bcd4ff', size: 3 });
      break;
    case 'horsehead': {
      // "parede" vermelha brilhante atrás e a silhueta escura na frente
      for (let i = 0; i < N(30000); i++) {
        color.set(pick(['#ff4d6d', '#ff6f91', '#e0435f', '#ff9fb8']));
        const wx = gauss() * 4.5;
        const wy = gauss() * 3.5 + 1;
        L.gas.add(wx, wy, -1.5 + gauss() * 0.5, color, rand(0.15, 0.5));
      }
      const inHorse = (x, y) => {
        const neck = Math.abs(x + 0.2) < 1.1 - y * 0.05 && y > -7 && y < 1.5;
        const head = (x - 0.3) ** 2 / 2.2 + (y - 2.4) ** 2 / 1.4 < 1;
        const snout = (x - 1.9) ** 2 / 1.3 + (y - 1.6) ** 2 / 0.5 < 1;
        const ear = (x + 0.4) ** 2 / 0.12 + (y - 3.8) ** 2 / 0.5 < 1;
        const base = y < -4.5 && Math.abs(x) < 9;
        return neck || head || snout || ear || base;
      };
      let placed = 0;
      while (placed < N(40000)) {
        const x = rand(-9, 9);
        const y = rand(-7, 5);
        if (!inHorse(x, y)) continue;
        color.set('#050204');
        L.dust.add(x, y, gauss() * 0.35, color, 1);
        L.dust.add(x + gauss() * 0.05, y + gauss() * 0.05, 0.4, color, 1);
        placed++;
      }
      addStars(T, L, { n: 60, R: 5, colors: ['#ffffff', '#cfe3ff'], k: [0.3, 0.8], layer: 'stars', center: [0, 0, 1] });
      break;
    }
    case 'coalsack': {
      // faixa da Via Láctea cheia de estrelas, com a mancha escura; ao lado, as 4 estrelas do Cruzeiro do Sul
      for (let i = 0; i < N(38000); i++) {
        const x = rand(-12, 12);
        const y = x * 0.25 + gauss() * 3;
        const z = gauss() * 1.5;
        const d = Math.hypot(x - 1, (y - 0.25) * 1.2);
        if (d < 3.2 && rand() < 0.93) continue; // o Saco de Carvão esconde as estrelas de trás
        color.set(pick(['#ffffff', '#fff3e0', '#cfe3ff', '#ffd9a0']));
        L.stars.add(x, y, z, color, rand(0.5, 1));
        if (i % 3 === 0 && !(d < 3.4)) {
          color.set(pick(['#8ea6d8', '#c9c2ff', '#ffe6c2']));
          L.gas.add(x, y, z - 1, color, rand(0.05, 0.15));
        }
      }
      for (let i = 0; i < N(9000); i++) {
        color.set('#050308');
        L.dust.add(1 + gauss() * 1.5, 0.25 + gauss() * 1.2, 1 + gauss() * 0.4, color, 1);
      }
      const cross = [
        [-4.2, 5.2],
        [-4.6, 1.6],
        [-6.2, 3.6],
        [-2.8, 3.2],
      ];
      for (const [x, y] of cross) glows.push({ kind: 'twinkle', pos: [x, y, 2], color: '#dbe7ff', size: 1.8 });
      break;
    }
    case 'ring':
      addTorus(T, L, { R: 3.4, tube: 1.3, count: 30000, tilt: 0.15 });
      addShell(T, L, { R: 3, ax: [1, 1.6, 1], thick: 0.25, count: 6000, colors: ['#5eead4', '#7fdbff'], k: [0.02, 0.08] });
      glows.push({ kind: 'glow', pos: [0, 0, 0], color: '#ffffff', size: 1.2 });
      break;
    case 'helix':
      addTorus(T, L, { R: 4.2, tube: 1.5, count: 26000, inner: ['#5eead4', '#8fe3ff', '#9cc3ff'], outer: ['#ff6f91', '#ff4d6d', '#ffb36b'], tilt: 0.2 });
      addTorus(T, L, { R: 3.4, tube: 0.9, count: 14000, inner: ['#8fe3ff'], outer: ['#ff9fb8'], tilt: 1.35 });
      glows.push({ kind: 'glow', pos: [0, 0, 0], color: '#ffffff', size: 1.2 });
      break;
    case 'catseye':
      addShell(T, L, { R: 2.6, ax: [1, 1.8, 1], thick: 0.08, count: 12000, colors: ['#5eead4', '#8fe3ff'] });
      addShell(T, L, { R: 3.4, ax: [1.7, 1, 1], thick: 0.08, count: 12000, colors: ['#ff6f91', '#ff9fb8'] });
      addShell(T, L, { R: 5.5, ax: [1, 1, 1], thick: 0.25, count: 9000, colors: ['#ff4d6d', '#b9a6ff'], k: [0.05, 0.25] });
      glows.push({ kind: 'glow', pos: [0, 0, 0], color: '#ffffff', size: 1 });
      break;
    case 'crab':
      addShell(T, L, { R: 4.5, ax: [1.4, 1, 1], thick: 0.35, count: 24000, colors: ['#ff6f3c', '#ff4d6d', '#ffb36b', '#ff8a5c'], filaments: true, k: [0.12, 0.45] });
      addShell(T, L, { R: 3.2, ax: [1.4, 1, 1], thick: 0.5, count: 9000, colors: ['#5b8cff', '#7fb2ff', '#8b5cf6'], k: [0.04, 0.14] });
      glows.push({ kind: 'pulsar', pos: [0, 0, 0], color: '#e8f0ff', size: 1.4 });
      break;
    case 'veil':
      addShell(T, L, { R: 7, thick: 0.08, count: 30000, colors: ['#2dd4bf', '#5eead4', '#ff4d6d', '#e0435f', '#5b8cff'], filaments: true, arc: Math.PI * 1.6, k: [0.1, 0.45] });
      break;
    case 'bubble':
      addShell(T, L, { R: 4, thick: 0.05, count: 22000, colors: ['#8fe3ff', '#5eead4', '#b9e6ff'], k: [0.15, 0.6] });
      addCloud(T, L, { colors: ['#ff4d6d', '#ff6f91', '#e0435f'], blobs: 6, R: 12, count: 16000, flat: 0.8 });
      glows.push({ kind: 'glow', pos: [1.2, 0.6, 0.4], color: '#ffffff', size: 1.8 });
      break;
    case 'crescent':
      addShell(T, L, { R: 5, ax: [1.3, 1, 0.9], thick: 0.12, count: 22000, colors: ['#ff4d6d', '#e0435f', '#2dd4bf', '#5eead4'], filaments: true, arc: Math.PI * 1.35, k: [0.12, 0.5] });
      glows.push({ kind: 'glow', pos: [0, 0, 0], color: '#ffffff', size: 1.5 });
      break;
    default:
      addSpiral(T, L, {});
  }

  // estrelas de fundo, bem longe
  for (let i = 0; i < N(1500); i++) {
    const r = rand(40, 70);
    const th = rand(0, Math.PI * 2);
    const ph = Math.acos(rand(-1, 1));
    color.set(pick(['#ffffff', '#cfe3ff', '#ffe6c2']));
    L.bg.add(r * Math.sin(ph) * Math.cos(th), r * Math.cos(ph), r * Math.sin(ph) * Math.sin(th), color, rand(0.3, 0.8));
  }

  const group = new THREE.Group();
  const body = new THREE.Group(); // o objeto em si (gira devagar); as estrelas de fundo ficam paradas
  const bgPoints = L.bg.build();
  bgPoints.userData.noPick = true; // estrelas de fundo não servem de alvo para o zoom
  group.add(bgPoints);
  for (const key of ['gas', 'stars', 'knots', 'bright', 'dust']) {
    if (L[key].pos.length) body.add(L[key].build());
  }
  if (marker) body.add(marker);

  const sprites = glows.map((g) => {
    const mat = new THREE.SpriteMaterial({ map: sprite(THREE), color: g.color, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending });
    const sp = new THREE.Sprite(mat);
    sp.position.set(...g.pos);
    sp.scale.setScalar(g.size);
    sp.userData = { ...g, noPick: true };
    body.add(sp);
    return sp;
  });
  group.add(body);

  const spin = spec.spin ?? 0.05;
  // o giro acumula o tempo só enquanto está ligado (ao dar zoom num detalhe, ele para e o detalhe não foge)
  let last = null;
  let angle = 0;
  return {
    group,
    initialTilt: spec.tilt ?? 0.9,
    animate(t, reduce, running = true) {
      const dt = last === null ? 0 : Math.min(0.1, t - last);
      last = t;
      if (!reduce && running) angle += dt * spin;
      body.rotation.y = angle;
      for (const sp of sprites) {
        const g = sp.userData;
        if (g.kind === 'pulsar') sp.scale.setScalar(g.size * (reduce ? 1 : 0.6 + 0.4 * Math.abs(Math.sin(t * 9))));
        else if (g.kind === 'twinkle' && !reduce) sp.scale.setScalar(g.size * (0.85 + 0.15 * Math.sin(t * 2 + g.pos[0])));
      }
      if (marker && !reduce) marker.scale.setScalar(1 + 0.35 * Math.sin(t * 3));
    },
  };
}
