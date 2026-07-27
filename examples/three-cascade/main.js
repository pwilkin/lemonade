import * as THREE from 'three';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

const SERVER = localStorage.getItem('lemonade.base')
  || new URLSearchParams(location.search).get('server')
  || `${location.protocol}//${location.hostname}:8000`;
const GOLD = new THREE.Color(0xf0b64a);

const TEMP_BY_HINT = { creative: 0.85, balanced: 0.7, precise: 0.4 };
const CTX_BY_HINT = { small: 8192, medium: 32768, large: 65536, max: 131072 };

const PRESETS = [
  { id: 's-creative', name: 'Creative', icon: 'bulb', temperature_hint: 'creative', context_hint: 'medium' },
  { id: 's-code', name: 'Code', icon: 'code', temperature_hint: 'precise', context_hint: 'large' },
  { id: 's-long-context', name: 'Long Context', icon: 'doc', temperature_hint: 'balanced', context_hint: 'max' },
  { id: 's-thorough', name: 'Thorough', icon: 'glass', temperature_hint: 'precise', context_hint: 'large' },
];

const FALLBACK_MODELS = [
  { id: 'Qwen3.5-0.8B-GGUF', recipe: 'llamacpp' },
  { id: 'Nanbeige4.1-3B-GGUF-Q4_K_M', recipe: 'llamacpp' },
  { id: 'Ornith-1.0-35B-GGUF-IQ4_XS', recipe: 'llamacpp' },
  { id: 'Tiny-Test-Model-GGUF', recipe: 'llamacpp' },
];

const state = {
  preset: null,
  model: null,
  phase: 'select',
  t: 0,
  models: FALLBACK_MODELS,
  messages: [],
  busy: false,
};

function effective() {
  const p = state.preset;
  const m = state.model;
  if (!p) return null;
  const modelCtxCap = m ? (m.ctxCap || 131072) : null;
  let ctx = CTX_BY_HINT[p.context_hint];
  let clamped = false;
  if (modelCtxCap && ctx > modelCtxCap) { ctx = modelCtxCap; clamped = true; }
  return {
    rows: [
      ['cpu', 'Context Size', String(ctx), clamped ? 'clamp' : 'gold'],
      ['temp', 'Temperature', TEMP_BY_HINT[p.temperature_hint].toFixed(2), 'gold'],
      ['bolt', 'Flash Attention', m ? 'On' : '—', m ? 'green' : 'dim'],
      ['db', 'KV-Cache', m ? (m.kv || 'Q8_0') : '—', m ? 'gold' : 'dim'],
      ['tag', 'Backend', m ? (m.recipe || 'llamacpp') : '—', m ? 'gold' : 'dim'],
    ],
  };
}

const canvas = document.getElementById('scene');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.82;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x08080a);
scene.fog = new THREE.FogExp2(0x08080a, 0.028);

const camera = new THREE.PerspectiveCamera(32, innerWidth / innerHeight, 0.1, 200);
const CAM_SELECT = { pos: new THREE.Vector3(0, 9.5, 13.5), look: new THREE.Vector3(0, 0.3, 0.2) };
const CAM_CHAT = { pos: new THREE.Vector3(0, 1.2, 10.2), look: new THREE.Vector3(0, 1.2, 0) };
camera.position.copy(CAM_SELECT.pos);
camera.lookAt(CAM_SELECT.look);

const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
scene.environmentIntensity = 0.35;

scene.add(new THREE.AmbientLight(0xffffff, 0.12));
const key = new THREE.DirectionalLight(0xffe6bd, 0.7);
key.position.set(4, 9, 6);
scene.add(key);
const rim = new THREE.DirectionalLight(0x6f8ecc, 0.5);
rim.position.set(-6, 3, -5);
scene.add(rim);
const core = new THREE.PointLight(0xffc766, 30, 14, 2);
core.position.set(0, 1.5, -0.35);
scene.add(core);

const floor = new THREE.Mesh(
  new THREE.PlaneGeometry(70, 70),
  new THREE.MeshStandardMaterial({ color: 0x0c0c10, roughness: 0.55, metalness: 0.3 })
);
floor.rotation.x = -Math.PI / 2;
floor.position.y = -4.6;
scene.add(floor);

function tex(w, h, draw) {
  const c = document.createElement('canvas');
  const s = 2;
  c.width = w * s; c.height = h * s;
  const g = c.getContext('2d');
  g.scale(s, s);
  draw(g, w, h);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = renderer.capabilities.getMaxAnisotropy();
  t.needsUpdate = true;
  return t;
}

function icon(g, kind, x, y, s, color) {
  g.save();
  g.translate(x, y);
  g.strokeStyle = color; g.fillStyle = color;
  g.lineWidth = s * 0.09; g.lineCap = 'round'; g.lineJoin = 'round';
  const u = s / 2;
  if (kind === 'bulb') {
    g.beginPath(); g.arc(0, -u * 0.25, u * 0.62, Math.PI * 0.9, Math.PI * 0.1); g.stroke();
    g.beginPath(); g.moveTo(-u * 0.36, u * 0.3); g.lineTo(-u * 0.28, u * 0.62);
    g.lineTo(u * 0.28, u * 0.62); g.lineTo(u * 0.36, u * 0.3); g.stroke();
    g.beginPath(); g.moveTo(-u * 0.2, u * 0.62); g.lineTo(u * 0.2, u * 0.62); g.stroke();
    for (let i = 0; i < 5; i++) {
      const a = -Math.PI * 0.85 + i * (Math.PI * 0.7 / 4);
      g.beginPath();
      g.moveTo(Math.cos(a) * u * 0.85, Math.sin(a) * u * 0.85 - u * 0.25);
      g.lineTo(Math.cos(a) * u * 1.12, Math.sin(a) * u * 1.12 - u * 0.25);
      g.stroke();
    }
  } else if (kind === 'code') {
    g.font = `600 ${s * 0.78}px ui-monospace, monospace`;
    g.textAlign = 'center'; g.textBaseline = 'middle';
    g.fillText('</>', 0, 0);
  } else if (kind === 'doc') {
    g.beginPath(); g.roundRect(-u * 0.72, -u * 0.85, u * 1.44, u * 1.7, u * 0.18); g.stroke();
    for (let i = 0; i < 4; i++) {
      const yy = -u * 0.45 + i * u * 0.36;
      g.beginPath(); g.moveTo(-u * 0.42, yy); g.lineTo(u * (i === 3 ? 0.05 : 0.42), yy); g.stroke();
    }
  } else if (kind === 'glass') {
    g.beginPath(); g.arc(-u * 0.12, -u * 0.12, u * 0.62, 0, Math.PI * 2); g.stroke();
    g.beginPath(); g.moveTo(u * 0.33, u * 0.33); g.lineTo(u * 0.92, u * 0.92); g.stroke();
  } else if (kind === 'chip') {
    g.beginPath(); g.roundRect(-u * 0.6, -u * 0.6, u * 1.2, u * 1.2, u * 0.2); g.stroke();
    g.beginPath(); g.roundRect(-u * 0.26, -u * 0.26, u * 0.52, u * 0.52, u * 0.1); g.stroke();
    for (let i = -1; i <= 1; i++) {
      g.beginPath(); g.moveTo(i * u * 0.32, -u * 0.6); g.lineTo(i * u * 0.32, -u * 0.88); g.stroke();
      g.beginPath(); g.moveTo(i * u * 0.32, u * 0.6); g.lineTo(i * u * 0.32, u * 0.88); g.stroke();
      g.beginPath(); g.moveTo(-u * 0.6, i * u * 0.32); g.lineTo(-u * 0.88, i * u * 0.32); g.stroke();
      g.beginPath(); g.moveTo(u * 0.6, i * u * 0.32); g.lineTo(u * 0.88, i * u * 0.32); g.stroke();
    }
  } else if (kind === 'cpu') {
    icon(g, 'chip', 0, 0, s, color); g.restore(); return;
  } else if (kind === 'temp') {
    g.beginPath(); g.arc(0, u * 0.5, u * 0.34, 0, Math.PI * 2); g.stroke();
    g.beginPath(); g.moveTo(0, u * 0.16); g.lineTo(0, -u * 0.8); g.stroke();
    g.beginPath(); g.arc(0, -u * 0.8, u * 0.16, Math.PI, 0); g.stroke();
  } else if (kind === 'bolt') {
    g.beginPath();
    g.moveTo(u * 0.22, -u * 0.9); g.lineTo(-u * 0.5, u * 0.1);
    g.lineTo(-u * 0.02, u * 0.1); g.lineTo(-u * 0.22, u * 0.9);
    g.lineTo(u * 0.5, -u * 0.1); g.lineTo(u * 0.02, -u * 0.1);
    g.closePath(); g.stroke();
  } else if (kind === 'db') {
    g.beginPath(); g.ellipse(0, -u * 0.55, u * 0.62, u * 0.24, 0, 0, Math.PI * 2); g.stroke();
    g.beginPath(); g.moveTo(-u * 0.62, -u * 0.55); g.lineTo(-u * 0.62, u * 0.55);
    g.ellipse(0, u * 0.55, u * 0.62, u * 0.24, 0, Math.PI, 0, true);
    g.lineTo(u * 0.62, -u * 0.55); g.stroke();
    g.beginPath(); g.ellipse(0, 0, u * 0.62, u * 0.24, 0, 0, Math.PI); g.stroke();
  } else if (kind === 'tag') {
    g.font = `600 ${s * 0.6}px ui-monospace, monospace`;
    g.textAlign = 'center'; g.textBaseline = 'middle';
    g.fillText('<>', 0, 0);
  }
  g.restore();
}

function glassMat(tint = 0xdfe8f2, opts = {}) {
  return new THREE.MeshPhysicalMaterial({
    color: tint,
    metalness: 0,
    roughness: 0.06,
    transmission: 1.0,
    thickness: 1.1,
    ior: 1.5,
    attenuationColor: new THREE.Color(0x22304a),
    attenuationDistance: 1.6,
    clearcoat: 1,
    clearcoatRoughness: 0.04,
    envMapIntensity: 0.5,
    transparent: true,
    ...opts,
  });
}

const root = new THREE.Group();
scene.add(root);

const PANEL_W = 8.2, PANEL_D = 2.7, PANEL_T = 0.16;
const HEAD_W = 4.4, HEAD_D = 1.1;
const layers = [];
const pickables = [];

function makePanel(cfg) {
  const group = new THREE.Group();
  group.position.set(0, cfg.y, cfg.z);

  const slab = new THREE.Mesh(
    new THREE.BoxGeometry(PANEL_W, PANEL_T, PANEL_D),
    glassMat(0xc9dcf2, { thickness: 0.9, attenuationDistance: 2.2 })
  );
  group.add(slab);

  const edge = new THREE.Mesh(
    new THREE.BoxGeometry(PANEL_W + 0.02, PANEL_T + 0.02, PANEL_D + 0.02),
    new THREE.MeshBasicMaterial({ color: 0x4a3a1c, transparent: true, opacity: 0.55, side: THREE.BackSide })
  );
  group.add(edge);

  const headT = tex(1320, 330, (g, w, h) => {
    g.clearRect(0, 0, w, h);
    g.fillStyle = '#ffe6ad';
    g.font = '600 74px system-ui, sans-serif';
    g.fillText(cfg.title, 12, 82);
    g.fillStyle = 'rgba(232,226,216,.5)';
    g.font = 'italic 300 44px system-ui, sans-serif';
    cfg.sub.split('\n').forEach((line, i) => g.fillText(line, 12, 152 + i * 54));
  });
  const head = new THREE.Mesh(
    new THREE.PlaneGeometry(HEAD_W, HEAD_D),
    new THREE.MeshBasicMaterial({ map: headT, transparent: true, depthWrite: false })
  );
  head.rotation.x = -Math.PI / 2;
  head.position.set(-PANEL_W / 2 + HEAD_W / 2 + 0.34, PANEL_T / 2 + 0.012, -PANEL_D / 2 + HEAD_D / 2 + 0.16);
  group.add(head);

  root.add(group);
  const layer = { ...cfg, group, slab, head, tiles: [] };
  layers.push(layer);
  return layer;
}

const TILE_W = 1.72, TILE_D = 0.98, TILE_T = 0.1;

function makeTile(layer, item, i, n) {
  const spanW = PANEL_W - 1.0;
  const step = spanW / n;
  const x = -spanW / 2 + step * (i + 0.5);

  const g = new THREE.Group();
  g.position.set(x, PANEL_T / 2 + TILE_T / 2, PANEL_D / 2 - TILE_D / 2 - 0.2);

  const body = new THREE.Mesh(
    new THREE.BoxGeometry(Math.min(TILE_W, step - 0.12), TILE_T, TILE_D),
    new THREE.MeshPhysicalMaterial({
      color: 0x11131a, metalness: 0.1, roughness: 0.28,
      transmission: 0.5, thickness: 0.3, ior: 1.4,
      clearcoat: 1, clearcoatRoughness: 0.15, transparent: true, opacity: 0.96,
    })
  );
  g.add(body);

  const halo = new THREE.Mesh(
    new THREE.PlaneGeometry(Math.min(TILE_W, step - 0.12) * 1.5, TILE_D * 1.9),
    new THREE.MeshBasicMaterial({
      color: GOLD, transparent: true, opacity: 0,
      blending: THREE.AdditiveBlending, depthWrite: false,
      map: radialTex(),
    })
  );
  halo.rotation.x = -Math.PI / 2;
  halo.position.y = -TILE_T / 2 + 0.005;
  g.add(halo);

  const faceW = Math.min(TILE_W, step - 0.12);
  const face = new THREE.Mesh(
    new THREE.PlaneGeometry(faceW, TILE_D),
    new THREE.MeshBasicMaterial({ transparent: true, depthWrite: false })
  );
  face.rotation.x = -Math.PI / 2;
  face.position.y = TILE_T / 2 + 0.006;
  g.add(face);

  layer.group.add(g);
  const tile = {
    layer, item, group: g, body, face, halo,
    baseY: g.position.y, hover: false, selected: false, glow: 0, lift: 0,
    faceW,
  };
  paintTile(tile);
  layer.tiles.push(tile);
  pickables.push(body);
  body.userData.tile = tile;
  return tile;
}

function paintTile(t) {
  const W = 520, H = Math.round(520 * (TILE_D / t.faceW));
  const sel = t.selected, hov = t.hover;
  const map = tex(W, H, (g, w, h) => {
    g.clearRect(0, 0, w, h);
    const r = 26;
    g.beginPath(); g.roundRect(6, 6, w - 12, h - 12, r);
    g.fillStyle = sel ? 'rgba(240,182,74,.14)' : hov ? 'rgba(240,182,74,.06)' : 'rgba(255,255,255,.018)';
    g.fill();
    g.strokeStyle = sel ? 'rgba(255,209,124,.95)' : hov ? 'rgba(240,182,74,.5)' : 'rgba(240,182,74,.2)';
    g.lineWidth = sel ? 4 : 2.5;
    g.stroke();

    const col = sel ? '#ffd98a' : hov ? '#f0b64a' : 'rgba(240,182,74,.75)';
    const label = t.item.name || t.item.id;
    const iconKind = t.item.icon || 'chip';
    icon(g, iconKind, w / 2, h * 0.36, h * 0.3, col);

    g.fillStyle = sel ? '#ffe8b4' : 'rgba(236,226,208,.9)';
    g.textAlign = 'center'; g.textBaseline = 'middle';
    let fs = 44;
    g.font = `500 ${fs}px system-ui, sans-serif`;
    while (g.measureText(label).width > w - 48 && fs > 20) {
      fs -= 2; g.font = `500 ${fs}px system-ui, sans-serif`;
    }
    g.fillText(label, w / 2, h * 0.74);
  });
  if (t.face.material.map) t.face.material.map.dispose();
  t.face.material.map = map;
  t.face.material.needsUpdate = true;
}

let _radial;
function radialTex() {
  if (_radial) return _radial;
  _radial = tex(256, 256, (g, w, h) => {
    const grd = g.createRadialGradient(w / 2, h / 2, 0, w / 2, h / 2, w / 2);
    grd.addColorStop(0, 'rgba(255,255,255,1)');
    grd.addColorStop(0.4, 'rgba(255,255,255,.35)');
    grd.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = grd; g.fillRect(0, 0, w, h);
  });
  return _radial;
}

const layerIntent = makePanel({ key: 'intent', title: 'Intent / Preset', sub: 'What do I want to achieve?', y: 3.15, z: -0.85 });
const layerModel = makePanel({ key: 'model', title: 'Model Tuning', sub: 'How does this model\ninterpret the intention?', y: 0.75, z: 0.10 });

PRESETS.forEach((p, i) => makeTile(layerIntent, p, i, PRESETS.length));

function buildModelTiles() {
  layerModel.tiles.forEach(t => {
    layerModel.group.remove(t.group);
    const idx = pickables.indexOf(t.body);
    if (idx >= 0) pickables.splice(idx, 1);
  });
  layerModel.tiles = [];
  state.models.slice(0, 4).forEach((m, i) => makeTile(layerModel, m, i, Math.min(4, state.models.length)));
}
buildModelTiles();

const EFF_SELECT = new THREE.Vector3(0, -1.95, 1.15);
const EFF_CHAT = new THREE.Vector3(0, 1.2, 0.1);

const effGroup = new THREE.Group();
effGroup.position.copy(EFF_SELECT);
root.add(effGroup);

const EFF_W = 6.7, EFF_D = 3.6;
const effSlab = new THREE.Mesh(
  new THREE.BoxGeometry(EFF_W, PANEL_T, EFF_D),
  new THREE.MeshPhysicalMaterial({
    color: 0x0b0d14, metalness: 0.25, roughness: 0.18,
    transmission: 0.35, thickness: 0.5, ior: 1.45,
    clearcoat: 1, clearcoatRoughness: 0.08, transparent: true, opacity: 0.98,
  })
);
effGroup.add(effSlab);

const effEdge = new THREE.Mesh(
  new THREE.BoxGeometry(EFF_W + 0.02, PANEL_T + 0.02, EFF_D + 0.02),
  new THREE.MeshBasicMaterial({ color: 0x5a4520, transparent: true, opacity: 0.7, side: THREE.BackSide })
);
effGroup.add(effEdge);

const effFace = new THREE.Mesh(
  new THREE.PlaneGeometry(EFF_W * 0.985, EFF_D * 0.985),
  new THREE.MeshBasicMaterial({ transparent: true, depthWrite: false })
);
effFace.rotation.x = -Math.PI / 2;
effFace.position.y = PANEL_T / 2 + 0.01;
effGroup.add(effFace);
effSlab.renderOrder = 10;
effEdge.renderOrder = 10;
effFace.renderOrder = 11;

function paintEffective() {
  const W = 1320, H = Math.round(1320 * (EFF_D / EFF_W));
  const e = effective();
  const map = tex(W, H, (g, w, h) => {
    g.clearRect(0, 0, w, h);
    const pad = 52;

    icon(g, 'chip', pad + 40, 78, 62, '#f0b64a');
    g.beginPath(); g.roundRect(pad + 4, 42, 74, 74, 16);
    g.strokeStyle = 'rgba(240,182,74,.35)'; g.lineWidth = 2; g.stroke();

    g.fillStyle = '#ffe6ad';
    g.font = '600 46px system-ui, sans-serif';
    g.textAlign = 'left'; g.textBaseline = 'alphabetic';
    g.fillText('Effective Load Settings', pad + 100, 72);
    g.fillStyle = 'rgba(232,226,216,.45)';
    g.font = 'italic 300 30px system-ui, sans-serif';
    g.fillText(e ? 'Actual runtime configuration' : 'Choose an intent above', pad + 100, 112);

    const rows = e ? e.rows : [
      ['cpu', 'Context Size', '—', 'dim'], ['temp', 'Temperature', '—', 'dim'],
      ['bolt', 'Flash Attention', '—', 'dim'], ['db', 'KV-Cache', '—', 'dim'],
      ['tag', 'Backend', '—', 'dim'],
    ];
    const top = 168, rh = 76, gap = 10;
    rows.forEach((r, i) => {
      const y = top + i * (rh + gap);
      g.beginPath(); g.roundRect(pad, y, w - pad * 2, rh, 14);
      g.fillStyle = 'rgba(255,255,255,.028)'; g.fill();
      g.strokeStyle = 'rgba(240,182,74,.14)'; g.lineWidth = 1.5; g.stroke();

      icon(g, r[0], pad + 44, y + rh / 2, 38, 'rgba(240,182,74,.8)');

      g.textAlign = 'left'; g.textBaseline = 'middle';
      g.fillStyle = 'rgba(236,226,208,.9)';
      g.font = '400 34px system-ui, sans-serif';
      g.fillText(r[1], pad + 88, y + rh / 2);

      const colors = { gold: '#f0b64a', green: '#7fd68a', dim: 'rgba(232,226,216,.3)', clamp: '#ffa057' };
      g.textAlign = 'right';
      g.fillStyle = colors[r[3]] || '#f0b64a';
      g.font = '600 36px system-ui, sans-serif';
      g.fillText(r[2], w - pad - 28, y + rh / 2);
    });
  });
  if (effFace.material.map) effFace.material.map.dispose();
  effFace.material.map = map;
  effFace.material.needsUpdate = true;
}
paintEffective();

const funnel = new THREE.Group();
funnel.position.set(0, 1.95, -0.35);
root.add(funnel);

for (let i = 0; i < 4; i++) {
  const k = i / 3;
  const rTop = 1.15 - k * 0.62;
  const rBot = rTop * 0.34;
  const hgt = 0.85 - k * 0.34;
  const ring = new THREE.Mesh(
    new THREE.CylinderGeometry(rTop, rBot, hgt, 72, 1, true),
    glassMat(0xdce8fa, {
      thickness: 0.22, roughness: 0.02, attenuationDistance: 4.0,
      side: THREE.DoubleSide, envMapIntensity: 1.1,
    })
  );
  ring.position.y = -0.05 - i * 0.16;
  funnel.add(ring);
}

const beamMat = new THREE.ShaderMaterial({
  uniforms: {
    uTime: { value: 0 },
    uIntensity: { value: 0.35 },
    uColor: { value: new THREE.Color(0xffc266) },
  },
  vertexShader: `
    varying vec2 vUv; varying vec3 vN; varying vec3 vV;
    void main(){
      vUv = uv;
      vN = normalize(normalMatrix * normal);
      vec4 mv = modelViewMatrix * vec4(position,1.0);
      vV = normalize(-mv.xyz);
      gl_Position = projectionMatrix * mv;
    }`,
  fragmentShader: `
    uniform float uTime; uniform float uIntensity; uniform vec3 uColor;
    varying vec2 vUv; varying vec3 vN; varying vec3 vV;
    void main(){
      float fres = pow(1.0 - abs(dot(normalize(vN), normalize(vV))), 1.6);
      float streak = 0.65 + 0.35 * sin(vUv.x * 42.0 + uTime * 1.4);
      float fade = smoothstep(0.0, 0.35, vUv.y) * smoothstep(1.0, 0.72, vUv.y);
      float a = fres * streak * (0.35 + fade) * uIntensity;
      gl_FragColor = vec4(uColor * (1.2 + fres), a);
    }`,
  transparent: true,
  blending: THREE.AdditiveBlending,
  depthWrite: false,
  depthTest: false,
  side: THREE.DoubleSide,
});

const beamTop = new THREE.Mesh(new THREE.CylinderGeometry(1.05, 0.8, 4.4, 64, 1, true), beamMat);
beamTop.position.set(0, 4.3, -0.2);
beamTop.renderOrder = 5;
root.add(beamTop);

const beamBot = new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.02, 3.0, 64, 1, true), beamMat);
beamBot.position.set(0, -0.1, -0.3);
beamBot.renderOrder = 5;
root.add(beamBot);

const shafts = new THREE.Mesh(new THREE.CylinderGeometry(3.0, 1.15, 7.0, 48, 1, true), beamMat.clone());
shafts.material.uniforms.uIntensity.value = 0.1;
shafts.position.set(0, 8.0, -0.4);
shafts.renderOrder = 4;
root.add(shafts);

const spark = new THREE.Sprite(new THREE.SpriteMaterial({
  map: radialTex(), color: 0xffd79a, transparent: true,
  blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false, opacity: 0.9,
}));
spark.scale.set(1.8, 1.8, 1);
spark.position.set(0, -1.3, -0.9);
spark.renderOrder = 6;
root.add(spark);

const moteCount = 420;
const mpos = new Float32Array(moteCount * 3);
for (let i = 0; i < moteCount; i++) {
  const r = 1.2 + Math.random() * 5.5;
  const a = Math.random() * Math.PI * 2;
  mpos[i * 3] = Math.cos(a) * r;
  mpos[i * 3 + 1] = -3 + Math.random() * 9;
  mpos[i * 3 + 2] = Math.sin(a) * r * 0.6;
}
const moteGeo = new THREE.BufferGeometry();
moteGeo.setAttribute('position', new THREE.BufferAttribute(mpos, 3));
const motes = new THREE.Points(moteGeo, new THREE.PointsMaterial({
  size: 0.035, color: 0xffcf90, transparent: true, opacity: 0.5,
  blending: THREE.AdditiveBlending, depthWrite: false, map: radialTex(),
}));
root.add(motes);

const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(innerWidth, innerHeight), 0.6, 0.85, 0.6);
composer.addPass(bloom);
composer.addPass(new OutputPass());

const ray = new THREE.Raycaster();
const ptr = new THREE.Vector2(-10, -10);
let hovered = null;

addEventListener('pointermove', (e) => {
  ptr.x = (e.clientX / innerWidth) * 2 - 1;
  ptr.y = -(e.clientY / innerHeight) * 2 + 1;
});

addEventListener('pointerdown', () => {
  if (state.phase !== 'select' || !hovered) return;
  const t = hovered;
  t.layer.tiles.forEach(o => { if (o !== t) { o.selected = false; paintTile(o); } });
  t.selected = !t.selected;
  paintTile(t);
  if (t.layer.key === 'intent') state.preset = t.selected ? t.item : null;
  else state.model = t.selected ? t.item : null;
  paintEffective();
  updateCta();
});

const cta = document.getElementById('cta');
const hint = document.getElementById('hint');
const chatEl = document.getElementById('chat');

function updateCta() {
  const ready = state.preset && state.model;
  cta.classList.toggle('on', !!ready && state.phase === 'select');
  hint.style.opacity = ready ? 0 : (state.phase === 'select' ? 1 : 0);
  hint.textContent = !state.preset ? 'Pick an intent, then a model'
    : !state.model ? 'Now pick a model' : '';
}

cta.addEventListener('click', () => enterChat());
document.getElementById('back').addEventListener('click', () => exitChat());

function enterChat() {
  state.phase = 'chat';
  cta.classList.remove('on');
  hint.style.opacity = 0;
  document.getElementById('chat-title').textContent = state.model.id;
  document.getElementById('chat-meta').textContent =
    `${state.preset.name} · ${effective().rows[0][2]} ctx · T=${effective().rows[1][2]}`;
  if (!state.messages.length) sysMsg('Model loads on first message — that can take a moment.');
  setTimeout(() => { chatEl.classList.add('on'); document.getElementById('input').focus(); }, 620);
}

function exitChat() {
  state.phase = 'select';
  chatEl.classList.remove('on');
  updateCta();
}

const _v = new THREE.Vector3();
function fitChatToPanel() {
  const half = new THREE.Vector3(EFF_W / 2, 0, EFF_D / 2);
  let minX = 1e9, minY = 1e9, maxX = -1e9, maxY = -1e9;
  for (const sx of [-1, 1]) for (const sz of [-1, 1]) {
    _v.set(half.x * sx, PANEL_T / 2, half.z * sz);
    effGroup.localToWorld(_v);
    _v.project(camera);
    const px = (_v.x * 0.5 + 0.5) * innerWidth;
    const py = (-_v.y * 0.5 + 0.5) * innerHeight;
    minX = Math.min(minX, px); maxX = Math.max(maxX, px);
    minY = Math.min(minY, py); maxY = Math.max(maxY, py);
  }
  chatEl.style.left = `${minX}px`;
  chatEl.style.top = `${minY}px`;
  chatEl.style.width = `${maxX - minX}px`;
  chatEl.style.height = `${maxY - minY}px`;
}

const logEl = document.getElementById('log');
const inputEl = document.getElementById('input');
const sendEl = document.getElementById('send');
const errEl = document.getElementById('err');

function showErr(m) {
  errEl.textContent = m; errEl.classList.add('on');
  setTimeout(() => errEl.classList.remove('on'), 5200);
}
function addMsg(cls, text) {
  const d = document.createElement('div');
  d.className = `msg ${cls}`; d.textContent = text;
  logEl.appendChild(d); logEl.scrollTop = logEl.scrollHeight;
  return d;
}
const sysMsg = (t) => addMsg('sys', t);

async function send() {
  const text = inputEl.value.trim();
  if (!text || state.busy) return;
  inputEl.value = '';
  addMsg('user', text);
  state.messages.push({ role: 'user', content: text });
  state.busy = true; sendEl.disabled = true;

  const botEl = addMsg('bot', '');
  let acc = '';
  const e = effective();

  try {
    const res = await fetch(`${SERVER}/api/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: state.model.id,
        messages: state.messages,
        stream: true,
        temperature: parseFloat(e.rows[1][2]),
      }),
    });
    if (!res.ok) throw new Error(`${res.status} ${await res.text().catch(() => '')}`.slice(0, 200));

    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const line of lines) {
        const s = line.trim();
        if (!s.startsWith('data:')) continue;
        const payload = s.slice(5).trim();
        if (payload === '[DONE]') continue;
        try {
          const j = JSON.parse(payload);
          const d = j.choices?.[0]?.delta?.content;
          if (d) { acc += d; botEl.textContent = acc; logEl.scrollTop = logEl.scrollHeight; }
        } catch {}
      }
    }
    state.messages.push({ role: 'assistant', content: acc });
  } catch (err) {
    botEl.remove();
    showErr(`Chat failed: ${err.message}`);
    state.messages.pop();
  } finally {
    state.busy = false; sendEl.disabled = false;
  }
}

sendEl.addEventListener('click', send);
inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});

const LLM_RECIPES = new Set(['llamacpp', 'ryzenai-llm', 'flm', 'vllm']);
(async () => {
  try {
    const r = await fetch(`${SERVER}/api/v1/models`);
    const j = await r.json();
    const all = j.data || j.models || [];
    const llms = all.filter(m => LLM_RECIPES.has(m.recipe));
    if (llms.length) {
      state.models = llms.slice(0, 4).map(m => ({
        id: m.id, recipe: m.recipe,
        ctxCap: m.max_prompt_length || m.context_length || 131072,
        kv: 'Q8_0',
      }));
      buildModelTiles();
      paintEffective();
    }
  } catch (err) {
    showErr(`Could not reach ${SERVER} — using sample models.`);
  }
})();

const TILT_UP = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI / 2);

const clock = new THREE.Clock();
const camPos = camera.position.clone();
const camLook = CAM_SELECT.look.clone();

function ease(x) { return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2; }

function tick() {
  requestAnimationFrame(tick);
  const dt = Math.min(clock.getDelta(), 0.05);
  const time = clock.elapsedTime;

  const target = state.phase === 'chat' ? 1 : 0;
  state.t += (target - state.t) * Math.min(1, dt * 3.4);
  const k = ease(THREE.MathUtils.clamp(state.t, 0, 1));

  if (state.phase === 'select') {
    ray.setFromCamera(ptr, camera);
    const hits = ray.intersectObjects(pickables, false);
    const next = hits.length ? hits[0].object.userData.tile : null;
    if (next !== hovered) {
      if (hovered) { hovered.hover = false; paintTile(hovered); }
      hovered = next;
      if (hovered) { hovered.hover = true; paintTile(hovered); }
      document.body.style.cursor = hovered ? 'pointer' : 'default';
    }
  } else if (hovered) {
    hovered.hover = false; paintTile(hovered); hovered = null;
    document.body.style.cursor = 'default';
  }

  for (const layer of layers) {
    for (const t of layer.tiles) {
      const wantLift = (t.selected ? 0.16 : 0) + (t.hover ? 0.07 : 0);
      t.lift += (wantLift - t.lift) * Math.min(1, dt * 9);
      t.group.position.y = t.baseY + t.lift;
      const wantGlow = t.selected ? 0.55 : t.hover ? 0.22 : 0;
      t.glow += (wantGlow - t.glow) * Math.min(1, dt * 8);
      t.halo.material.opacity = t.glow * (0.85 + 0.15 * Math.sin(time * 2.4));
    }
  }

  layerIntent.group.position.y = 3.15 + k * 5.4;
  layerIntent.group.scale.setScalar(1 - k * 0.25);
  layerModel.group.position.y = 0.75 + k * 3.6;
  layerModel.group.scale.setScalar(1 - k * 0.25);
  setOpacity(layerIntent.group, 1 - k * 1.3);
  setOpacity(layerModel.group, 1 - k * 1.3);

  funnel.scale.setScalar(1 - k * 0.9);
  setOpacity(funnel, 1 - k * 1.4);
  funnel.rotation.y = time * 0.12;

  effGroup.position.lerpVectors(EFF_SELECT, EFF_CHAT, k);
  effGroup.scale.set(1 + k * 0.05, 1, 1 + k * 0.26);
  effFace.material.opacity = THREE.MathUtils.clamp(1 - k * 1.6, 0, 1);
  effFace.visible = effFace.material.opacity > 0.002;

  const resolved = (state.preset ? 0.5 : 0) + (state.model ? 0.5 : 0);
  const want = (0.10 + resolved * 0.26) * (1 - k);
  beamMat.uniforms.uIntensity.value += (want - beamMat.uniforms.uIntensity.value) * Math.min(1, dt * 3);
  beamMat.uniforms.uTime.value = time;
  shafts.material.uniforms.uTime.value = time;
  shafts.material.uniforms.uIntensity.value = (0.03 + resolved * 0.05) * (1 - k);
  spark.material.opacity = (0.12 + resolved * 0.22) * (1 - k) * (0.85 + 0.15 * Math.sin(time * 3.1));
  spark.scale.setScalar(1.5 + resolved * 0.8 + Math.sin(time * 2.2) * 0.1);
  core.intensity = (3 + resolved * 7) * (1 - k * 0.8);
  bloom.strength = 0.34 + resolved * 0.16;

  motes.rotation.y = time * 0.03;
  motes.material.opacity = 0.36 * (1 - k * 0.6);

  camPos.lerpVectors(CAM_SELECT.pos, CAM_CHAT.pos, k);
  camLook.lerpVectors(CAM_SELECT.look, CAM_CHAT.look, k);
  const sway = state.phase === 'select' ? 1 : 0.15;
  const sx = THREE.MathUtils.clamp(ptr.x, -1, 1);
  const sy = THREE.MathUtils.clamp(ptr.y, -1, 1);
  camera.position.set(
    camPos.x + sx * 0.5 * sway + Math.sin(time * 0.25) * 0.12 * sway,
    camPos.y + sy * 0.3 * sway,
    camPos.z
  );
  camera.lookAt(camLook);

  if (k > 0.001) {
    const billboard = new THREE.Quaternion().copy(camera.quaternion).multiply(TILT_UP);
    effGroup.quaternion.identity().slerp(billboard, k);
  } else {
    effGroup.quaternion.identity();
  }

  if (k > 0.02) {
    effGroup.updateMatrixWorld(true);
    fitChatToPanel();
  }

  composer.render();
}

function setOpacity(obj, v) {
  const o = THREE.MathUtils.clamp(v, 0, 1);
  obj.visible = o > 0.01;
  obj.traverse(c => {
    if (!c.material) return;
    const mats = Array.isArray(c.material) ? c.material : [c.material];
    mats.forEach(m => {
      if (m.userData.baseOpacity === undefined) m.userData.baseOpacity = m.opacity;
      m.transparent = true;
      m.opacity = m.userData.baseOpacity * o;
    });
  });
}

addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
  composer.setSize(innerWidth, innerHeight);
});

window.__lemon = { state, effective, layers, camera };

updateCta();
tick();
