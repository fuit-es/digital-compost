// Compost viewer: draws the Python simulation streamed over /ws (see network/viewer_hub.py).

const RECORD = 7; // id, x, y, angle, scale, volume, tint
const SERVER_KEYS = new Set(["d", "t", "h", "g", "i", "o", "c", "q", "e", "k", "p", "escape"]);

const world = { w: 1280, h: 800 };
let config = null;
let flags = {};
let busy = false;

const chunks = new Map(); // id -> { sprite, ghosts, tex, w, h, verts, curve, variants }
const glues = new Map();  // id -> { glow, wheel, x, y, meals, glowRgb }
const thumbs = new Map(); // history entry id -> canvas
const particles = [];
const particlePool = [];

// ------------------------------------------------------------------
// Pixi setup
// ------------------------------------------------------------------
const app = new PIXI.Application();
await app.init({
  resizeTo: window,
  antialias: true,
  resolution: window.devicePixelRatio || 1,
  autoDensity: true,
  background: 0xffffff,
});
document.getElementById("stage").appendChild(app.canvas);

const scene = new PIXI.Container();
const chunkLayer = new PIXI.Container();
chunkLayer.sortableChildren = true;
const dotLayer = new PIXI.Graphics();
const debugLayer = new PIXI.Graphics();
const glueLayer = new PIXI.Container();
const particleLayer = new PIXI.Container();
const worldMask = new PIXI.Graphics();
scene.addChild(chunkLayer, dotLayer, debugLayer, glueLayer, particleLayer, worldMask);
scene.mask = worldMask;
app.stage.addChild(scene);

const glowTexture = PIXI.Texture.from(radialCanvas(128));
const dotTexture = PIXI.Texture.from(discCanvas(16));

function radialCanvas(size) {
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const ctx = c.getContext("2d");
  const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  g.addColorStop(0, "rgba(255,255,255,1)");
  g.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  return c;
}

function discCanvas(radius) {
  const c = document.createElement("canvas");
  c.width = c.height = radius * 2;
  const ctx = c.getContext("2d");
  ctx.fillStyle = "#fff";
  ctx.beginPath();
  ctx.arc(radius, radius, radius, 0, Math.PI * 2);
  ctx.fill();
  return c;
}

function rgbaCanvas(buf, off, w, h) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const pixels = new Uint8ClampedArray(buf, off, w * h * 4);
  c.getContext("2d").putImageData(new ImageData(pixels, w, h), 0, 0);
  return c;
}

const rgbInt = ([r, g, b]) => (r << 16) | (g << 8) | b;
const css = ([r, g, b]) => `rgb(${r},${g},${b})`;

// Fit the world into the window (letterboxed when the world size is fixed)
let fit = { s: 1, x: 0, y: 0 };
function layout() {
  const sw = window.innerWidth;
  const sh = window.innerHeight;
  const s = Math.min(sw / world.w, sh / world.h);
  fit = { s, x: (sw - world.w * s) / 2, y: (sh - world.h * s) / 2 };
  scene.scale.set(s);
  scene.position.set(fit.x, fit.y);
  worldMask.clear().rect(0, 0, world.w, world.h).fill(0xffffff);
  layoutOverlays();
}
window.addEventListener("resize", () => {
  layout();
  clearTimeout(layout.timer);
  layout.timer = setTimeout(sendSize, 250);
});

// ------------------------------------------------------------------
// Connection
// ------------------------------------------------------------------
let ws = null;
const decoder = new TextDecoder();

function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => {
    showStatus(null);
    sendSize();
  };
  ws.onclose = () => {
    showStatus("Disconnected from the compost, retrying…");
    setTimeout(connect, 1000);
  };
  ws.onmessage = (event) => {
    const buf = event.data;
    const n = new DataView(buf).getUint32(0, true);
    const header = JSON.parse(decoder.decode(new Uint8Array(buf, 4, n)));
    const off = (4 + n + 3) & ~3;
    handlers[header.t]?.(header, buf, off);
  };
}

function send(message) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(message));
}

function sendSize() {
  send({ t: "size", w: window.innerWidth, h: window.innerHeight });
}

const handlers = {
  hello(h) {
    config = h.config;
    resetScene();
    setWorld(h.world);
    buildPill();
  },
  add(h, buf, off) {
    const tex = PIXI.Texture.from(rgbaCanvas(buf, off, h.w, h.h));
    const sprite = new PIXI.Sprite(tex);
    sprite.anchor.set(0.5);
    sprite.visible = false; // until its first frame
    chunkLayer.addChild(sprite);
    chunks.set(h.id, { sprite, ghosts: [], tex, w: h.w, h: h.h, verts: h.verts, curve: h.curve, variants: new Map() });
  },
  rm(h) {
    for (const id of h.ids) removeChunk(id);
  },
  glue(h) {
    const glow = new PIXI.Sprite(glowTexture);
    glow.anchor.set(0.5);
    glow.tint = rgbInt(h.glow);
    const wheel = new PIXI.Graphics();
    glueLayer.addChild(glow, wheel);
    glues.set(h.id, { glow, wheel, x: h.x, y: h.y, meals: h.meals.map(rgbInt) });
  },
  glue_rm(h) {
    for (const id of h.ids) {
      const g = glues.get(id);
      if (!g) continue;
      g.glow.destroy();
      g.wheel.destroy();
      glues.delete(id);
    }
  },
  burst(h) {
    for (const b of h.bursts) spawnBurst(b);
  },
  thumb(h, buf, off) {
    thumbs.set(h.id, rgbaCanvas(buf, off, h.w, h.h));
  },
  panels(h) {
    renderPanels(h.worms);
  },
  f(h, buf, off) {
    setWorld(h.world);
    applyFlags(h);
    drawFrame(new Float32Array(buf, off, (buf.byteLength - off) / 4), h.glues);
  },
};

function setWorld([w, h]) {
  if (w === world.w && h === world.h) return;
  world.w = w;
  world.h = h;
  layout();
}

function resetScene() {
  for (const id of [...chunks.keys()]) removeChunk(id);
  handlers.glue_rm({ ids: [...glues.keys()] });
  for (const p of particles) p.sprite.destroy();
  particles.length = 0;
  thumbs.clear();
  renderPanels([]);
}

function removeChunk(id) {
  const c = chunks.get(id);
  if (!c) return;
  c.sprite.destroy();
  for (const g of c.ghosts) g.destroy();
  for (const t of c.variants.values()) t.destroy(true);
  c.tex.destroy(true);
  chunks.delete(id);
}

// ------------------------------------------------------------------
// Frame drawing
// ------------------------------------------------------------------
function drawFrame(rec, glueStates) {
  dotLayer.clear();
  debugLayer.clear();
  for (let i = 0; i < rec.length; i += RECORD) {
    const c = chunks.get(rec[i]);
    if (!c) continue;
    const x = rec[i + 1], y = rec[i + 2], angle = rec[i + 3], scale = rec[i + 4];
    const volume = rec[i + 5], tint = rec[i + 6];
    const tex = volume > 0 && c.curve ? volumeTexture(c, volume) : c.tex;

    const positions = [[x, y]];
    if (flags.torus) positions.push(...torusGhosts(c, x, y, angle, scale));
    while (c.ghosts.length < positions.length - 1) {
      const ghost = new PIXI.Sprite(c.tex);
      ghost.anchor.set(0.5);
      chunkLayer.addChild(ghost);
      c.ghosts.push(ghost);
    }
    const sprites = [c.sprite, ...c.ghosts];
    sprites.forEach((sprite, k) => {
      const pos = positions[k];
      sprite.visible = Boolean(pos);
      if (!pos) return;
      sprite.texture = tex;
      sprite.position.set(pos[0], pos[1]);
      sprite.rotation = angle;
      sprite.scale.set(scale);
      sprite.zIndex = tint >= 0 ? 1 : 0; // glued chunks on top
      if (flags.debug) drawOutline(c.verts, pos[0], pos[1], angle);
    });
    if (tint >= 0 && flags.glue) dotLayer.circle(x, y, 5).fill(tint);
  }
  if (flags.debug) debugLayer.stroke({ width: 1, color: 0xff0000 });
  drawGlues(glueStates);
}

// Extra positions where a chunk crossing an edge shows up on the other side
function torusGhosts(c, x, y, angle, scale) {
  const cos = Math.abs(Math.cos(angle)), sin = Math.abs(Math.sin(angle));
  const hw = ((c.w * cos + c.h * sin) / 2) * scale;
  const hh = ((c.w * sin + c.h * cos) / 2) * scale;
  const dx = x - hw < 0 ? world.w : x + hw > world.w ? -world.w : 0;
  const dy = y - hh < 0 ? world.h : y + hh > world.h ? -world.h : 0;
  const out = [];
  if (dx) out.push([x + dx, y]);
  if (dy) out.push([x, y + dy]);
  if (dx && dy) out.push([x + dx, y + dy]);
  return out;
}

function drawOutline(verts, x, y, angle) {
  const cos = Math.cos(angle), sin = Math.sin(angle);
  const pts = [];
  for (let i = 0; i < verts.length; i += 2) {
    pts.push(verts[i] * cos - verts[i + 1] * sin + x, verts[i] * sin + verts[i + 1] * cos + y);
  }
  debugLayer.poly(pts, true);
}

// Sound chunks get thicker and brighter as they get louder
function volumeTexture(c, volume) {
  const vs = config.sound.hover.volume_scaling;
  const p = Math.pow(volume, vs.perceptual_exponent ?? 0.7);
  const lw = Math.max(1, Math.floor(c.curve.lw * (1 + p * (vs.line_width_multiplier ?? 2))));
  const [r, g, b, a] = c.curve.rgba;
  const alpha = Math.min(1, a * (1 + p * (vs.alpha_boost ?? 0.5)));
  const lwKey = Math.floor(lw / 2) * 2;
  const alphaKey = Math.floor(alpha * 10) / 10;
  const key = `${lwKey}:${alphaKey}`;
  let tex = c.variants.get(key);
  if (!tex) {
    const canvas = document.createElement("canvas");
    canvas.width = c.w;
    canvas.height = c.h;
    const ctx = canvas.getContext("2d");
    ctx.lineWidth = Math.max(1, lwKey);
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.strokeStyle = `rgba(${r * 255},${g * 255},${b * 255},${alphaKey})`;
    const pts = c.curve.pts;
    ctx.beginPath();
    ctx.moveTo(pts[0], pts[1]);
    for (let i = 2; i < pts.length; i += 2) ctx.lineTo(pts[i], pts[i + 1]);
    ctx.stroke();
    tex = PIXI.Texture.from(canvas);
    c.variants.set(key, tex);
  }
  return tex;
}

// Glowing dot with a spinning wheel of the worm's meal colors
function drawGlues(states) {
  const v = config.worm.glue.visual;
  glueLayer.visible = Boolean(flags.glue);
  for (const [id, pulse, full] of states) {
    const g = glues.get(id);
    if (!g) continue;
    const pf = v.pulse_min + (v.pulse_max - v.pulse_min) * Math.sin(pulse);
    const outer = Math.floor(v.glow_radius * 2 * pf);
    const core = Math.floor(v.glow_radius * pf);
    g.glow.visible = outer > 0;
    g.glow.position.set(g.x, g.y);
    g.glow.width = g.glow.height = outer * 2;
    g.glow.alpha = v.glow_alpha_max / 255;
    g.wheel.clear();
    if (outer <= 0 || core <= 0 || !g.meals.length) continue;
    const start = pulse * v.spin_speed;
    const step = (Math.PI * 2) / g.meals.length;
    g.meals.forEach((color, k) => {
      const a0 = start + k * step;
      g.wheel.moveTo(g.x, g.y).arc(g.x, g.y, core, a0, a0 + step).lineTo(g.x, g.y).fill(color);
    });
    if (full) g.wheel.circle(g.x, g.y, Math.max(1, core + 2)).stroke({ width: v.full_ring_width, color: 0xffffff });
  }
}

// ------------------------------------------------------------------
// Consumption particles
// ------------------------------------------------------------------
function spawnBurst({ x, y, rgb, extent }) {
  const p = config.worm.consumption_particles;
  const spread = extent * 0.5;
  const ratio = Math.min(1, extent / p.max_size);
  const count = Math.floor(p.min_count + (p.max_count - p.min_count) * ratio);
  const tint = rgbInt(rgb);
  for (let i = 0; i < count; i++) {
    const a = Math.random() * Math.PI * 2;
    const r = spread * Math.sqrt(Math.random());
    const dir = Math.random() * Math.PI * 2;
    const speed = Math.random() * p.speed;
    const rMin = Math.max(1, p.radius - p.radius_jitter);
    const radius = rMin + Math.floor(Math.random() * (p.radius + p.radius_jitter - rMin + 1));
    const life = 1 + Math.floor(Math.random() * p.lifetime_frames);
    const sprite = particlePool.pop() || new PIXI.Sprite(dotTexture);
    sprite.anchor.set(0.5);
    sprite.tint = tint;
    sprite.scale.set(radius / 16);
    sprite.visible = true;
    particleLayer.addChild(sprite);
    particles.push({
      sprite, life, maxLife: life,
      x: x + Math.cos(a) * r, y: y + Math.sin(a) * r,
      vx: Math.cos(dir) * speed, vy: Math.sin(dir) * speed,
    });
  }
}

function stepParticles(dt) {
  if (!particles.length) return;
  const drag = Math.pow(config.worm.consumption_particles.drag, dt);
  let alive = 0;
  for (const p of particles) {
    p.life -= dt;
    if (p.life <= 0) {
      particleLayer.removeChild(p.sprite);
      particlePool.push(p.sprite);
      continue;
    }
    p.x += p.vx * dt;
    p.y += p.vy * dt;
    p.vx *= drag;
    p.vy *= drag;
    p.sprite.position.set(p.x, p.y);
    p.sprite.alpha = p.life / p.maxLife;
    particles[alive++] = p;
  }
  particles.length = alive;
}

// ------------------------------------------------------------------
// Overlays: pill bar, history panels, hourglass
// ------------------------------------------------------------------
const pill = document.getElementById("pill");
const panels = document.getElementById("panels");
const hourglass = document.getElementById("hourglass");
const statusEl = document.getElementById("status");

const ICONS = {
  plus: '<svg viewBox="0 0 10 10"><path d="M5 0v10M0 5h10"/></svg>',
  export: '<svg viewBox="0 0 10 10"><path d="M5 0v9M2 6l3 3 3-3"/></svg>',
  clear: '<svg viewBox="0 0 10 10"><path d="M1 1l8 8M9 1l-8 8"/></svg>',
  quit: '<svg viewBox="0 0 10 10"><rect x="1" y="1" width="8" height="8" fill="#fff"/></svg>',
};
const BUTTONS = [
  { key: "u", icon: "plus", title: "Upload" },
  { key: "i", label: "IN", title: "Record audio input" },
  { key: "o", label: "OUT", title: "Record compost output" },
  { key: "e", icon: "export", title: "Export glues" },
  { key: "c", icon: "clear", title: "Clear" },
  { key: "q", icon: "quit", title: "Quit" },
];
const FLAGS = [
  ["D", "dark", "#b48cff", "Dark mode"],
  ["T", "torus", "#50dcdc", "Torus world"],
  ["H", "history", "#50dc50", "History panels"],
  ["G", "glue", "#dc50dc", "Glue visuals"],
  ["K", "debug", "#ffdc3c", "Debug outlines"],
];

function buildPill() {
  const buttons = BUTTONS.map((b) => {
    const content = b.icon
      ? ICONS[b.icon]
      : `<span class="dot"></span><span class="label">${b.label}</span>`;
    return `<button data-key="${b.key}" title="${b.title}">${content}<span class="key">${b.key.toUpperCase()}</span></button>`;
  }).join("");
  const letters = FLAGS.map(([letter, name, , title]) => `<span data-flag="${name}" title="${title}">${letter}</span>`).join("");
  pill.innerHTML = `${buttons}<span class="divider"></span><span class="flags">${letters}</span>`
    + `<span class="divider"></span><span class="count">0</span>`;
  for (const button of pill.querySelectorAll("button")) {
    button.addEventListener("click", () => pressKey(button.dataset.key));
  }
}

function applyFlags(h) {
  flags = h.flags;
  busy = h.busy;
  document.body.classList.toggle("dark", flags.dark);
  app.renderer.background.color = flags.dark ? 0x000000 : 0xffffff;
  pill.hidden = !flags.ui;
  panels.hidden = !flags.history;
  pill.querySelector('[data-key="i"]')?.classList.toggle("recording", flags.rec_in);
  pill.querySelector('[data-key="o"]')?.classList.toggle("recording", flags.rec_out);
  for (const [, name, color] of FLAGS) {
    const el = pill.querySelector(`[data-flag="${name}"]`);
    if (el) el.style.color = flags[name] ? color : "";
  }
  const count = pill.querySelector(".count");
  if (count) {
    const n = chunks.size;
    count.textContent = n;
    count.classList.toggle("nonzero", n > 0);
  }
  document.title = `Compost — ${h.fps} FPS`;
}

function renderPanels(worms) {
  if (!config) return;
  const pc = config.simulation.ui.worm_history_panel;
  const size = pc.chunk_preview_size;
  const gap = pc.spacing;
  const width = pc.show_hsv_text ? pc.display_area.width : Math.min(pc.display_area.width, gap * 3 + size * 2);
  panels.replaceChildren(...worms.map((worm) => {
    const panel = document.createElement("div");
    panel.className = "panel";
    Object.assign(panel.style, {
      width: `${width}px`,
      height: `${pc.display_area.height}px`,
      padding: `${gap}px`,
      background: css(config.colors.PANEL_BG),
    });
    const header = document.createElement("div");
    header.className = "header";
    header.textContent = worm.header;
    header.style.marginBottom = `${gap}px`;
    panel.appendChild(header);
    for (const row of worm.rows) {
      const line = document.createElement("div");
      line.className = "row";
      line.style.gap = `${gap}px`;
      line.style.marginBottom = `${gap}px`;
      const src = thumbs.get(row.thumb);
      const thumb = document.createElement("canvas");
      thumb.width = thumb.height = size;
      if (src) thumb.getContext("2d").drawImage(src, 0, 0);
      if (row.thumb_border) thumb.style.boxShadow = `inset 0 0 0 2px ${css(row.thumb_border)}`;
      const swatch = document.createElement("div");
      swatch.className = "swatch";
      Object.assign(swatch.style, { width: `${size}px`, height: `${size}px`, background: css(row.rgb) });
      if (row.swatch_border) swatch.style.boxShadow = `inset 0 0 0 2px ${css(row.swatch_border)}`;
      if (row.pct) {
        swatch.textContent = row.pct;
        swatch.style.color = css(row.pct_color);
      }
      line.append(thumb, swatch);
      if (row.hsv) {
        const text = document.createElement("span");
        text.className = "hsv";
        text.textContent = row.hsv;
        line.appendChild(text);
      }
      panel.appendChild(line);
    }
    return panel;
  }));
  for (const id of [...thumbs.keys()]) {
    if (!worms.some((w) => w.rows.some((r) => r.thumb === id))) thumbs.delete(id);
  }
  layoutOverlays();
}

function layoutOverlays() {
  if (!config) return;
  const area = config.simulation.ui.worm_history_panel.display_area;
  panels.style.left = `${area.x}px`;
  panels.style.top = `${area.y}px`;
  const ind = config.simulation.ui.processing_indicator;
  const dpr = window.devicePixelRatio || 1;
  hourglass.width = ind.width * dpr;
  hourglass.height = ind.height * dpr;
  hourglass.style.width = `${ind.width}px`;
  hourglass.style.height = `${ind.height}px`;
  hourglass.style.top = `${ind.margin}px`;
  hourglass.style.right = `${ind.margin}px`;
}

// Animated hourglass while uploads are being segmented
function drawHourglass(t) {
  const show = busy && flags.indicator;
  hourglass.hidden = !show;
  if (!show) return;
  const { width: w, height: h } = config.simulation.ui.processing_indicator;
  const ctx = hourglass.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const midX = Math.floor(w / 2), midY = Math.floor(h / 2);
  ctx.strokeStyle = css(config.colors.WHITE);
  const line = (x0, y0, x1, y1, width) => {
    ctx.lineWidth = width;
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();
  };
  line(0, 1, w, 1, 2);
  line(0, h - 1, w, h - 1, 2);
  line(1, 2, midX, midY, 1);
  line(w - 1, 2, midX, midY, 1);
  line(midX, midY, 1, h - 2, 1);
  line(midX, midY, w - 1, h - 2, 1);

  // Sand drains from the top half into the bottom half every 1.25 s
  const fill = 1 - ((t / 1.25) % 1);
  const halfH = Math.floor(h / 2) - 4;
  const maxHalfW = Math.floor(w / 2) - 3;
  ctx.fillStyle = "rgb(220,190,100)";
  const topRows = Math.floor(fill * halfH);
  for (let row = 0; row < topRows; row++) {
    const hw = Math.floor(maxHalfW * (row / halfH)) + 1;
    ctx.fillRect(midX - hw, midY - 2 - row, hw * 2 + 1, 1);
  }
  const botRows = Math.floor((1 - fill) * halfH);
  for (let row = 0; row < botRows; row++) {
    const hw = Math.floor(maxHalfW * (1 - row / halfH)) + 1;
    ctx.fillRect(midX - hw, h - 3 - row, hw * 2 + 1, 1);
  }
  if (fill > 0.05) {
    const grainY = midY + Math.floor(((t * 3) % 1) * (halfH + 2));
    if (grainY < h - 3 - botRows) ctx.fillRect(midX - 1, grainY - 1, 2, 2);
  }
}

function showStatus(text) {
  statusEl.hidden = !text;
  statusEl.textContent = text || "";
}

// ------------------------------------------------------------------
// Input: keys, mouse (proximity audio), uploads
// ------------------------------------------------------------------
const fileInput = document.getElementById("file-input");

function pressKey(key) {
  if (key === "u") fileInput.click();
  else if (key === "f") toggleFullscreen();
  else if (SERVER_KEYS.has(key)) send({ t: "key", k: key });
}

function toggleFullscreen() {
  if (document.fullscreenElement) document.exitFullscreen();
  else document.documentElement.requestFullscreen();
}

window.addEventListener("keydown", (e) => {
  if (e.metaKey || e.ctrlKey || e.altKey || e.repeat) return;
  pressKey(e.key.toLowerCase());
});

let mouse = null;
let mouseDirty = false;
window.addEventListener("pointermove", (e) => {
  mouse = { x: (e.clientX - fit.x) / fit.s, y: (e.clientY - fit.y) / fit.s };
  mouseDirty = true;
});
document.documentElement.addEventListener("pointerleave", () => {
  mouse = null;
  mouseDirty = true;
});

async function upload(files) {
  for (const file of files) {
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch("/post-compost-upload", { method: "POST", body: form });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(body.error || res.statusText);
    } catch (err) {
      showStatus(`${file.name}: ${err.message}`);
      setTimeout(() => showStatus(null), 4000);
    }
  }
}

fileInput.addEventListener("change", () => {
  upload([...fileInput.files]);
  fileInput.value = "";
});
window.addEventListener("dragover", (e) => {
  e.preventDefault();
  document.body.classList.add("dragging");
});
window.addEventListener("dragleave", (e) => {
  if (!e.relatedTarget) document.body.classList.remove("dragging");
});
window.addEventListener("drop", (e) => {
  e.preventDefault();
  document.body.classList.remove("dragging");
  upload([...e.dataTransfer.files]);
});

// ------------------------------------------------------------------
// Main loop
// ------------------------------------------------------------------
app.ticker.add((ticker) => {
  if (mouseDirty) {
    send({ t: "mouse", x: mouse?.x ?? null, y: mouse?.y ?? null });
    mouseDirty = false;
  }
  if (!config) return;
  stepParticles(ticker.deltaTime);
  drawHourglass(performance.now() / 1000);
});

layout();
connect();
