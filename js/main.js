// Application: WebGPU setup, input, interface and the frame loop.
import { ClothSim, DEFAULT_SETTINGS } from './sim.js';
import { Renderer } from './render.js';
import { FABRICS } from './fabrics.js';
import { SCENE_NAMES, makeScene } from './scenes.js';
import { clamp, invert, lookAt, multiply, perspective, transformPoint, vec3 } from './math.js';

const $ = (id) => document.getElementById(id);

function fail(message) {
  $('error').textContent = message;
  $('error').hidden = false;
  console.error(message);
}

class Camera {
  constructor() { this.target = [0, 0.8, 0]; this.azimuth = 0.6; this.elevation = 0.3; this.distance = 3.5; this.fov = 35; }
  position() {
    const c = Math.cos(this.elevation);
    return vec3.add(this.target, vec3.scale([Math.sin(this.azimuth) * c, Math.sin(this.elevation), Math.cos(this.azimuth) * c], this.distance));
  }
}

async function main() {
  if (!navigator.gpu) return fail('This browser has no WebGPU. Use Chrome or Edge 113 or later, or Safari 26.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return fail('No WebGPU capable GPU was found.');
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBuffersPerShaderStage: Math.min(adapter.limits.maxStorageBuffersPerShaderStage, 10),
    },
  });
  device.lost.then((info) => fail(`The GPU device was lost: ${info.message}`));
  device.addEventListener('uncapturederror', (e) => console.error('WebGPU:', e.error.message));

  const canvas = $('canvas');
  const context = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });

  const sim = new ClothSim(device);
  const renderer = new Renderer(device, context, format);
  const errors = [...(await sim.checkShaders()), ...(await renderer.checkShaders())];
  if (errors.length) return fail('Shader error:\n' + errors.join('\n'));

  const params = new URLSearchParams(location.search);
  const state = {
    sceneIndex: clamp(parseInt(params.get('scene') ?? '0', 10) || 0, 0, SCENE_NAMES.length - 1),
    fabric: parseInt(params.get('fabric') ?? '-1', 10),
    resolution: parseInt(params.get('res') ?? '128', 10) || 128,
    settings: { ...DEFAULT_SETTINGS },
    paused: false,
    time: 0,
    accumulator: 0,
    grab: { active: false, target: [0, 0, 0], rest: [0, 0], cloth: 0, particle: 0, radius: 0.05, stiffness: 1, depth: 0 },
    picking: false,
  };
  const camera = new Camera();
  let scene, fabrics;

  function loadScene() {
    scene = makeScene(state.sceneIndex, state.resolution, state.fabric);
    fabrics = scene.cloths.map((c) => FABRICS[c.fabric]);
    sim.build(scene);
    renderer.setScene(scene, sim);
    camera.target = [...scene.cameraTarget];
    camera.azimuth = scene.cameraAzimuth;
    camera.elevation = scene.cameraElevation;
    camera.distance = scene.cameraDistance;
    camera.fov = scene.cameraFov;
    state.time = 0;
    state.accumulator = 0;
    state.grab.active = false;
    $('info').textContent = `${sim.particleCount.toLocaleString('en-US')} particles, ${sim.triangleCount.toLocaleString('en-US')} triangles`;
  }

  // ---- interface
  const sceneSelect = $('scene'), fabricSelect = $('fabric'), resSelect = $('res');
  SCENE_NAMES.forEach((n, i) => sceneSelect.add(new Option(n, i)));
  fabricSelect.add(new Option('Scene default', -1));
  FABRICS.forEach((f, i) => fabricSelect.add(new Option(f.name, i)));
  for (const r of [64, 96, 128, 192, 256]) resSelect.add(new Option(`${r} × ${r}`, r));
  sceneSelect.value = state.sceneIndex;
  fabricSelect.value = state.fabric;
  resSelect.value = state.resolution;
  const reload = () => { loadScene(); };
  sceneSelect.onchange = () => { state.sceneIndex = +sceneSelect.value; reload(); };
  fabricSelect.onchange = () => { state.fabric = +fabricSelect.value; reload(); };
  resSelect.onchange = () => { state.resolution = +resSelect.value; reload(); };
  $('self').onchange = (e) => { state.settings.selfCollision = e.target.checked; };
  $('wind').oninput = (e) => { state.settings.windScale = +e.target.value; };
  $('pause').onclick = () => { state.paused = !state.paused; $('pause').textContent = state.paused ? 'Resume' : 'Pause'; };
  $('reset').onclick = () => { sim.reset(); state.time = 0; state.accumulator = 0; };
  $('hide').onclick = () => { $('panel').classList.toggle('collapsed'); };
  window.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
    if (e.code === 'Space') { $('pause').click(); e.preventDefault(); }
    if (e.code === 'KeyR') $('reset').click();
  });

  // ---- input: drag the cloth with the left button, orbit otherwise
  function cursorRay(x, y) {
    const aspect = canvas.width / canvas.height;
    const eye = camera.position();
    const viewProj = multiply(perspective((camera.fov * Math.PI) / 180, aspect, 0.05, 400), lookAt(eye, camera.target, [0, 1, 0]));
    const rect = canvas.getBoundingClientRect();
    const ndc = [((x - rect.left) / rect.width) * 2 - 1, 1 - ((y - rect.top) / rect.height) * 2];
    const far = transformPoint(invert(viewProj), [ndc[0], ndc[1], 1]);
    return { origin: eye, dir: vec3.normalize(vec3.sub(far, eye)) };
  }
  let drag = null;
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  canvas.addEventListener('pointerdown', async (e) => {
    try { canvas.setPointerCapture(e.pointerId); } catch { /* synthetic events have no capturable pointer */ }
    const mode = e.button === 2 || e.ctrlKey ? 'orbit' : e.button === 1 || e.shiftKey ? 'pan' : 'grab';
    drag = { mode, x: e.clientX, y: e.clientY, id: e.pointerId };
    if (mode !== 'grab' || state.picking) return;
    state.picking = true;
    const pos = await sim.readPositions();
    state.picking = false;
    if (!drag || drag.id !== e.pointerId) return;
    const { origin, dir } = cursorRay(drag.x, drag.y);
    let best = 1e9, bestIndex = -1, bestDepth = 0;
    for (let i = 0; i < pos.length / 4; ++i) {
      const d = [pos[i * 4] - origin[0], pos[i * 4 + 1] - origin[1], pos[i * 4 + 2] - origin[2]];
      const t = vec3.dot(d, dir);
      if (t <= 0) continue;
      const dist = vec3.length(vec3.sub(d, vec3.scale(dir, t)));
      const score = dist + t * 0.002;
      if (score < best) { best = score; bestIndex = i; bestDepth = t; }
    }
    if (bestIndex < 0 || best > 0.08 * camera.distance) {
      drag.mode = 'orbit';
      return;
    }
    const rest = sim.data.rest[bestIndex];
    Object.assign(state.grab, { active: true, rest: [rest[0], rest[1]], cloth: rest[2], particle: bestIndex, depth: bestDepth,
      radius: Math.max(sim.data.ranges[rest[2]].spacing * 4, 0.03) });
    const ray = cursorRay(drag.x, drag.y);
    state.grab.target = vec3.add(ray.origin, vec3.scale(ray.dir, bestDepth));
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!drag || drag.id !== e.pointerId) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    drag.x = e.clientX;
    drag.y = e.clientY;
    if (drag.mode === 'orbit') {
      camera.azimuth -= dx * 0.005;
      camera.elevation = clamp(camera.elevation + dy * 0.005, -1.5, 1.5);
    } else if (drag.mode === 'pan') {
      const view = lookAt(camera.position(), camera.target, [0, 1, 0]);
      const right = [view[0], view[4], view[8]], up = [view[1], view[5], view[9]];
      camera.target = vec3.add(camera.target, vec3.scale(vec3.add(vec3.scale(right, -dx), vec3.scale(up, dy)), camera.distance * 0.0012));
    } else if (state.grab.active) {
      const ray = cursorRay(e.clientX, e.clientY);
      state.grab.target = vec3.add(ray.origin, vec3.scale(ray.dir, state.grab.depth));
    }
  });
  const endDrag = (e) => {
    if (drag && drag.id === e.pointerId) drag = null;
    state.grab.active = false;
  };
  canvas.addEventListener('pointerup', endDrag);
  canvas.addEventListener('pointercancel', endDrag);
  canvas.addEventListener('wheel', (e) => {
    camera.distance = clamp(camera.distance * Math.pow(0.9, -e.deltaY / 100), 0.2, 60);
    e.preventDefault();
  }, { passive: false });

  loadScene();

  // ---- frame loop: substeps of fixed length, as many as fit in the time since the last frame (at most 48)
  let last = performance.now();
  let smoothMs = 16;
  let lastStats = 0;
  function frame(now) {
    const elapsed = Math.min((now - last) / 1000, 0.1);
    last = now;
    smoothMs += (elapsed * 1000 - smoothMs) * 0.05;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(canvas.clientWidth * dpr)), h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    renderer.resize(canvas.width, canvas.height);

    const nominal = state.settings.substeps;
    const substepLength = 1 / 60 / nominal;
    let steps = 0;
    if (!state.paused) {
      state.accumulator += elapsed;
      steps = Math.floor(state.accumulator / substepLength);
      state.accumulator -= steps * substepLength;
      steps = Math.min(steps, nominal * 2);
    }
    const encoder = device.createCommandEncoder();
    if (steps > 0) {
      const dt = steps * substepLength;
      sim.prepareFrame(scene, fabrics, state.settings, state.grab, state.time, dt, steps);
      sim.encodeStep(encoder, state.settings, scene, steps, dt);
      state.time += dt;
    } else if (!sim.normalsReady) {
      sim.prepareFrame(scene, fabrics, state.settings, state.grab, state.time, 0, 1);
      sim.encodeNormals(encoder);
      sim.normalsReady = true;
    }
    renderer.render(encoder, context.getCurrentTexture().createView(), sim, scene, fabrics, camera, state.time);
    device.queue.submit([encoder.finish()]);

    if (now - lastStats > 250) {
      lastStats = now;
      $('stats').textContent = `${(1000 / smoothMs).toFixed(0)} fps · ${smoothMs.toFixed(1)} ms · ${steps} substeps this frame`;
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

main().catch((e) => fail(e.stack || String(e)));
