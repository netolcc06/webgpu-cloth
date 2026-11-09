// The eight scenes. A cloth is a grid in material space (metres), placed in the world by a function.
import { gamma, identity, multiply, quat, rotationY, scaling, translation, clamp, mix } from './math.js';
import { FABRICS, findFabric } from './fabrics.js';

export const COLLIDER = { SPHERE: 0, CAPSULE: 1, BOX: 2, CYLINDER: 3, TORUS: 4 };
export const ENV = { STUDIO: 0, DAYLIGHT: 1 };
export const SCENE_NAMES = ['Drape', 'Round table', 'Flag', 'Curtain', 'Layered sheets', 'Twirling skirt', 'Fabric swatches', 'Ball impact'];

const PI = Math.PI;

function collider(fields) {
  return Object.assign({ type: COLLIDER.SPHERE, size: [0.5, 0, 0, 0], position: [0, 0, 0], rotation: quat.identity(), friction: 1.0,
    color: [0.5, 0.5, 0.5], roughness: 0.5, metallic: 0, animate: null }, fields);
}

function light(dir, color, intensity, angular) {
  const l = Math.hypot(...dir);
  return { direction: dir.map((v) => v / l), color, intensity, angular };
}

function baseScene(name) {
  return { name, cloths: [], colliders: [], lights: [], pinGroups: [], wind: [0, 0, 0], turbulence: 0, windFrequency: 0.8,
    selfCollision: true, selfScale: 1.0, environment: ENV.STUDIO, environmentIntensity: 1.0, shadowRadius: 0,
    floorColor: [0.3, 0.3, 0.3], floorRoughness: 0.65, floorFade: [3.5, 18], cameraTarget: [0, 0.8, 0], cameraAzimuth: 0.6,
    cameraElevation: 0.3, cameraDistance: 3.5, cameraFov: 35 };
}

function cloth(fields) {
  return Object.assign({ nu: 128, nv: 128, width: 1, height: 1, wrapU: false, fabric: 0, place: null, pin: null, cutRadius: 0, layer: 0 }, fields);
}

const fabricOr = (override, name) => (override >= 0 ? override : findFabric(name));

function horizontalSheet(res, size, height, tilt, fabric) {
  return cloth({ nu: res, nv: res, width: size, height: size, fabric, place: (u, v) => {
    const x = u - size * 0.5, z = size * 0.5 - v;
    return [x, height + tilt * (x * 0.6 + z * 0.8), z];
  } });
}

function drape(res, fo) {
  const s = baseScene('Drape');
  s.cloths.push(horizontalSheet(res, 2.2, 1.55, 0.04, fabricOr(fo, 'Velvet')));
  s.colliders.push(collider({ type: COLLIDER.SPHERE, size: [0.5, 0, 0, 0], position: [0, 0.9, 0], color: [0.32, 0.32, 0.32], roughness: 0.45 }));
  s.colliders.push(collider({ type: COLLIDER.CYLINDER, size: [0.05, 0.45, 0, 0.01], position: [0, 0.45, 0], color: [0.1, 0.1, 0.1], roughness: 0.4, metallic: 1 }));
  s.lights.push(light([0.55, 0.85, 0.45], [1.0, 0.96, 0.9], 2.4, 0.06), light([-0.7, 0.35, -0.6], [0.75, 0.85, 1.0], 1.4, 0.1));
  s.cameraTarget = [0, 0.62, 0]; s.cameraDistance = 4.2; s.cameraElevation = 0.3;
  return s;
}

function table(res, fo) {
  const s = baseScene('Round table');
  const c = horizontalSheet(res, 2.6, 0.86, 0.004, fabricOr(fo, 'Silk Damask'));
  c.cutRadius = 1.25;
  s.cloths.push(c);
  const wood = gamma(0.28, 0.16, 0.09);
  s.colliders.push(collider({ type: COLLIDER.CYLINDER, size: [0.72, 0.02, 0, 0.015], position: [0, 0.76, 0], color: wood, roughness: 0.4, friction: 1.5 }));
  s.colliders.push(collider({ type: COLLIDER.CYLINDER, size: [0.05, 0.38, 0, 0.01], position: [0, 0.38, 0], color: wood, roughness: 0.4 }));
  s.lights.push(light([0.4, 0.85, 0.55], [1.0, 0.95, 0.88], 3.2, 0.05), light([-0.8, 0.3, -0.5], [0.8, 0.87, 1.0], 1.0, 0.1));
  s.floorColor = gamma(0.55, 0.53, 0.5);
  s.cameraTarget = [0, 0.42, 0]; s.cameraDistance = 4.2; s.cameraElevation = 0.32;
  return s;
}

function flag(res, fo) {
  const s = baseScene('Flag');
  const width = 2.1, height = 1.4, poleX = -1.0, top = 3.4;
  s.cloths.push(cloth({ nu: res, nv: Math.max(8, Math.floor(res * height / width)), width, height, fabric: fabricOr(fo, 'Striped Oxford'),
    place: (u, v) => [poleX + u, top - height + v, 0.02 * Math.sin(u * 3)],
    pin: (u, v, i, j) => (i === 0 && j % 2 === 0 ? 0 : -1) }));
  s.colliders.push(collider({ type: COLLIDER.CAPSULE, size: [0.025, 1.9, 0, 0], position: [poleX - 0.03, 1.9, 0], color: [0.85, 0.85, 0.85], roughness: 0.25, metallic: 1 }));
  s.wind = [6.5, 0, 0.9]; s.turbulence = 2.8; s.windFrequency = 0.9; s.selfScale = 0.9; s.environment = ENV.DAYLIGHT;
  s.lights.push(light([0.45, 0.75, 0.5], [1.0, 0.95, 0.85], 5.0, 0.012));
  s.floorColor = gamma(0.36, 0.42, 0.24); s.floorRoughness = 0.95; s.floorFade = [25, 90];
  s.cameraTarget = [0.2, 2.6, 0]; s.cameraDistance = 5.2; s.cameraElevation = -0.08; s.cameraAzimuth = 0.25;
  return s;
}

function curtain(res, fo) {
  const s = baseScene('Curtain');
  const width = 4.2, railWidth = 2.6, height = 2.5, top = 2.7, pleats = 11;
  const nu = res, nv = Math.max(8, Math.floor(res * height / width));
  s.cloths.push(cloth({ nu, nv, width, height, fabric: fabricOr(fo, 'Cotton Voile'),
    place: (u, v) => {
      const a = u / width, x = (a - 0.5) * railWidth;
      const fold = Math.cos(a * pleats * 2 * PI) * 0.5 + 0.5;
      const amplitude = 0.5 * Math.sqrt(width * width - railWidth * railWidth) / pleats;
      return [x, top - height + v, amplitude * (1 - fold)];
    },
    pin: (u, v, i, j) => {
      if (j !== nv - 1) return -1;
      const hooks = (nu - 1) / pleats;
      return (i + 0.5) % hooks < 1 ? 1 : -1;
    } }));
  s.pinGroups.push(() => identity());
  s.pinGroups.push((t) => {
    const phase = clamp(0.5 + 0.5 * Math.sin(t * 0.35 - 1.2), 0, 1);
    return scaling([mix(1, 0.82, phase), 1, 1]);
  });
  const railRot = quat.angleAxis(PI / 2, [0, 0, 1]);
  s.colliders.push(collider({ type: COLLIDER.CAPSULE, size: [0.015, 1.5, 0, 0], position: [0, top + 0.03, -0.02], rotation: railRot, color: [0.12, 0.12, 0.12], metallic: 1, roughness: 0.35 }));
  const plaster = gamma(0.80, 0.74, 0.66);
  const wall = (center, half) => s.colliders.push(collider({ type: COLLIDER.BOX, size: [...half, 0.004], position: center, color: plaster, roughness: 0.8 }));
  const wallZ = -0.42, th = 0.09, wh = 1.2, windowTop = 2.55, sill = 0.55, wallTop = 3.6, far = 5.0;
  wall([-(wh + far), wallTop * 0.5, wallZ], [far, wallTop * 0.5, th]);
  wall([wh + far, wallTop * 0.5, wallZ], [far, wallTop * 0.5, th]);
  wall([0, (wallTop + windowTop) * 0.5, wallZ], [wh, (wallTop - windowTop) * 0.5, th]);
  wall([0, sill * 0.5, wallZ], [wh, sill * 0.5, th]);
  s.colliders.push(collider({ type: COLLIDER.BOX, size: [70, 0.02, 30, 0.005], position: [0, 0, wallZ - th - 30], color: gamma(0.34, 0.46, 0.2), roughness: 0.95 }));
  s.environment = ENV.DAYLIGHT; s.shadowRadius = 7; s.wind = [0, 0, 0.1]; s.turbulence = 0.22; s.windFrequency = 1.1; s.selfScale = 0.8;
  s.lights.push(light([-0.28, 0.62, -0.75], [1.0, 0.94, 0.84], 6.0, 0.012), light([0.5, 0.7, 0.8], [1.0, 0.92, 0.82], 1.1, 0.15));
  s.environmentIntensity = 0.8; s.floorColor = gamma(0.34, 0.25, 0.18); s.floorRoughness = 0.35; s.floorFade = [30, 90];
  s.cameraTarget = [0, 1.45, 0]; s.cameraDistance = 6.0; s.cameraElevation = 0.03; s.cameraAzimuth = 0.3;
  return s;
}

function layered(res, fo) {
  const s = baseScene('Layered sheets');
  const r = Math.max(32, Math.floor(res * 0.5));
  const names = ['Velvet', 'Silk Charmeuse', 'Wool Tartan', 'Raw Denim'];
  for (let i = 0; i < 4; ++i) {
    const c = horizontalSheet(r, 1.6, 0.9 + i * 0.25, 0.03, fabricOr(fo, names[i]));
    c.layer = i;
    const angle = i * 0.5, shift = [0.12 * Math.cos(i * 2.4), 0, 0.12 * Math.sin(i * 2.4)];
    const basePlace = c.place;
    c.place = (u, v) => {
      const p = basePlace(u, v);
      return [p[0] * Math.cos(angle) - p[2] * Math.sin(angle) + shift[0], p[1], p[0] * Math.sin(angle) + p[2] * Math.cos(angle) + shift[2]];
    };
    s.cloths.push(c);
  }
  s.colliders.push(collider({ type: COLLIDER.BOX, size: [0.32, 0.2, 0.32, 0.03], position: [0, 0.2, 0], rotation: quat.angleAxis(0.5, [0, 1, 0]), color: gamma(0.6, 0.55, 0.48), roughness: 0.6 }));
  s.colliders.push(collider({ type: COLLIDER.SPHERE, size: [0.22, 0, 0, 0], position: [0.1, 0.6, 0.05], color: [0.65, 0.65, 0.65], roughness: 0.3 }));
  s.lights.push(light([0.5, 0.85, 0.4], [1.0, 0.96, 0.9], 3.0, 0.06), light([-0.7, 0.3, -0.6], [0.8, 0.87, 1.0], 1.0, 0.1));
  s.cameraTarget = [0, 0.35, 0]; s.cameraDistance = 3.3; s.cameraElevation = 0.42;
  return s;
}

function skirt(res, fo) {
  const s = baseScene('Twirling skirt');
  const waistRadius = 0.175, hemRadius = 0.44, length = 0.85, waistY = 1.05;
  const circumference = 2 * PI * hemRadius * 0.9;
  const nv = Math.max(16, Math.floor(res * length / circumference));
  s.cloths.push(cloth({ nu: res, nv, width: circumference, height: length, wrapU: true, fabric: fabricOr(fo, 'Silk Charmeuse'),
    place: (u, v) => {
      const a = u / circumference * 2 * PI, t = 1 - v / length;
      const radius = mix(waistRadius, hemRadius, t), wave = 1 + 0.1 * Math.cos(a * 9) * t;
      return [-Math.sin(a) * radius * wave, waistY - t * length * 0.92, -Math.cos(a) * radius * wave];
    },
    pin: (u, v, i, j) => (j === nv - 1 ? 1 : -1) }));
  const hipTurn = (t) => 1.1 * Math.sin(t * 1.5) + 0.55 * Math.sin(t * 2.6 + 0.7);
  s.pinGroups.push(() => identity());
  s.pinGroups.push((t) => multiply(translation([0.05 * Math.sin(t * 1.5), 0, 0]), rotationY(hipTurn(t))));
  const body = (base) => (t) => {
    const turn = quat.angleAxis(hipTurn(t), [0, 1, 0]);
    const p = quat.rotate(turn, base);
    return { position: [p[0] + 0.05 * Math.sin(t * 1.5), p[1], p[2]], rotation: turn };
  };
  const linen = gamma(0.72, 0.67, 0.56), wood = gamma(0.20, 0.13, 0.08);
  const add = (type, size, base, color, roughness) => s.colliders.push(collider({ type, size, position: base, color, roughness, animate: body(base) }));
  add(COLLIDER.SPHERE, [0.175, 0, 0, 0], [0, 0.93, 0], linen, 0.85);
  add(COLLIDER.CAPSULE, [0.125, 0.11, 0, 0], [0, 1.21, 0], linen, 0.85);
  add(COLLIDER.CAPSULE, [0.15, 0.07, 0, 0], [0, 1.42, 0], linen, 0.85);
  add(COLLIDER.CYLINDER, [0.045, 0.06, 0, 0.012], [0, 1.62, 0], wood, 0.35);
  add(COLLIDER.CYLINDER, [0.018, 0.41, 0, 0.004], [0, 0.42, 0], wood, 0.35);
  add(COLLIDER.CYLINDER, [0.2, 0.018, 0, 0.012], [0, 0.018, 0], wood, 0.35);
  s.selfScale = 0.8;
  s.lights.push(light([0.5, 0.8, 0.55], [1.0, 0.96, 0.9], 3.0, 0.05), light([-0.8, 0.4, -0.6], [0.75, 0.85, 1.0], 1.8, 0.08));
  s.cameraTarget = [0, 0.75, 0]; s.cameraDistance = 4.2; s.cameraElevation = 0.1;
  return s;
}

function swatches(res, fo) {
  const s = baseScene('Fabric swatches');
  const count = fo >= 0 ? 1 : FABRICS.length;
  const perRow = 7, width = 0.56, height = 0.8, gapX = 0.12, gapY = 0.26;
  const rows = Math.ceil(count / perRow);
  const r = Math.max(32, Math.floor(res / 4));
  const totalWidth = perRow * width + (perRow - 1) * gapX;
  const topY = 0.25 + rows * height + (rows - 1) * gapY;
  for (let k = 0; k < count; ++k) {
    const row = Math.floor(k / perRow), col = k % perRow;
    const x0 = -totalWidth * 0.5 + col * (width + gapX), top = topY - row * (height + gapY);
    const nu = r, nv = Math.floor(r * height / width);
    const offsetZ = 0.02 * (k % 2), gather = 0.9;
    const hookEvery = Math.max(2, Math.floor((nu - 1) / 6));
    s.cloths.push(cloth({ nu, nv, width, height, fabric: fo >= 0 ? fo : k,
      place: (u, v) => [x0 + width * 0.5 * (1 - gather) + u * gather, top - height + v, offsetZ + 0.02 * Math.sin((u / width) * 6 * PI)],
      pin: (u, v, i, j) => (j === nv - 1 && i % hookEvery === 0 ? 0 : -1) }));
  }
  const railRot = quat.angleAxis(PI / 2, [0, 0, 1]);
  for (let row = 0; row < rows; ++row)
    s.colliders.push(collider({ type: COLLIDER.CAPSULE, size: [0.012, totalWidth * 0.5 + 0.1, 0, 0], position: [0, topY - row * (height + gapY) + 0.018, -0.02],
      rotation: railRot, color: [0.15, 0.15, 0.15], metallic: 1, roughness: 0.3 }));
  s.wind = [0.03, 0, 0.08]; s.turbulence = 0.2; s.windFrequency = 1.6; s.selfCollision = false;
  s.lights.push(light([0.35, 0.65, 0.75], [1.0, 0.96, 0.9], 3.2, 0.05), light([-0.7, 0.25, -0.65], [0.75, 0.85, 1.0], 2.0, 0.1));
  s.cameraTarget = [0, topY * 0.5 + 0.1, 0]; s.cameraDistance = 2.6 + 1.15 * rows; s.cameraElevation = 0.02; s.cameraAzimuth = 0;
  return s;
}

function ballImpact(res, fo) {
  const s = baseScene('Ball impact');
  const c = horizontalSheet(res, 2.0, 1.0, 0.0, fabricOr(fo, 'Canvas'));
  const nu = c.nu, nv = c.nv;
  c.pin = (u, v, i, j) => {
    const cu = i < Math.floor(nu / 20) || i >= nu - Math.floor(nu / 20);
    const cv = j < Math.floor(nv / 20) || j >= nv - Math.floor(nv / 20);
    return cu && cv ? 0 : -1;
  };
  s.cloths.push(c);
  s.pinGroups.push((t) => {
    const x = clamp(t, 0, 1);
    const k = mix(1, 0.86, x * x * (3 - 2 * x));
    return scaling([k, 1, k]);
  });
  s.colliders.push(collider({ type: COLLIDER.SPHERE, size: [0.28, 0, 0, 0], position: [0, 2, 0], color: gamma(0.72, 0.14, 0.09), roughness: 0.2,
    animate: (t) => {
      const cycle = ((t + 1) % 3) / 3, down = Math.sin(cycle * PI);
      return { position: [0.15 * Math.sin(t * 0.7), 2.1 - 1.45 * down * down, 0.1 * Math.cos(t * 0.5)], rotation: quat.identity() };
    } }));
  for (let i = 0; i < 4; ++i)
    s.colliders.push(collider({ type: COLLIDER.CYLINDER, size: [0.03, 0.5, 0, 0.005], position: [i < 2 ? -0.86 : 0.86, 0.5, i % 2 === 0 ? -0.86 : 0.86],
      color: [0.1, 0.1, 0.1], metallic: 1, roughness: 0.3 }));
  s.lights.push(light([0.5, 0.8, 0.5], [1.0, 0.96, 0.9], 3.0, 0.05), light([-0.7, 0.3, -0.6], [0.75, 0.85, 1.0], 1.0, 0.1));
  s.cameraTarget = [0, 0.85, 0]; s.cameraDistance = 4.8; s.cameraElevation = 0.22;
  return s;
}

export function makeScene(index, res, fabricOverride = -1) {
  const s = [drape, table, flag, curtain, layered, skirt, swatches, ballImpact][index](res, fabricOverride);
  if (s.pinGroups.length === 0) s.pinGroups.push(() => identity());
  return s;
}

// Pose of a collider at time t.
export function colliderPose(c, t) {
  if (c.animate) return c.animate(t);
  return { position: c.position, rotation: c.rotation };
}
