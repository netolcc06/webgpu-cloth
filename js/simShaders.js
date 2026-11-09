// WGSL compute shaders of the solver. Each entry lists its storage buffers in binding order (group 1); the
// declarations are generated from that list, so that the WGSL and the bind group layout cannot disagree.
// Group 0 holds the frame uniform and the per-dispatch parameters (a dynamic offset replaces push constants).

export const PRELUDE = /* wgsl */ `
struct Phys { stretch: vec4f, bend: vec4f, contact: vec4f, air: vec4f, extra: vec4f };
struct Collider { posA: vec4f, rotA: vec4f, posB: vec4f, rotB: vec4f, size: vec4f, angVel: vec4f };
struct Frame {
  gravityDt: vec4f, wind: vec4f, windParams: vec4f, selfRebuild: vec4f, selfDepth: vec4f, selfp: vec4f,
  grabTarget: vec4f, grabRest: vec4f, misc: vec4f, counts: vec4u, grabIndex: vec4u,
  phys: array<Phys, 20>, colliders: array<Collider, 16>, pinsA: array<mat4x4f, 8>, pinsB: array<mat4x4f, 8>,
};
struct Params { first: u32, count: u32, dtEff: f32, t: f32 };
struct StretchTri { idx: vec4u, grad: vec4f, extra: vec4f };
struct BendEdge { idx: vec4u, K: vec4f, extra: vec4f };
struct LimitEdge { a: u32, b: u32, rest: f32, cloth: u32 };
struct Lra { pin: u32, maxDistance: f32 };

@group(0) @binding(0) var<uniform> sim: Frame;
@group(0) @binding(1) var<uniform> pc: Params;

const NEIGH_STRIDE = 33u;
const MAX_NEIGHBORS = 32u;
const CELL_CAPACITY = 48u;
const SIDE_BIT = 0x80000000u;
const INDEX_MASK = 0x7fffffffu;

fn cellOf(p: vec3f) -> vec3i { return vec3i(floor(p / sim.selfp.w)); }

fn cellHash(c: vec3i) -> u32 {
  var h = (bitcast<u32>(c.x) * 73856093u) ^ (bitcast<u32>(c.y) * 19349663u) ^ (bitcast<u32>(c.z) * 83492791u);
  h = h ^ (h >> 13u);
  return h & sim.counts.y;
}

fn rotateByQuat(q: vec4f, v: vec3f) -> vec3f { return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v); }
fn quatConjugate(q: vec4f) -> vec4f { return vec4f(-q.xyz, q.w); }
`;

const ENTRY = '@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid: vec3u)';

export const SHADERS = {
  // Velocity from the positions of the last solve, with a weak membrane viscosity that removes mesh scale
  // oscillations (the heavy-ball iteration of one solver pass per substep sustains them).
  velocity: {
    buffers: [['pos', 'r', 'vec4f'], ['prev', 'r', 'vec4f'], ['vel', 'rw', 'vec4f'], ['gridNeighbors', 'r', 'vec4u']],
    code: `${ENTRY} {
  let i = gid.x;
  if (i >= pc.count) { return; }
  let dt = sim.gravityDt.w;
  let vi = (pos[i].xyz - prev[i].xyz) / dt;
  let n = gridNeighbors[i];
  var sum = vec3f(0.0);
  var count = 0.0;
  for (var k = 0u; k < 4u; k++) {
    let j = n[k];
    if (j != i) {
      sum += (pos[j].xyz - prev[j].xyz) / dt - vi;
      count += 1.0;
    }
  }
  let viscosity = f32(sim.counts.w) * 0.0001;
  let toMean = sum / max(count, 1.0);
  let noise = select(0.0, max(1.0 - length(toMean) / 0.12, 0.0), viscosity > 0.0);
  vel[i] = vec4f(vi + toMean * clamp(4.0 * viscosity + 0.45 * noise, 0.0, 1.0), 0.0);
}`,
  },

  // Explicit prediction. Pinned particles follow their group; the grabbed patch has no inertia of its own.
  integrate: {
    buffers: [['initPos', 'r', 'vec4f'], ['rest', 'r', 'vec4f'], ['accel', 'r', 'vec4f'], ['pos', 'rw', 'vec4f'], ['prev', 'rw', 'vec4f'], ['vel', 'rw', 'vec4f']],
    code: `${ENTRY} {
  let i = gid.x;
  if (i >= pc.count) { return; }
  let dt = sim.gravityDt.w;
  let init = initPos[i];
  let x = pos[i].xyz;
  if (init.w >= 0.0) {
    let g = u32(init.w);
    let goal = mix((sim.pinsA[g] * vec4f(init.xyz, 1.0)).xyz, (sim.pinsB[g] * vec4f(init.xyz, 1.0)).xyz, pc.t);
    prev[i] = vec4f(x, 0.0);
    pos[i] = vec4f(goal, 0.0);
    vel[i] = vec4f((goal - x) / dt, 0.0);
    return;
  }
  let r = rest[i];
  var w = 1.0 / max(r.w * sim.phys[u32(r.z)].air.x, 1e-9);
  var v = vel[i].xyz;
  v *= max(0.0, 1.0 - sim.misc.x * dt);
  v += dt * (sim.gravityDt.xyz + accel[i].xyz);
  if (sim.grabTarget.w > 0.0 && r.z == sim.grabRest.z) {
    let dm = r.xy - sim.grabRest.xy;
    let falloff = 1.0 - dot(dm, dm) / (sim.grabTarget.w * sim.grabTarget.w);
    if (falloff > 0.0) {
      let keep = 1.0 - falloff * falloff;
      w *= keep;
      v *= keep;
    }
  }
  let speed = length(v);
  if (speed > sim.windParams.w) { v *= sim.windParams.w / speed; }
  vel[i] = vec4f(v, 0.0);
  prev[i] = vec4f(x, 0.0);
  pos[i] = vec4f(x + dt * v, w);
}`,
  },

  // Membrane: strain based XPBD (Mueller et al. 2014), warp, weft and shear with their own compliance.
  stretch: {
    buffers: [['stretchTris', 'r', 'StretchTri'], ['prev', 'r', 'vec4f'], ['pos', 'rw', 'vec4f']],
    code: `${ENTRY} {
  let id = gid.x;
  if (id >= pc.count) { return; }
  let tri = stretchTris[pc.first + id];
  let i0 = tri.idx.x; let i1 = tri.idx.y; let i2 = tri.idx.z;
  let p0 = pos[i0]; let p1 = pos[i1]; let p2 = pos[i2];
  var x0 = p0.xyz; var x1 = p1.xyz; var x2 = p2.xyz;
  let w0 = p0.w; let w1 = p1.w; let w2 = p2.w;
  let m = sim.phys[tri.idx.w];
  let dt = pc.dtEff;
  let s1 = tri.grad.x; let s2 = tri.grad.y; let s0 = -s1 - s2;
  let t1 = tri.grad.z; let t2 = tri.grad.w; let t0 = -t1 - t2;
  let invAreaDt2 = 1.0 / (tri.extra.x * dt * dt);
  let gamma = m.bend.z / dt;
  let v0 = x0 - prev[i0].xyz; let v1 = x1 - prev[i1].xyz; let v2 = x2 - prev[i2].xyz;
  {
    let Fv = t0 * x0 + t1 * x1 + t2 * x2;
    let len = length(Fv);
    if (len > 1e-7) {
      let n = Fv / len;
      let C = len - 1.0;
      let W = w0 * t0 * t0 + w1 * t1 * t1 + w2 * t2 * t2;
      let alpha = m.stretch.x * invAreaDt2 * select(1.0, m.stretch.w, C < 0.0);
      let Cdot = dot(n, t0 * v0 + t1 * v1 + t2 * v2);
      let dl = -(C + gamma * Cdot) / ((1.0 + gamma) * W + alpha + 1e-12);
      x0 += w0 * t0 * dl * n; x1 += w1 * t1 * dl * n; x2 += w2 * t2 * dl * n;
    }
  }
  {
    let Fu = s0 * x0 + s1 * x1 + s2 * x2;
    let len = length(Fu);
    if (len > 1e-7) {
      let n = Fu / len;
      let C = len - 1.0;
      let W = w0 * s0 * s0 + w1 * s1 * s1 + w2 * s2 * s2;
      let alpha = m.stretch.y * invAreaDt2 * select(1.0, m.stretch.w, C < 0.0);
      let Cdot = dot(n, s0 * v0 + s1 * v1 + s2 * v2);
      let dl = -(C + gamma * Cdot) / ((1.0 + gamma) * W + alpha + 1e-12);
      x0 += w0 * s0 * dl * n; x1 += w1 * s1 * dl * n; x2 += w2 * s2 * dl * n;
    }
  }
  {
    let Fu = s0 * x0 + s1 * x1 + s2 * x2;
    let Fv = t0 * x0 + t1 * x1 + t2 * x2;
    let C = dot(Fu, Fv);
    let g0 = s0 * Fv + t0 * Fu; let g1 = s1 * Fv + t1 * Fu; let g2 = s2 * Fv + t2 * Fu;
    let W = w0 * dot(g0, g0) + w1 * dot(g1, g1) + w2 * dot(g2, g2);
    let alpha = m.stretch.z * invAreaDt2;
    let Cdot = dot(g0, v0) + dot(g1, v1) + dot(g2, v2);
    let dl = -(C + gamma * Cdot) / ((1.0 + gamma) * W + alpha + 1e-12);
    x0 += w0 * dl * g0; x1 += w1 * dl * g1; x2 += w2 * dl * g2;
  }
  pos[i0] = vec4f(x0, w0);
  pos[i1] = vec4f(x1, w1);
  pos[i2] = vec4f(x2, w2);
}`,
  },

  // Bending: the isometric model (Bergou et al. 2006) as a linear vector constraint, exact in XPBD.
  bend: {
    buffers: [['bendEdges', 'r', 'BendEdge'], ['prev', 'r', 'vec4f'], ['pos', 'rw', 'vec4f']],
    code: `${ENTRY} {
  let id = gid.x;
  if (id >= pc.count) { return; }
  let e = bendEdges[pc.first + id];
  let i0 = e.idx.x; let i1 = e.idx.y; let i2 = e.idx.z; let i3 = e.idx.w;
  let p0 = pos[i0]; let p1 = pos[i1]; let p2 = pos[i2]; let p3 = pos[i3];
  let K = e.K;
  let m = sim.phys[u32(e.extra.z)];
  let k = max(mix(m.bend.x, m.bend.y, e.extra.y), 1e-9);
  let dt = pc.dtEff;
  let alpha = e.extra.x / (k * dt * dt);
  let gamma = m.bend.w / dt;
  let C = K.x * p0.xyz + K.y * p1.xyz + K.z * p2.xyz + K.w * p3.xyz;
  let Cdot = K.x * (p0.xyz - prev[i0].xyz) + K.y * (p1.xyz - prev[i1].xyz) + K.z * (p2.xyz - prev[i2].xyz) + K.w * (p3.xyz - prev[i3].xyz);
  let W = p0.w * K.x * K.x + p1.w * K.y * K.y + p2.w * K.z * K.z + p3.w * K.w * K.w;
  let dl = -(C + gamma * Cdot) / ((1.0 + gamma) * W + alpha + 1e-12);
  pos[i0] = vec4f(p0.xyz + p0.w * K.x * dl, p0.w);
  pos[i1] = vec4f(p1.xyz + p1.w * K.y * dl, p1.w);
  pos[i2] = vec4f(p2.xyz + p2.w * K.z * dl, p2.w);
  pos[i3] = vec4f(p3.xyz + p3.w * K.w * dl, p3.w);
}`,
  },

  // One-sided long range limits along the yarns. Only the fast part of a correction moves the previous position.
  limit: {
    buffers: [['limitEdges', 'r', 'LimitEdge'], ['pos', 'rw', 'vec4f'], ['prev', 'rw', 'vec4f']],
    code: `${ENTRY} {
  let id = gid.x;
  if (id >= pc.count) { return; }
  let e = limitEdges[pc.first + id];
  let a = pos[e.a]; let b = pos[e.b];
  let w = a.w + b.w;
  if (w <= 0.0) { return; }
  let d = b.xyz - a.xyz;
  let len2 = dot(d, d);
  let lim = e.rest * (1.0 + sim.phys[e.cloth].contact.w);
  if (len2 <= lim * lim) { return; }
  let len = sqrt(len2);
  let dir = d / len;
  let prevA = prev[e.a].xyz; let prevB = prev[e.b].xyz;
  let separation = max(dot((b.xyz - prevB) - (a.xyz - prevA), dir), 0.0);
  let correction = (len - lim) * 0.6;
  let visible = min(correction, separation + 0.15 * sim.gravityDt.w);
  let stabilised = correction - visible;
  let total = dir * (correction / w);
  let shift = dir * (stabilised / w);
  pos[e.a] = vec4f(a.xyz + a.w * total, a.w);
  pos[e.b] = vec4f(b.xyz - b.w * total, b.w);
  prev[e.a] = vec4f(prevA + a.w * shift, 0.0);
  prev[e.b] = vec4f(prevB - b.w * shift, 0.0);
}`,
  },

  // Long range attachments (Kim et al. 2012): never farther from the nearest pin than through the fabric.
  lra: {
    buffers: [['attachments', 'r', 'Lra'], ['rest', 'r', 'vec4f'], ['pos', 'rw', 'vec4f'], ['prev', 'rw', 'vec4f']],
    code: `${ENTRY} {
  let i = gid.x;
  if (i >= pc.count) { return; }
  let a = attachments[i];
  if (a.pin == 0xffffffffu) { return; }
  let p = pos[i];
  if (p.w <= 0.0) { return; }
  let d = p.xyz - pos[a.pin].xyz;
  let len2 = dot(d, d);
  let lim = a.maxDistance * (1.0 + sim.phys[u32(rest[i].z)].contact.w);
  if (len2 <= lim * lim) { return; }
  let len = sqrt(len2);
  let dir = d / len;
  let previous = prev[i].xyz;
  let separation = max(dot((p.xyz - previous) - (pos[a.pin].xyz - prev[a.pin].xyz), dir), 0.0);
  let correction = (len - lim) * 0.6;
  let stabilised = correction - min(correction, separation + 0.15 * sim.gravityDt.w);
  pos[i] = vec4f(p.xyz - dir * correction, p.w);
  prev[i] = vec4f(previous - dir * stabilised, 0.0);
}`,
  },

  // Floor and analytic signed distance fields, with Coulomb and static friction relative to moving surfaces.
  collide: {
    buffers: [['rest', 'r', 'vec4f'], ['pos', 'rw', 'vec4f'], ['prev', 'rw', 'vec4f']],
    code: `
fn colliderSdf(kind: i32, size: vec4f, p: vec3f) -> vec4f {
  if (kind == 0) {
    let len = max(length(p), 1e-6);
    return vec4f(p / len, len - size.x);
  }
  if (kind == 1) {
    var q = p;
    q.y -= clamp(q.y, -size.y, size.y);
    let len = max(length(q), 1e-6);
    return vec4f(q / len, len - size.x);
  }
  if (kind == 2) {
    let r = size.w;
    let q = abs(p) - (size.xyz - vec3f(r));
    let outside = max(q, vec3f(0.0));
    let outLen = length(outside);
    if (outLen > 1e-6) { return vec4f(outside / outLen * sign(p), outLen - r); }
    var n = vec3f(0.0);
    var d = 0.0;
    if (q.x > q.y && q.x > q.z) { n.x = sign(p.x); d = q.x; }
    else if (q.y > q.z) { n.y = sign(p.y); d = q.y; }
    else { n.z = sign(p.z); d = q.z; }
    return vec4f(n, d - r);
  }
  if (kind == 3) {
    let r = size.w;
    let radial = length(p.xz);
    let q = vec2f(radial - (size.x - r), abs(p.y) - (size.y - r));
    let radialDir = select(vec2f(1.0, 0.0), p.xz / radial, radial > 1e-6);
    let outside = max(q, vec2f(0.0));
    let outLen = length(outside);
    var n2 = vec2f(0.0);
    var d = 0.0;
    if (outLen > 1e-6) { n2 = outside / outLen; d = outLen - r; }
    else if (q.x > q.y) { n2 = vec2f(1.0, 0.0); d = q.x - r; }
    else { n2 = vec2f(0.0, 1.0); d = q.y - r; }
    return vec4f(radialDir.x * n2.x, sign(p.y) * n2.y, radialDir.y * n2.x, d);
  }
  let radial = length(p.xz);
  let radialDir = select(vec2f(1.0, 0.0), p.xz / radial, radial > 1e-6);
  let q = vec2f(radial - size.x, p.y);
  let len = max(length(q), 1e-6);
  return vec4f(radialDir.x * q.x / len, q.y / len, radialDir.y * q.x / len, len - size.y);
}

fn quatNlerp(a: vec4f, bIn: vec4f, t: f32) -> vec4f {
  var b = bIn;
  if (dot(a, b) < 0.0) { b = -b; }
  return normalize(mix(a, b, t));
}

${ENTRY} {
  let i = gid.x;
  if (i >= pc.count) { return; }
  let pw = pos[i];
  if (pw.w <= 0.0) { return; }
  let r = rest[i];
  let m = sim.phys[u32(r.z)];
  let thickness = m.contact.z;
  let mu = m.contact.x;
  let dt = sim.gravityDt.w;
  var xPrev = prev[i].xyz;
  var x = pw.xyz;
  let restingLoad = length(sim.gravityDt.xyz) * dt * dt;
  let contactMargin = 0.0007;
  let stiction = 0.15;
  {
    let gap = x.y - (sim.misc.z + thickness);
    if (gap < contactMargin) {
      let pen = max(-gap, 0.0);
      let approach = max(xPrev.y - x.y, 0.0);
      x.y += pen;
      xPrev.y += max(pen - approach, 0.0);
      var mv = x - xPrev;
      mv.y = 0.0;
      let len = length(mv);
      let friction = sim.misc.w * mu;
      let stuck = len < friction * stiction * dt;
      let lim = friction * select(restingLoad, pen, pen > 0.0);
      if (len > 1e-10) { x -= select(mv * min(lim / len, 1.0), mv, stuck); }
    }
  }
  for (var c = 0u; c < sim.counts.z; c++) {
    let col = sim.colliders[c];
    let center = mix(col.posA.xyz, col.posB.xyz, pc.t);
    let rot = quatNlerp(col.rotA, col.rotB, pc.t);
    let local = rotateByQuat(quatConjugate(rot), x - center);
    let hit = colliderSdf(i32(col.posA.w), col.size, local);
    let gap = hit.w - thickness - m.air.w;
    if (gap >= contactMargin) { continue; }
    let pen = max(-gap, 0.0);
    let n = rotateByQuat(rot, hit.xyz);
    let surfaceVel = (col.posB.xyz - col.posA.xyz) / max(pc.dtEff, 1e-6) + cross(col.angVel.xyz, x - center);
    var mv = (x - xPrev) - surfaceVel * dt;
    let approach = max(-dot(mv, n), 0.0);
    x += n * pen;
    xPrev += n * max(pen - approach, 0.0);
    mv = (x - xPrev) - surfaceVel * dt;
    mv -= n * dot(mv, n);
    let len = length(mv);
    let friction = mu * col.posB.w;
    let pressed = pen > 0.0 || n.y > 0.2;
    let stuck = pressed && len < friction * stiction * dt;
    let lim = friction * select(restingLoad, pen, pen > 0.0);
    if (len > 1e-10) { x -= select(mv * min(lim / len, 1.0), mv, stuck); }
  }
  pos[i] = vec4f(x, pw.w);
  prev[i] = vec4f(xPrev, 0.0);
}`,
  },

  // The hand: a patch translated so that the grabbed particle follows the cursor, at most 5 m/s.
  grab: {
    buffers: [['rest', 'r', 'vec4f'], ['pos', 'rw', 'vec4f']],
    code: `${ENTRY} {
  let id = gid.x;
  if (id >= pc.count) { return; }
  let i = pc.first + id;
  let grabbed = sim.grabIndex.x;
  let ownDispatch = pc.count == 1u;
  if (i == grabbed && !ownDispatch) { return; }
  let r = rest[i];
  if (r.z != sim.grabRest.z) { return; }
  let dm = r.xy - sim.grabRest.xy;
  let radius2 = sim.grabTarget.w * sim.grabTarget.w;
  let d2 = dot(dm, dm);
  if (d2 >= radius2) { return; }
  var mv = sim.grabTarget.xyz - pos[grabbed].xyz;
  let maxMove = 5.0 * sim.gravityDt.w;
  let len = length(mv);
  if (len > maxMove) { mv *= maxMove / len; }
  let f = 1.0 - d2 / radius2;
  let p = pos[i];
  pos[i] = vec4f(p.xyz + mv * sim.grabRest.w * f * f, p.w);
}`,
  },

  // Aerodynamics: pressure drag along a wide-stencil normal and skin friction, in a turbulent wind.
  aero: {
    buffers: [['pos', 'r', 'vec4f'], ['vel', 'r', 'vec4f'], ['rest', 'r', 'vec4f'], ['aeroStencil', 'r', 'vec4u'], ['accel', 'rw', 'vec4f']],
    code: `
fn hash3(p: vec3i) -> u32 {
  let v = bitcast<vec3u>(p) * vec3u(1597334673u, 3812015801u, 2798796415u);
  let h = (v.x ^ v.y ^ v.z) * 1597334677u;
  return h ^ (h >> 16u);
}
fn valueNoise(p: vec3f) -> f32 {
  let fl = floor(p);
  let f = p - fl;
  let u = f * f * (3.0 - 2.0 * f);
  let c = vec3i(fl);
  let s = 1.0 / 4294967295.0;
  let n000 = f32(hash3(c)) * s; let n100 = f32(hash3(c + vec3i(1, 0, 0))) * s;
  let n010 = f32(hash3(c + vec3i(0, 1, 0))) * s; let n110 = f32(hash3(c + vec3i(1, 1, 0))) * s;
  let n001 = f32(hash3(c + vec3i(0, 0, 1))) * s; let n101 = f32(hash3(c + vec3i(1, 0, 1))) * s;
  let n011 = f32(hash3(c + vec3i(0, 1, 1))) * s; let n111 = f32(hash3(c + vec3i(1, 1, 1))) * s;
  return mix(mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y), mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y), u.z);
}
fn turbulence(p: vec3f) -> vec3f {
  let a = vec3f(valueNoise(p), valueNoise(p + vec3f(31.7, 11.3, 47.9)), valueNoise(p + vec3f(5.3, 73.1, 19.7)));
  let q = p * 2.3 + vec3f(13.1);
  let b = vec3f(valueNoise(q), valueNoise(q + vec3f(31.7, 11.3, 47.9)), valueNoise(q + vec3f(5.3, 73.1, 19.7)));
  return (a * 2.0 - 1.0) + 0.5 * (b * 2.0 - 1.0);
}
${ENTRY} {
  let i = gid.x;
  if (i >= pc.count) { return; }
  let meanWind = sim.wind.xyz;
  let gust = sim.wind.w;
  let time = sim.windParams.x; let freq = sim.windParams.y; let rho = sim.windParams.z;
  let x = pos[i].xyz;
  var flow = meanWind;
  if (gust > 0.0) { flow += gust * turbulence((x - meanWind * time) * freq + vec3f(0.0, 0.0, time * 0.7)); }
  let st = aeroStencil[i];
  let wide = cross(pos[st.x].xyz - pos[st.y].xyz, pos[st.z].xyz - pos[st.w].xyz);
  let wideLen = length(wide);
  if (wideLen < 1e-12) { accel[i] = vec4f(0.0); return; }
  let n = wide / wideLen;
  let meanVel = (vel[i].xyz + vel[st.x].xyz + vel[st.y].xyz + vel[st.z].xyz + vel[st.w].xyz) * 0.2;
  let relative = flow - meanVel;
  let vn = dot(relative, n);
  let vt = relative - vn * n;
  let vtLen = length(vt);
  let m = sim.phys[u32(rest[i].z)];
  var a = (0.5 * rho / max(m.air.x, 1e-4)) * (m.air.y * vn * abs(vn) * n + m.air.z * vtLen * vt);
  let aLen = length(a) * sim.gravityDt.w;
  let lim = length(relative);
  if (aLen > lim && aLen > 1e-9) { a *= lim / aLen; }
  accel[i] = vec4f(a, 0.0);
}`,
  },

  // ---- cloth-cloth collision: spatial hash of triangles, candidate lists, contacts

  resetArgs: {
    buffers: [['rebuildArgs', 'rw', 'u32']],
    code: `${ENTRY} {
  if (gid.x != 0u) { return; }
  rebuildArgs[0] = 0u; rebuildArgs[3] = 0u; rebuildArgs[6] = 0u;
}`,
  },

  // Decides whether the candidate lists need rebuilding: they stay valid until some particle used up the margin.
  skin: {
    buffers: [['pos', 'r', 'vec4f'], ['rebuildPos', 'r', 'vec4f'], ['rebuildArgs', 'rw', 'u32']],
    code: `${ENTRY} {
  let i = gid.x;
  if (i >= pc.count) { return; }
  let d = pos[i].xyz - rebuildPos[i].xyz;
  if (dot(d, d) > sim.selfRebuild.w) {
    rebuildArgs[0] = u32(sim.selfRebuild.x);
    rebuildArgs[3] = u32(sim.selfRebuild.y);
    rebuildArgs[6] = u32(sim.selfRebuild.z);
  }
}`,
  },

  hashClear: {
    buffers: [['cellCounts', 'rw', 'atomic<u32>']],
    code: `${ENTRY} {
  let c = gid.x;
  if (c >= pc.count) { return; }
  atomicStore(&cellCounts[c], 0u);
}`,
  },

  hash: {
    buffers: [['stretchTris', 'r', 'StretchTri'], ['pos', 'r', 'vec4f'], ['triCenters', 'rw', 'vec4f'], ['cellCounts', 'rw', 'atomic<u32>'], ['cellEntries', 'rw', 'u32']],
    code: `${ENTRY} {
  let t = gid.x;
  if (t >= pc.count) { return; }
  let idx = stretchTris[t].idx;
  let center = (pos[idx.x].xyz + pos[idx.y].xyz + pos[idx.z].xyz) / 3.0;
  triCenters[t] = vec4f(center, 0.0);
  let h = cellHash(cellOf(center));
  let slot = atomicAdd(&cellCounts[h], 1u);
  if (slot < CELL_CAPACITY) { cellEntries[h * CELL_CAPACITY + slot] = t; }
}`,
  },

  // Candidate triangles of each particle. Triangles close through the fabric are skipped. Each candidate keeps the
  // side of the triangle the particle belongs to across rebuilds, so that a particle that went through is brought back.
  neighbors: {
    buffers: [['pos', 'r', 'vec4f'], ['rest', 'r', 'vec4f'], ['stretchTris', 'r', 'StretchTri'], ['triCenters', 'r', 'vec4f'],
      ['cellCounts', 'r', 'u32'], ['cellEntries', 'r', 'u32'], ['neighbors', 'rw', 'u32']],
    code: `
fn distanceToTriangle(p: vec3f, a: vec3f, b: vec3f, c: vec3f) -> f32 {
  let ab = b - a; let ac = c - a; let ap = p - a;
  let d1 = dot(ab, ap); let d2 = dot(ac, ap);
  if (d1 <= 0.0 && d2 <= 0.0) { return length(ap); }
  let bp = p - b;
  let d3 = dot(ab, bp); let d4 = dot(ac, bp);
  if (d3 >= 0.0 && d4 <= d3) { return length(bp); }
  let vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) { return length(ap - ab * (d1 / max(d1 - d3, 1e-20))); }
  let cp = p - c;
  let d5 = dot(ab, cp); let d6 = dot(ac, cp);
  if (d6 >= 0.0 && d5 <= d6) { return length(cp); }
  let vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) { return length(ap - ac * (d2 / max(d2 - d6, 1e-20))); }
  let va = d3 * d6 - d5 * d4;
  if (va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0) { return length(bp - (c - b) * ((d4 - d3) / max((d4 - d3) + (d5 - d6), 1e-20))); }
  let denom = 1.0 / max(va + vb + vc, 1e-20);
  return length(ap - ab * (vb * denom) - ac * (vc * denom));
}
${ENTRY} {
  let i = gid.x;
  if (i >= pc.count) { return; }
  let x = pos[i].xyz;
  let ri = rest[i];
  let home = cellOf(x);
  let search2 = sim.selfp.z * sim.selfp.z;
  let exclude2 = sim.selfp.y * sim.selfp.y;
  let period = sim.phys[u32(ri.z)].extra.x;
  let base = i * NEIGH_STRIDE;
  var previous: array<u32, 32>;
  let previousCount = min(neighbors[base], MAX_NEIGHBORS);
  for (var k = 0u; k < previousCount; k++) { previous[k] = neighbors[base + 1u + k]; }
  var found = 0u;
  var chosen: array<u32, 32>;
  var distances: array<f32, 32>;
  var farthest = -1.0;
  var farthestSlot = 0u;
  for (var dz = -1; dz <= 1; dz++) {
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        let cell = home + vec3i(dx, dy, dz);
        let h = cellHash(cell);
        let n = min(cellCounts[h], CELL_CAPACITY);
        for (var s = 0u; s < n; s++) {
          let t = cellEntries[h * CELL_CAPACITY + s];
          let center = triCenters[t].xyz;
          let off = center - x;
          let d2 = dot(off, off);
          if (d2 > search2) { continue; }
          if (any(cellOf(center) != cell)) { continue; }
          let idx = stretchTris[t].idx;
          if (idx.x == i || idx.y == i || idx.z == i) { continue; }
          if (f32(idx.w) == ri.z) {
            var dm = rest[idx.x].xy - ri.xy;
            if (period > 0.0) { dm.x -= period * round(dm.x / period); }
            if (dot(dm, dm) < exclude2) { continue; }
          }
          if (distanceToTriangle(x, pos[idx.x].xyz, pos[idx.y].xyz, pos[idx.z].xyz) > sim.selfDepth.y) { continue; }
          if (found < MAX_NEIGHBORS) {
            chosen[found] = t;
            distances[found] = d2;
            if (d2 > farthest) { farthest = d2; farthestSlot = found; }
            found++;
          } else if (d2 < farthest) {
            chosen[farthestSlot] = t;
            distances[farthestSlot] = d2;
            farthest = -1.0;
            for (var k = 0u; k < MAX_NEIGHBORS; k++) {
              if (distances[k] > farthest) { farthest = distances[k]; farthestSlot = k; }
            }
          }
        }
      }
    }
  }
  let maxDepth = sim.selfDepth.x;
  for (var k = 0u; k < found; k++) {
    let t = chosen[k];
    let idx = stretchTris[t].idx;
    let a = pos[idx.x].xyz;
    let cr = cross(pos[idx.y].xyz - a, pos[idx.z].xyz - a);
    let crLen = length(cr);
    let planeDist = select(0.0, dot(x - a, cr) / crLen, crLen > 1e-10);
    var sideBit = select(0u, SIDE_BIT, planeDist < 0.0);
    if (abs(planeDist) < maxDepth) {
      for (var p = 0u; p < previousCount; p++) {
        if ((previous[p] & INDEX_MASK) == t) { sideBit = previous[p] & SIDE_BIT; break; }
      }
    }
    neighbors[base + 1u + k] = t | sideBit;
  }
  neighbors[base] = found;
}`,
  },

  mark: {
    buffers: [['pos', 'r', 'vec4f'], ['rebuildPos', 'rw', 'vec4f']],
    code: `${ENTRY} {
  let i = gid.x;
  if (i >= pc.count) { return; }
  rebuildPos[i] = vec4f(pos[i].xyz, 0.0);
}`,
  },

  // Vertex against triangle, the correction shared with the triangle by barycentric weights (momentum conserved),
  // summed with fixed point atomics and scaled by the largest Jacobi load among the four particles.
  self: {
    buffers: [['neighbors', 'r', 'u32'], ['stretchTris', 'r', 'StretchTri'], ['pos', 'r', 'vec4f'], ['prev', 'r', 'vec4f'], ['rest', 'r', 'vec4f'],
      ['contactCounts', 'r', 'u32'], ['contactAccum', 'rw', 'atomic<i32>']],
    code: `
const kFixedPoint = 1.0e8;
const kLoadUnits = 1024.0;
const kStiction = 0.15;
const kMaxStabilisation = 1.0;

fn accumulate(v: u32, position: vec3f, previous: vec3f, load: f32) {
  let b = v * 8u;
  atomicAdd(&contactAccum[b + 0u], i32(position.x * kFixedPoint));
  atomicAdd(&contactAccum[b + 1u], i32(position.y * kFixedPoint));
  atomicAdd(&contactAccum[b + 2u], i32(position.z * kFixedPoint));
  atomicAdd(&contactAccum[b + 3u], i32(load * kLoadUnits));
  atomicAdd(&contactAccum[b + 4u], i32(previous.x * kFixedPoint));
  atomicAdd(&contactAccum[b + 5u], i32(previous.y * kFixedPoint));
  atomicAdd(&contactAccum[b + 6u], i32(previous.z * kFixedPoint));
}

fn closestOnSegment(x: vec3f, p: vec3f, q: vec3f) -> f32 {
  let e = q - p;
  return clamp(dot(x - p, e) / max(dot(e, e), 1e-14), 0.0, 1.0);
}

${ENTRY} {
  let i = gid.x;
  if (i >= pc.count) { return; }
  let base = i * NEIGH_STRIDE;
  let n = min(neighbors[base], MAX_NEIGHBORS);
  if (n == 0u) { return; }
  let pi = pos[i];
  let x = pi.xyz;
  let xp = prev[i].xyz;
  let wi = pi.w;
  let h = sim.selfp.x;
  let dt = sim.gravityDt.w;
  let muI = sim.phys[u32(rest[i].z)].contact.y;
  let countI = contactCounts[i];
  let maxDepth = sim.selfDepth.x;
  for (var k = 0u; k < n; k++) {
    let candidate = neighbors[base + 1u + k];
    let tri = stretchTris[candidate & INDEX_MASK];
    let idx = tri.idx;
    let pa = pos[idx.x]; let pb = pos[idx.y]; let pcc = pos[idx.z];
    let a = pa.xyz; let b = pb.xyz; let c = pcc.xyz;
    let e1 = b - a; let e2 = c - a;
    let cr = cross(e1, e2);
    let crLen2 = dot(cr, cr);
    if (crLen2 < 1e-16) { continue; }
    let toX = x - a;
    let planeDist = dot(toX, cr) / sqrt(crLen2);
    let side = select(1.0, -1.0, (candidate & SIDE_BIT) != 0u);
    let wrongSide = side * planeDist < 0.0;
    if (abs(planeDist) >= select(h, maxDepth, wrongSide)) { continue; }
    let ap = prev[idx.x].xyz; let bp = prev[idx.y].xyz; let cp = prev[idx.z].xyz;
    let baryB = dot(cross(toX, e2), cr) / crLen2;
    let baryC = dot(cross(e1, toX), cr) / crLen2;
    var bary = vec3f(1.0 - baryB - baryC, baryB, baryC);
    var normal = vec3f(0.0);
    var C = 0.0;
    var crossed = false;
    if (all(bary >= vec3f(-0.01))) {
      normal = (cr / sqrt(crLen2)) * side;
      C = h - side * planeDist;
      crossed = wrongSide;
      if (C <= 0.0) { continue; }
      bary = clamp(bary, vec3f(0.0), vec3f(1.0));
      bary /= bary.x + bary.y + bary.z;
    } else {
      if (wrongSide) { continue; }
      let t0 = closestOnSegment(x, a, b); let t1 = closestOnSegment(x, b, c); let t2 = closestOnSegment(x, c, a);
      let q0 = mix(a, b, t0); let q1 = mix(b, c, t1); let q2 = mix(c, a, t2);
      let d0 = dot(x - q0, x - q0); let d1 = dot(x - q1, x - q1); let d2 = dot(x - q2, x - q2);
      var q = q0;
      if (d0 <= d1 && d0 <= d2) { q = q0; bary = vec3f(1.0 - t0, t0, 0.0); }
      else if (d1 <= d2) { q = q1; bary = vec3f(0.0, 1.0 - t1, t1); }
      else { q = q2; bary = vec3f(t2, 0.0, 1.0 - t2); }
      let dist2 = min(d0, min(d1, d2));
      if (dist2 >= h * h || dist2 < 1e-12) { continue; }
      let dist = sqrt(dist2);
      normal = (x - q) / dist;
      C = h - dist;
    }
    let wa = pa.w; let wb = pb.w; let wc = pcc.w;
    let W = wi + wa * bary.x * bary.x + wb * bary.y * bary.y + wc * bary.z * bary.z;
    if (W <= 0.0) { continue; }
    let rel = (x - xp) - (bary.x * (a - ap) + bary.y * (b - bp) + bary.z * (c - cp));
    let relN = dot(rel, normal);
    let cancelled = min(C, max(-relN, 0.0));
    let stabilised = select(min(C - cancelled, kMaxStabilisation * dt), C - cancelled, crossed);
    let total = cancelled + stabilised;
    let tangent = rel - normal * relN;
    let tangentLen = length(tangent);
    let mu = 0.5 * (muI + sim.phys[idx.w].contact.y);
    var impulse = normal * total;
    if (tangentLen > 1e-10) {
      let stuck = tangentLen < mu * kStiction * dt;
      impulse -= tangent * select(min(mu * total / tangentLen, 1.0), 1.0, stuck);
    }
    let largest = max(max(countI, contactCounts[idx.x]), max(contactCounts[idx.y], contactCounts[idx.z]));
    let scale = sim.misc.y / max(f32(largest) / kLoadUnits, 1.0) / W;
    let J = impulse * scale;
    let Jp = normal * stabilised * scale;
    accumulate(i, wi * J, wi * Jp, wi / W);
    accumulate(idx.x, -wa * bary.x * J, -wa * bary.x * Jp, wa * bary.x * bary.x / W);
    accumulate(idx.y, -wb * bary.y * J, -wb * bary.y * Jp, wb * bary.y * bary.y / W);
    accumulate(idx.z, -wc * bary.z * J, -wc * bary.z * Jp, wc * bary.z * bary.z / W);
  }
}`,
  },

  selfApply: {
    buffers: [['contactAccum', 'rw', 'atomic<i32>'], ['contactCounts', 'rw', 'u32'], ['pos', 'rw', 'vec4f'], ['prev', 'rw', 'vec4f']],
    code: `${ENTRY} {
  let i = gid.x;
  if (i >= pc.count) { return; }
  let b = i * 8u;
  let load = atomicLoad(&contactAccum[b + 3u]);
  contactCounts[i] = u32(max(load, 0));
  let ax = atomicLoad(&contactAccum[b + 0u]); let ay = atomicLoad(&contactAccum[b + 1u]); let az = atomicLoad(&contactAccum[b + 2u]);
  if (load == 0 && ax == 0 && ay == 0 && az == 0) { return; }
  let toMetres = 1.0e-8;
  var mv = vec3f(f32(ax), f32(ay), f32(az)) * toMetres;
  var shift = vec3f(f32(atomicLoad(&contactAccum[b + 4u])), f32(atomicLoad(&contactAccum[b + 5u])), f32(atomicLoad(&contactAccum[b + 6u]))) * toMetres;
  let len = length(mv);
  if (len > sim.selfp.x) {
    let k = sim.selfp.x / len;
    mv *= k;
    shift *= k;
  }
  let p = pos[i];
  pos[i] = vec4f(p.xyz + mv, p.w);
  prev[i] = vec4f(prev[i].xyz + shift, 0.0);
  for (var k = 0u; k < 8u; k++) { atomicStore(&contactAccum[b + k], 0); }
}`,
  },

  // Shading frame: area weighted normal, and the warp tangent measured from the deformation itself.
  normals: {
    buffers: [['vtxOffsets', 'r', 'u32'], ['vtxTris', 'r', 'u32'], ['stretchTris', 'r', 'StretchTri'], ['pos', 'r', 'vec4f'], ['rest', 'r', 'vec4f'],
      ['deltas', 'rw', 'vec4f'], ['tangents', 'rw', 'vec4f']],
    code: `${ENTRY} {
  let i = gid.x;
  if (i >= pc.count) { return; }
  var n = vec3f(0.0);
  var warp = vec3f(0.0);
  for (var k = vtxOffsets[i]; k < vtxOffsets[i + 1u]; k++) {
    let t = stretchTris[vtxTris[k]];
    let x0 = pos[t.idx.x].xyz;
    let e1 = pos[t.idx.y].xyz - x0;
    let e2 = pos[t.idx.z].xyz - x0;
    n += cross(e1, e2);
    warp += t.extra.x * (t.grad.z * e1 + t.grad.w * e2);
  }
  let nLen = length(n);
  n = select(vec3f(0.0, 1.0, 0.0), n / nLen, nLen > 1e-14);
  warp -= n * dot(warp, n);
  let tLen = length(warp);
  warp = select(vec3f(0.0, 0.0, -1.0), warp / tLen, tLen > 1e-14);
  deltas[i] = vec4f(n, 0.0);
  tangents[i] = vec4f(warp, 0.0);
}`,
  },

  normalsSmooth: {
    buffers: [['vtxOffsets', 'r', 'u32'], ['vtxTris', 'r', 'u32'], ['stretchTris', 'r', 'StretchTri'], ['deltas', 'r', 'vec4f'], ['normals', 'rw', 'vec4f']],
    code: `${ENTRY} {
  let i = gid.x;
  if (i >= pc.count) { return; }
  var n = vec3f(0.0);
  for (var k = vtxOffsets[i]; k < vtxOffsets[i + 1u]; k++) {
    let t = stretchTris[vtxTris[k]];
    n += deltas[t.idx.x].xyz + deltas[t.idx.y].xyz + deltas[t.idx.z].xyz;
  }
  let len = length(n);
  normals[i] = vec4f(select(deltas[i].xyz, n / len, len > 1e-8), 0.0);
}`,
  },
};

// Full WGSL of one shader: prelude, generated storage declarations, body.
export function shaderSource(name) {
  const spec = SHADERS[name];
  const decls = spec.buffers
    .map(([buf, access, type], i) => `@group(1) @binding(${i}) var<storage, ${access === 'rw' ? 'read_write' : 'read'}> ${buf}: array<${type}>;`)
    .join('\n');
  return `${PRELUDE}\n${decls}\n${spec.code}`;
}
