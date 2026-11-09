// GPU cloth solver: buffers, compute pipelines and the recording of one frame of simulation.
import { SHADERS, shaderSource } from './simShaders.js';
import { buildClothData } from './mesh.js';
import { colliderPose } from './scenes.js';
import { quat } from './math.js';

const WG = 64;
const NEIGH_STRIDE = 33;
const CELL_CAPACITY = 48;
const PARAM_STRIDE = 256;
const MAX_PARAM_SLOTS = 4096;
const FRAME_WORDS = 1084;  // see struct Frame in simShaders.js

export const DEFAULT_SETTINGS = {
  substeps: 24, aeroEvery: 4, selfRebuildEvery: 6, selfCollision: true, selfScale: 1.0, selfDepth: 0.4, selfRelax: 0.9,
  damping: 0.25, maxSpeed: 22, viscosity: 0.06, gravity: -9.81, floorFriction: 1.0, airDensity: 1.2,
  wind: [0, 0, 0], turbulence: 0, windFrequency: 0.8, windScale: 1.0,
};

const groups = (n) => Math.ceil(n / WG);

export class ClothSim {
  constructor(device) {
    this.device = device;
    this.frameUbo = device.createBuffer({ size: FRAME_WORDS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.paramsUbo = device.createBuffer({ size: MAX_PARAM_SLOTS * PARAM_STRIDE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.paramsData = new ArrayBuffer(MAX_PARAM_SLOTS * PARAM_STRIDE);
    this.paramsU32 = new Uint32Array(this.paramsData);
    this.paramsF32 = new Float32Array(this.paramsData);
    this.frameData = new ArrayBuffer(FRAME_WORDS * 4);
    this.frameF32 = new Float32Array(this.frameData);
    this.frameU32 = new Uint32Array(this.frameData);

    this.group0Layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 16 } },
      ],
    });
    this.group0 = device.createBindGroup({
      layout: this.group0Layout,
      entries: [{ binding: 0, resource: { buffer: this.frameUbo } }, { binding: 1, resource: { buffer: this.paramsUbo, size: 16 } }],
    });

    this.pipelines = {};
    for (const name of Object.keys(SHADERS)) {
      const spec = SHADERS[name];
      const layout1 = device.createBindGroupLayout({
        entries: spec.buffers.map(([, access], i) => ({ binding: i, visibility: GPUShaderStage.COMPUTE, buffer: { type: access === 'rw' ? 'storage' : 'read-only-storage' } })),
      });
      const module = device.createShaderModule({ label: name, code: shaderSource(name) });
      const pipeline = device.createComputePipeline({
        label: name,
        layout: device.createPipelineLayout({ bindGroupLayouts: [this.group0Layout, layout1] }),
        compute: { module, entryPoint: 'main' },
      });
      this.pipelines[name] = { pipeline, layout1, spec, module };
    }
    this.buffers = null;
    this.substepCounter = 0;
  }

  async checkShaders() {
    const errors = [];
    for (const [name, p] of Object.entries(this.pipelines)) {
      const info = await p.module.getCompilationInfo();
      for (const m of info.messages) if (m.type === 'error') errors.push(`${name}:${m.lineNum}:${m.linePos} ${m.message}`);
    }
    return errors;
  }

  destroyBuffers() {
    if (!this.buffers) return;
    for (const b of Object.values(this.buffers)) b.destroy();
    this.buffers = null;
  }

  build(scene) {
    this.destroyBuffers();
    const d = buildClothData(scene);
    this.data = d;
    const device = this.device;
    const n = d.positions.length;
    const S = GPUBufferUsage.STORAGE;
    const make = (array, usage = 0) => {
      const bytes = Math.max(16, Math.ceil(array.byteLength / 4) * 4);
      const buf = device.createBuffer({ size: bytes, usage: S | GPUBufferUsage.COPY_DST | usage, mappedAtCreation: true });
      new Uint8Array(buf.getMappedRange()).set(new Uint8Array(array.buffer, array.byteOffset, array.byteLength));
      buf.unmap();
      return buf;
    };
    const zeros = (bytes, usage = 0) => device.createBuffer({ size: Math.max(16, bytes), usage: S | GPUBufferUsage.COPY_DST | usage });
    const vec4s = (rows) => {
      const out = new Float32Array(rows.length * 4);
      rows.forEach((r, i) => out.set(r, i * 4));
      return out;
    };
    const uvec4s = (rows) => {
      const out = new Uint32Array(rows.length * 4);
      rows.forEach((r, i) => out.set(r, i * 4));
      return out;
    };

    const positions = vec4s(d.positions);
    this.initialPositions = positions;
    const stretch = new ArrayBuffer(Math.max(1, d.stretch.length) * 48);
    {
      const u = new Uint32Array(stretch), f = new Float32Array(stretch);
      d.stretch.forEach((s, i) => {
        u.set(s.idx, i * 12);
        f.set(s.grad, i * 12 + 4);
        f[i * 12 + 8] = s.area;
      });
    }
    const bend = new ArrayBuffer(Math.max(1, d.bend.length) * 48);
    {
      const u = new Uint32Array(bend), f = new Float32Array(bend);
      d.bend.forEach((b, i) => {
        u.set(b.idx, i * 12);
        f.set(b.K, i * 12 + 4);
        f.set(b.extra, i * 12 + 8);
      });
    }
    const limits = new ArrayBuffer(Math.max(1, d.limits.length) * 16);
    {
      const u = new Uint32Array(limits), f = new Float32Array(limits);
      d.limits.forEach((l, i) => {
        u[i * 4] = l[0]; u[i * 4 + 1] = l[1]; f[i * 4 + 2] = l[2]; u[i * 4 + 3] = l[3];
      });
    }
    const attachments = new ArrayBuffer(n * 8);
    {
      const u = new Uint32Array(attachments), f = new Float32Array(attachments);
      d.attachments.forEach((a, i) => { u[i * 2] = a[0] >>> 0; f[i * 2 + 1] = a[1]; });
    }
    let cells = 1024;
    while (cells < d.stretch.length * 2) cells *= 2;
    this.hashMask = cells - 1;
    this.cells = cells;

    const renderVerts = new ArrayBuffer(d.renderVertices.length * 12);
    {
      const u = new Uint32Array(renderVerts), f = new Float32Array(renderVerts);
      d.renderVertices.forEach((r, i) => { f[i * 3] = r[0]; f[i * 3 + 1] = r[1]; u[i * 3 + 2] = r[2]; });
    }

    this.buffers = {
      pos: make(positions, GPUBufferUsage.COPY_SRC),
      prev: make(positions),
      vel: zeros(n * 16),
      accel: zeros(n * 16),
      rest: make(vec4s(d.rest)),
      initPos: make(vec4s(d.init)),
      stretchTris: make(new Uint8Array(stretch)),
      bendEdges: make(new Uint8Array(bend)),
      limitEdges: make(new Uint8Array(limits)),
      attachments: make(new Uint8Array(attachments)),
      aeroStencil: make(uvec4s(d.aeroStencil)),
      gridNeighbors: make(uvec4s(d.gridNeighbors)),
      vtxOffsets: make(d.vertexTriOffsets),
      vtxTris: make(d.vertexTris),
      cellCounts: zeros(cells * 4),
      cellEntries: zeros(cells * CELL_CAPACITY * 4),
      neighbors: zeros(n * NEIGH_STRIDE * 4),
      deltas: zeros(n * 16),
      normals: zeros(n * 16),
      tangents: zeros(n * 16),
      contactAccum: zeros(n * 32),
      contactCounts: zeros(n * 4),
      triCenters: zeros(d.stretch.length * 16),
      rebuildPos: zeros(n * 16),  // zero: the first check triggers a rebuild
      rebuildArgs: make(new Uint32Array([0, 1, 1, 0, 1, 1, 0, 1, 1]), GPUBufferUsage.INDIRECT),
      renderVertices: device.createBuffer({ size: renderVerts.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST }),
      indices: device.createBuffer({ size: Math.ceil(d.indices.length * 4 / 4) * 4, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST }),
      readback: device.createBuffer({ size: n * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
    };
    device.queue.writeBuffer(this.buffers.renderVertices, 0, renderVerts);
    device.queue.writeBuffer(this.buffers.indices, 0, new Uint32Array(d.indices));

    this.bindGroups = {};
    for (const [name, p] of Object.entries(this.pipelines))
      this.bindGroups[name] = device.createBindGroup({
        layout: p.layout1,
        entries: p.spec.buffers.map(([buf], i) => ({ binding: i, resource: { buffer: this.buffers[buf] } })),
      });
    this.substepCounter = 0;
    this.normalsReady = false;
  }

  reset() {
    const q = this.device.queue;
    q.writeBuffer(this.buffers.pos, 0, this.initialPositions);
    q.writeBuffer(this.buffers.prev, 0, this.initialPositions);
    const zero = (buf) => q.writeBuffer(buf, 0, new Uint8Array(buf.size));
    for (const name of ['vel', 'accel', 'cellCounts', 'neighbors', 'contactAccum', 'contactCounts', 'rebuildPos']) zero(this.buffers[name]);
    this.substepCounter = 0;
  }

  get particleCount() { return this.data.positions.length; }
  get triangleCount() { return this.data.stretch.length; }

  // Fills the frame uniform for a step from `time` to `time + dt` made of `substeps` substeps.
  prepareFrame(scene, fabrics, settings, grab, time, dt, substeps) {
    const f = this.frameF32, u = this.frameU32;
    f.fill(0);
    const d = this.data;
    const h = dt / Math.max(1, substeps);
    f.set([0, settings.gravity, 0, h], 0);
    f.set([scene.wind[0] * settings.windScale, scene.wind[1] * settings.windScale, scene.wind[2] * settings.windScale, scene.turbulence * settings.windScale], 4);
    f.set([time, scene.windFrequency, settings.airDensity, settings.maxSpeed], 8);

    const clothThickness = Math.max(0.0075, d.maxSpacing * 0.9) * scene.selfScale * settings.selfScale;
    const search = d.maxSpacing * 1.5 + clothThickness;
    const exclusion = Math.max(d.maxSpacing * 2.2, clothThickness * 2.2);
    const rebuildMotion = d.maxSpacing * 0.375;
    f.set([groups(this.cells), groups(d.stretch.length), groups(d.positions.length), rebuildMotion * rebuildMotion], 12);
    f.set([d.maxSpacing * settings.selfDepth, clothThickness + 3 * rebuildMotion, 0, 0], 16);
    f.set([clothThickness, exclusion, search, search], 20);
    f.set([grab.target[0], grab.target[1], grab.target[2], grab.active ? grab.radius : 0], 24);
    f.set([grab.rest[0], grab.rest[1], grab.cloth, grab.stiffness], 28);
    f.set([settings.damping, settings.selfRelax, 0, settings.floorFriction], 32);
    u.set([d.positions.length, this.hashMask, Math.min(scene.colliders.length, 16), Math.round(settings.viscosity * 10000)], 36);
    u.set([grab.particle, 0, 0, 0], 40);

    for (let i = 0; i < d.ranges.length && i < 20; ++i) {
      const fab = fabrics[i];
      const thickness = Math.max(0.0025, d.ranges[i].spacing * 0.4);
      const base = 44 + i * 20;
      f.set([1 / fab.stretchWarp, 1 / fab.stretchWeft, 1 / fab.shear, fab.compressionSoftening], base);
      f.set([fab.bendWarp, fab.bendWeft, fab.dampStretch, fab.dampBend], base + 4);
      f.set([fab.friction, fab.selfFriction, thickness, fab.strainLimit], base + 8);
      f.set([fab.density, fab.dragNormal, fab.dragTangent, scene.cloths[i].layer * clothThickness * 1.3], base + 12);
      f.set([scene.cloths[i].wrapU ? scene.cloths[i].width : 0, 0, 0, 0], base + 16);
    }
    scene.colliders.slice(0, 16).forEach((c, i) => {
      const a = colliderPose(c, time), b = colliderPose(c, time + dt);
      let dq = quat.multiply(b.rotation, quat.conjugate(a.rotation));
      if (dq[3] < 0) dq = dq.map((v) => -v);
      const omega = dt > 0 ? [2 * dq[0] / dt, 2 * dq[1] / dt, 2 * dq[2] / dt] : [0, 0, 0];
      const base = 444 + i * 24;
      f.set([...a.position, c.type], base);
      f.set(a.rotation, base + 4);
      f.set([...b.position, c.friction], base + 8);
      f.set(b.rotation, base + 12);
      f.set(c.size, base + 16);
      f.set([...omega, 0], base + 20);
    });
    for (let g = 0; g < 8; ++g) {
      const fn = scene.pinGroups[g] || scene.pinGroups[0];
      f.set(fn(time), 828 + g * 16);
      f.set(fn(time + dt), 956 + g * 16);
    }
    this.device.queue.writeBuffer(this.frameUbo, 0, this.frameData);
    this.grabActive = grab.active;
    this.grabParticle = grab.particle;
  }

  // Records `substeps` substeps and the shading frame in one compute pass. Writes the dispatch parameters.
  encodeStep(encoder, settings, scene, substeps, dt) {
    const d = this.data;
    const n = d.positions.length;
    const h = dt / Math.max(1, substeps);
    let slot = 0;
    const pass = encoder.beginComputePass();
    pass.setBindGroup(0, this.group0, [0]);
    let bound = null;
    const setSlot = (first, count, dtEff, t) => {
      if (slot >= MAX_PARAM_SLOTS) throw new Error('Too many dispatches in a frame');
      const w = slot * (PARAM_STRIDE / 4);
      this.paramsU32[w] = first;
      this.paramsU32[w + 1] = count;
      this.paramsF32[w + 2] = dtEff;
      this.paramsF32[w + 3] = t;
      pass.setBindGroup(0, this.group0, [slot * PARAM_STRIDE]);
      slot++;
    };
    const use = (name) => {
      if (bound === name) return;
      pass.setPipeline(this.pipelines[name].pipeline);
      pass.setBindGroup(1, this.bindGroups[name]);
      bound = name;
    };
    const dispatch = (name, first, count, dtEff, t) => {
      if (count === 0) return;
      use(name);
      setSlot(first, count, dtEff, t);
      pass.dispatchWorkgroups(groups(count));
    };
    const indirect = (name, offset, count, t) => {
      use(name);
      setSlot(0, count, h, t);
      pass.dispatchWorkgroupsIndirect(this.buffers.rebuildArgs, offset);
    };
    const selfCollision = settings.selfCollision && scene.selfCollision;
    for (let s = 0; s < substeps; ++s) {
      const t = (s + 1) / substeps;
      if (this.substepCounter % settings.aeroEvery === 0) dispatch('aero', 0, n, h, t);
      if (selfCollision && this.substepCounter % settings.selfRebuildEvery === 0) {
        dispatch('resetArgs', 0, 1, h, t);
        dispatch('skin', 0, n, h, t);
        indirect('hashClear', 0, this.cells, t);
        indirect('hash', 12, d.stretch.length, t);
        indirect('neighbors', 24, n, t);
        indirect('mark', 24, n, t);
      }
      dispatch('velocity', 0, n, h, t);
      dispatch('integrate', 0, n, h, t);
      for (const b of d.stretchBatches) dispatch('stretch', b.first, b.count, h, t);
      for (const b of d.bendBatches) dispatch('bend', b.first, b.count, h, t);
      if (d.limitPhases.length) for (const b of d.limitPhases[this.substepCounter % d.limitPhases.length]) dispatch('limit', b.first, b.count, h, t);
      if (d.hasAttachments) dispatch('lra', 0, n, h, t);
      dispatch('collide', 0, n, dt, t);
      if (this.grabActive) {
        dispatch('grab', 0, n, h, t);
        dispatch('grab', this.grabParticle, 1, h, t);
      }
      if (selfCollision) {
        dispatch('self', 0, n, h, t);
        dispatch('selfApply', 0, n, h, t);
      }
      this.substepCounter++;
    }
    dispatch('normals', 0, n, h, 1);
    dispatch('normalsSmooth', 0, n, h, 1);
    pass.end();
    this.device.queue.writeBuffer(this.paramsUbo, 0, this.paramsData, 0, slot * PARAM_STRIDE);
    this.normalsReady = true;
    return slot;
  }

  // Only the shading frame, when the simulation is paused or has not run yet.
  encodeNormals(encoder) {
    const n = this.data.positions.length;
    const pass = encoder.beginComputePass();
    const w = 0;
    this.paramsU32[w] = 0;
    this.paramsU32[w + 1] = n;
    this.paramsF32[w + 2] = 0;
    this.paramsF32[w + 3] = 1;
    pass.setBindGroup(0, this.group0, [0]);
    for (const name of ['normals', 'normalsSmooth']) {
      pass.setPipeline(this.pipelines[name].pipeline);
      pass.setBindGroup(1, this.bindGroups[name]);
      pass.dispatchWorkgroups(groups(n));
    }
    pass.end();
    this.device.queue.writeBuffer(this.paramsUbo, 0, this.paramsData, 0, 16);
  }

  // Positions as they are now, for picking with the mouse.
  async readPositions() {
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.buffers.pos, 0, this.buffers.readback, 0, this.buffers.readback.size);
    this.device.queue.submit([encoder.finish()]);
    await this.buffers.readback.mapAsync(GPUMapMode.READ);
    const copy = new Float32Array(this.buffers.readback.getMappedRange().slice(0));
    this.buffers.readback.unmap();
    return copy;
  }
}
