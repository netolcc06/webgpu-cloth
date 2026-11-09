// Renderer: soft shadow map from the key light, then a forward MSAA pass with sky, floor, colliders and cloth.
import { COMMON, CLOTH, MESH } from './renderShaders.js';
import { COLLIDER, colliderPose } from './scenes.js';
import { identity, invert, lookAt, multiply, ortho, perspective, quat, translation, vec3 } from './math.js';

const SHADOW_SIZE = 2048;
const DRAW_STRIDE = 256;
const MAX_DRAWS = 256;
const VIEW_FLOATS = 92;

// ---------------------------------------------------------------- primitive meshes (position + normal)

function meshBuilder() {
  const v = [], idx = [];
  return {
    v, idx,
    vertex(p, n) { v.push(p[0], p[1], p[2], n[0], n[1], n[2]); return v.length / 6 - 1; },
  };
}

function sphereLike(r, halfHeight, segments = 48, rings = 24) {
  const b = meshBuilder();
  const rows = rings + 1 + (halfHeight > 0 ? 1 : 0);
  const ringPos = [];
  for (let k = 0; k <= rings; ++k) {
    const theta = (k / rings) * Math.PI;
    ringPos.push([Math.cos(theta), Math.sin(theta), k <= rings / 2 ? halfHeight : -halfHeight]);
    if (halfHeight > 0 && k === rings / 2) ringPos.push([Math.cos(theta), Math.sin(theta), -halfHeight]);
  }
  for (const [cy, sy, off] of ringPos)
    for (let s = 0; s <= segments; ++s) {
      const phi = (s / segments) * Math.PI * 2;
      const n = [sy * Math.cos(phi), cy, sy * Math.sin(phi)];
      b.vertex([n[0] * r, n[1] * r + off, n[2] * r], n);
    }
  for (let k = 0; k < rows - 1; ++k)
    for (let s = 0; s < segments; ++s) {
      const a = k * (segments + 1) + s, c = a + segments + 1;
      b.idx.push(a, a + 1, c, a + 1, c + 1, c);
    }
  return b;
}

function cylinder(r, h, segments = 48) {
  const b = meshBuilder();
  for (const y of [-h, h])
    for (let s = 0; s <= segments; ++s) {
      const phi = (s / segments) * Math.PI * 2;
      const n = [Math.cos(phi), 0, Math.sin(phi)];
      b.vertex([n[0] * r, y, n[2] * r], n);
    }
  for (let s = 0; s < segments; ++s) b.idx.push(s, s + segments + 1, s + 1, s + 1, s + segments + 1, s + segments + 2);
  for (const [y, ny] of [[h, 1], [-h, -1]]) {
    const center = b.vertex([0, y, 0], [0, ny, 0]);
    const first = b.v.length / 6;
    for (let s = 0; s <= segments; ++s) {
      const phi = (s / segments) * Math.PI * 2;
      b.vertex([Math.cos(phi) * r, y, Math.sin(phi) * r], [0, ny, 0]);
    }
    for (let s = 0; s < segments; ++s) {
      if (ny > 0) b.idx.push(center, first + s + 1, first + s);
      else b.idx.push(center, first + s, first + s + 1);
    }
  }
  return b;
}

function box(hx, hy, hz) {
  const b = meshBuilder();
  const faces = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[-1, 0, 0], [0, 1, 0], [0, 0, -1]], [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
    [[0, -1, 0], [0, 0, -1], [1, 0, 0]], [[0, 0, 1], [1, 0, 0], [0, 1, 0]], [[0, 0, -1], [-1, 0, 0], [0, 1, 0]]];
  const hs = [hx, hy, hz];
  for (const [n, u, w] of faces) {
    const first = b.v.length / 6;
    for (const [su, sw] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
      const p = [0, 1, 2].map((k) => (n[k] + u[k] * su + w[k] * sw) * hs[k]);
      b.vertex(p, n);
    }
    b.idx.push(first, first + 2, first + 1, first, first + 3, first + 2);
  }
  return b;
}

function torus(R, r, segments = 64, sides = 24) {
  const b = meshBuilder();
  for (let i = 0; i <= segments; ++i) {
    const u = (i / segments) * Math.PI * 2;
    for (let j = 0; j <= sides; ++j) {
      const v = (j / sides) * Math.PI * 2;
      const n = [Math.cos(v) * Math.cos(u), Math.sin(v), Math.cos(v) * Math.sin(u)];
      b.vertex([(R + r * Math.cos(v)) * Math.cos(u), r * Math.sin(v), (R + r * Math.cos(v)) * Math.sin(u)], n);
    }
  }
  for (let i = 0; i < segments; ++i)
    for (let j = 0; j < sides; ++j) {
      const a = i * (sides + 1) + j, c = a + sides + 1;
      b.idx.push(a, c, a + 1, a + 1, c, c + 1);
    }
  return b;
}

function colliderMesh(c) {
  const s = c.size;
  switch (c.type) {
    case COLLIDER.SPHERE: return sphereLike(s[0], 0);
    case COLLIDER.CAPSULE: return sphereLike(s[0], s[1]);
    case COLLIDER.BOX: return box(s[0], s[1], s[2]);
    case COLLIDER.CYLINDER: return cylinder(s[0], s[1]);
    default: return torus(s[0], s[1]);
  }
}

// ---------------------------------------------------------------- renderer

export class Renderer {
  constructor(device, context, format) {
    this.device = device;
    this.context = context;
    this.format = format;
    this.viewUbo = device.createBuffer({ size: VIEW_FLOATS * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.drawUbo = device.createBuffer({ size: MAX_DRAWS * DRAW_STRIDE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.matUbo = device.createBuffer({ size: 20 * 160, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.drawData = new Float32Array(MAX_DRAWS * DRAW_STRIDE / 4);
    this.shadowMap = device.createTexture({ size: [SHADOW_SIZE, SHADOW_SIZE], format: 'depth32float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    this.shadowSampler = device.createSampler({ compare: 'less-equal', magFilter: 'linear', minFilter: 'linear' });

    const VF = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;
    this.mainLayout0 = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: VF, buffer: { type: 'uniform' } },
      { binding: 1, visibility: VF, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 96 } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'comparison' } },
    ] });
    this.shadowLayout0 = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: VF, buffer: { type: 'uniform' } },
      { binding: 1, visibility: VF, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 96 } },
    ] });
    this.clothLayout1 = device.createBindGroupLayout({ entries: [0, 1, 2].map((b) => ({ binding: b, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } })) });
    this.mainGroup0 = device.createBindGroup({ layout: this.mainLayout0, entries: [
      { binding: 0, resource: { buffer: this.viewUbo } },
      { binding: 1, resource: { buffer: this.drawUbo, size: 96 } },
      { binding: 2, resource: { buffer: this.matUbo } },
      { binding: 3, resource: this.shadowMap.createView() },
      { binding: 4, resource: this.shadowSampler },
    ] });
    this.shadowGroup0 = device.createBindGroup({ layout: this.shadowLayout0, entries: [
      { binding: 0, resource: { buffer: this.viewUbo } },
      { binding: 1, resource: { buffer: this.drawUbo, size: 96 } },
    ] });

    this.module = device.createShaderModule({ label: 'render', code: COMMON + CLOTH + MESH });
    const clothVertex = { arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x2' }, { shaderLocation: 1, offset: 8, format: 'uint32' }] };
    const meshVertex = { arrayStride: 24, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }, { shaderLocation: 1, offset: 12, format: 'float32x3' }] };
    const shadowDepth = { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less', depthBias: 2, depthBiasSlopeScale: 2.5 };
    const mainDepth = { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' };
    const layout = (groups) => device.createPipelineLayout({ bindGroupLayouts: groups });
    const m = this.module;
    this.pipelines = {
      shadowCloth: device.createRenderPipeline({ layout: layout([this.shadowLayout0, this.clothLayout1]), vertex: { module: m, entryPoint: 'vs_cloth_shadow', buffers: [clothVertex] },
        primitive: { topology: 'triangle-list', cullMode: 'none' }, depthStencil: shadowDepth }),
      shadowMesh: device.createRenderPipeline({ layout: layout([this.shadowLayout0]), vertex: { module: m, entryPoint: 'vs_mesh_shadow', buffers: [meshVertex] },
        primitive: { topology: 'triangle-list', cullMode: 'none' }, depthStencil: shadowDepth }),
      cloth: device.createRenderPipeline({ layout: layout([this.mainLayout0, this.clothLayout1]), vertex: { module: m, entryPoint: 'vs_cloth', buffers: [clothVertex] },
        fragment: { module: m, entryPoint: 'fs_cloth', targets: [{ format }] }, primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: mainDepth, multisample: { count: 4, alphaToCoverageEnabled: true } }),
      mesh: device.createRenderPipeline({ layout: layout([this.mainLayout0]), vertex: { module: m, entryPoint: 'vs_mesh', buffers: [meshVertex] },
        fragment: { module: m, entryPoint: 'fs_mesh', targets: [{ format }] }, primitive: { topology: 'triangle-list', cullMode: 'none' },
        depthStencil: mainDepth, multisample: { count: 4 } }),
      sky: device.createRenderPipeline({ layout: layout([this.mainLayout0]), vertex: { module: m, entryPoint: 'vs_sky' },
        fragment: { module: m, entryPoint: 'fs_sky', targets: [{ format }] }, primitive: { topology: 'triangle-list' },
        depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'always' }, multisample: { count: 4 } }),
    };
    this.width = 0;
    this.height = 0;
  }

  async checkShaders() {
    const info = await this.module.getCompilationInfo();
    return info.messages.filter((m) => m.type === 'error').map((m) => `render:${m.lineNum}:${m.linePos} ${m.message}`);
  }

  resize(width, height) {
    if (width === this.width && height === this.height) return;
    this.width = width;
    this.height = height;
    this.color?.destroy();
    this.depth?.destroy();
    this.color = this.device.createTexture({ size: [width, height], sampleCount: 4, format: this.format, usage: GPUTextureUsage.RENDER_ATTACHMENT });
    this.depth = this.device.createTexture({ size: [width, height], sampleCount: 4, format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT });
  }

  setScene(scene, sim) {
    for (const m of this.meshes || []) { m.vb.destroy(); m.ib.destroy(); }
    const upload = (b) => {
      const vb = this.device.createBuffer({ size: b.v.length * 4, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(vb, 0, new Float32Array(b.v));
      const ib = this.device.createBuffer({ size: Math.ceil(b.idx.length * 4 / 4) * 4, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
      this.device.queue.writeBuffer(ib, 0, new Uint32Array(b.idx));
      return { vb, ib, count: b.idx.length };
    };
    this.meshes = scene.colliders.map((c) => upload(colliderMesh(c)));
    const floor = meshBuilder();
    const F = 400;
    for (const [x, z] of [[-F, -F], [F, -F], [F, F], [-F, F]]) floor.vertex([x, 0, z], [0, 1, 0]);
    floor.idx.push(0, 2, 1, 0, 3, 2);
    this.floor = upload(floor);
    this.meshes.push(this.floor);

    this.clothGroup1 = this.device.createBindGroup({ layout: this.clothLayout1, entries: [
      { binding: 0, resource: { buffer: sim.buffers.pos } },
      { binding: 1, resource: { buffer: sim.buffers.normals } },
      { binding: 2, resource: { buffer: sim.buffers.tangents } },
    ] });

    // The shadow covers the cloth with some margin, unless the scene says otherwise.
    const p = sim.initialPositions;
    const lo = [1e9, 1e9, 1e9], hi = [-1e9, -1e9, -1e9];
    for (let i = 0; i < p.length; i += 4)
      for (let k = 0; k < 3; ++k) {
        lo[k] = Math.min(lo[k], p[i + k]);
        hi[k] = Math.max(hi[k], p[i + k]);
      }
    this.shadowCenter = [(lo[0] + hi[0]) / 2, Math.max((lo[1] + hi[1]) / 2 - 0.4, 0.4), (lo[2] + hi[2]) / 2];
    this.shadowRadius = scene.shadowRadius > 0 ? scene.shadowRadius : vec3.length(vec3.sub(hi, lo)) * 0.5 + 1.2;
  }

  writeMaterials(fabrics, ranges) {
    const f = new Float32Array(20 * 40);
    fabrics.slice(0, 20).forEach((fab, i) => {
      const b = i * 40;
      f.set([...fab.warpColor, fab.warpMetal], b);
      f.set([...fab.weftColor, fab.weftMetal], b + 4);
      f.set([...fab.backTint, fab.backSwap ? 1 : 0], b + 8);
      f.set([...fab.sheenColor, fab.sheenRough], b + 12);
      f.set([fab.warpRoughAlong, fab.warpRoughAcross, fab.weftRoughAlong, fab.weftRoughAcross], b + 16);
      f.set([fab.f0, fab.specTint, fab.orenNayar, fab.normalStrength], b + 20);
      f.set([...fab.transTint, fab.opacity], b + 24);
      f.set([fab.threadsU, fab.threadsV, fab.weave, fab.threads], b + 28);
      f.set([fab.slub, fab.colorVar, fab.motif, fab.motifSize], b + 32);
      const r = ranges[i];
      f.set([r.cutRadius, r.cutCenter[0], r.cutCenter[1], 0], b + 36);
    });
    this.device.queue.writeBuffer(this.matUbo, 0, f);
  }

  render(encoder, view, sim, scene, fabrics, camera, time) {
    const device = this.device;
    const aspect = this.width / this.height;
    const eye = camera.position();
    const proj = perspective((camera.fov * Math.PI) / 180, aspect, 0.05, 400);
    const viewMat = lookAt(eye, camera.target, [0, 1, 0]);
    const viewProj = multiply(proj, viewMat);

    const key = scene.lights[0], fill = scene.lights[1] || { direction: [0, 1, 0], color: [0, 0, 0], intensity: 0, angular: 0.1 };
    const R = this.shadowRadius;
    const range = R * 2 + 2;
    const lightEye = vec3.add(this.shadowCenter, vec3.scale(key.direction, R + 1));
    const up = Math.abs(key.direction[1]) > 0.99 ? [0, 0, 1] : [0, 1, 0];
    const shadowVP = multiply(ortho(-R, R, -R, R, 0, range), lookAt(lightEye, this.shadowCenter, up));

    const v = new Float32Array(VIEW_FLOATS);
    v.set(viewProj, 0);
    v.set(invert(viewProj), 16);
    v.set(shadowVP, 32);
    v.set([...eye, time], 48);
    v.set([...key.direction, key.angular], 52);
    v.set(key.color.map((c) => c * key.intensity), 56);
    v.set([...fill.direction, fill.angular], 60);
    v.set(fill.color.map((c) => c * fill.intensity), 64);
    v.set([scene.environment, scene.environmentIntensity, 1.0, 1 / SHADOW_SIZE], 68);
    v.set([...scene.floorColor, scene.floorRoughness], 72);
    v.set([scene.floorFade[0], scene.floorFade[1], 0, 0], 76);
    v.set([range, R * 2, 0.0015 / range, 0], 80);
    device.queue.writeBuffer(this.viewUbo, 0, v);
    this.writeMaterials(fabrics, sim.data.ranges);

    // Draw slots: colliders, floor, cloths.
    let slot = 0;
    const d = this.drawData;
    const setDraw = (model, color, params) => {
      const w = slot * (DRAW_STRIDE / 4);
      d.set(model, w);
      d.set(color, w + 16);
      d.set(params, w + 20);
      return slot++ * DRAW_STRIDE;
    };
    const colliderSlots = scene.colliders.map((c) => {
      const pose = colliderPose(c, time);
      return setDraw(multiply(translation(pose.position), quat.toMat4(pose.rotation)), [...c.color, c.roughness], [c.metallic, 0, 0, 0]);
    });
    const floorSlot = setDraw(identity(), [0, 0, 0, 1], [0, 1, 0, 0]);
    const clothSlots = sim.data.ranges.map((r, i) => setDraw(identity(), [0, 0, 0, 0], [r.cutRadius, r.cutCenter[0], r.cutCenter[1], i]));
    device.queue.writeBuffer(this.drawUbo, 0, d, 0, slot * DRAW_STRIDE / 4);

    // ---- shadow map
    {
      const pass = encoder.beginRenderPass({ colorAttachments: [], depthStencilAttachment: { view: this.shadowMap.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' } });
      pass.setPipeline(this.pipelines.shadowMesh);
      scene.colliders.forEach((c, i) => {
        const mesh = this.meshes[i];
        pass.setBindGroup(0, this.shadowGroup0, [colliderSlots[i]]);
        pass.setVertexBuffer(0, mesh.vb);
        pass.setIndexBuffer(mesh.ib, 'uint32');
        pass.drawIndexed(mesh.count);
      });
      pass.setPipeline(this.pipelines.shadowCloth);
      pass.setBindGroup(1, this.clothGroup1);
      pass.setVertexBuffer(0, sim.buffers.renderVertices);
      pass.setIndexBuffer(sim.buffers.indices, 'uint32');
      sim.data.ranges.forEach((r, i) => {
        pass.setBindGroup(0, this.shadowGroup0, [clothSlots[i]]);
        pass.drawIndexed(r.indexCount, 1, r.firstIndex);
      });
      pass.end();
    }

    // ---- main pass
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: this.color.createView(), resolveTarget: view, clearValue: [0, 0, 0, 1], loadOp: 'clear', storeOp: 'discard' }],
      depthStencilAttachment: { view: this.depth.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'discard' },
    });
    pass.setPipeline(this.pipelines.sky);
    pass.setBindGroup(0, this.mainGroup0, [floorSlot]);
    pass.draw(3);
    pass.setPipeline(this.pipelines.mesh);
    const drawMesh = (mesh, offset) => {
      pass.setBindGroup(0, this.mainGroup0, [offset]);
      pass.setVertexBuffer(0, mesh.vb);
      pass.setIndexBuffer(mesh.ib, 'uint32');
      pass.drawIndexed(mesh.count);
    };
    drawMesh(this.floor, floorSlot);
    scene.colliders.forEach((c, i) => drawMesh(this.meshes[i], colliderSlots[i]));
    pass.setPipeline(this.pipelines.cloth);
    pass.setBindGroup(1, this.clothGroup1);
    pass.setVertexBuffer(0, sim.buffers.renderVertices);
    pass.setIndexBuffer(sim.buffers.indices, 'uint32');
    sim.data.ranges.forEach((r, i) => {
      pass.setBindGroup(0, this.mainGroup0, [clothSlots[i]]);
      pass.drawIndexed(r.indexCount, 1, r.firstIndex);
    });
    pass.end();
  }
}
