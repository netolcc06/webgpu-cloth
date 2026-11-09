// CPU side cloth construction: grid meshes in material space, constraints, graph colouring and adjacency.

// Greedy graph colouring: constraints that share a particle never get the same colour, so every colour can be
// solved in parallel while the whole stays a Gauss-Seidel iteration. Items are reordered by colour.
function colourAndSort(items, vertsOf, vertexCount) {
  const used = new Uint32Array(vertexCount);
  const colour = new Uint8Array(items.length);
  let maxColour = 0;
  for (let c = 0; c < items.length; ++c) {
    let mask = 0;
    for (const v of vertsOf(items[c])) mask |= used[v];
    let col = 0;
    while (mask & (1 << col)) col++;
    if (col >= 32) throw new Error('Constraint graph needs more than 32 colours');
    colour[c] = col;
    maxColour = Math.max(maxColour, col);
    for (const v of vertsOf(items[c])) used[v] |= 1 << col;
  }
  const counts = new Array(maxColour + 1).fill(0);
  for (const c of colour) counts[c]++;
  const batches = [];
  let offset = 0;
  for (let c = 0; c <= maxColour; ++c) {
    batches.push({ first: offset, count: counts[c] });
    offset += counts[c];
  }
  const cursor = batches.map((b) => b.first);
  const sorted = new Array(items.length);
  for (let i = 0; i < items.length; ++i) sorted[cursor[colour[i]]++] = items[i];
  return { items: sorted, batches };
}

const cross2 = (a, b) => a[0] * b[1] - a[1] * b[0];

export function createClothData() {
  return { positions: [], init: [], rest: [], stretch: [], bend: [], pendingLimits: [], attachments: [], hasAttachments: false,
    aeroStencil: [], gridNeighbors: [], renderVertices: [], indices: [], ranges: [], maxSpacing: 0 };
}

export function addCloth(data, desc, clothId) {
  const { nu, nv } = desc;
  const du = desc.width / (desc.wrapU ? nu : nu - 1);
  const dv = desc.height / (nv - 1);
  const quadsU = desc.wrapU ? nu : nu - 1;

  // Vertices that survive the cut shape.
  const inside = new Uint8Array(nu * nv).fill(1);
  if (desc.cutRadius > 0) {
    const cx = desc.width * 0.5, cy = desc.height * 0.5, keep = desc.cutRadius + 1.5 * Math.max(du, dv);
    for (let j = 0; j < nv; ++j)
      for (let i = 0; i < nu; ++i) inside[j * nu + i] = Math.hypot(i * du - cx, j * dv - cy) <= keep ? 1 : 0;
  }
  const isInside = (i, j) => inside[j * nu + ((i + nu) % nu)] !== 0;
  const gridId = (i, j) => j * nu + (i % nu);

  const tris = [];
  const referenced = new Uint8Array(nu * nv);
  const addTri = (a, b, c) => {
    if (!isInside(a[0], a[1]) || !isInside(b[0], b[1]) || !isInside(c[0], c[1])) return;
    const t = { grid: [gridId(a[0], a[1]), gridId(b[0], b[1]), gridId(c[0], c[1])], coords: [a, b, c] };
    for (const g of t.grid) referenced[g] = 1;
    tris.push(t);
  };
  // Alternating diagonals: avoids the directional bias of a regular triangulation.
  for (let j = 0; j < nv - 1; ++j)
    for (let i = 0; i < quadsU; ++i) {
      const a = [i, j], b = [i + 1, j], c = [i + 1, j + 1], d = [i, j + 1];
      if ((i + j) % 2 === 0) {
        addTri(a, b, c);
        addTri(a, c, d);
      } else {
        addTri(a, b, d);
        addTri(b, c, d);
      }
    }

  // Particles.
  const firstParticle = data.positions.length;
  const particleOf = new Int32Array(nu * nv).fill(-1);
  for (let j = 0; j < nv; ++j)
    for (let i = 0; i < nu; ++i) {
      const g = j * nu + i;
      if (!referenced[g]) continue;
      const u = i * du, v = j * dv;
      particleOf[g] = data.positions.length;
      const p = desc.place(u, v);
      const pin = desc.pin ? desc.pin(u, v, i, j) : -1;
      data.positions.push([p[0], p[1], p[2], pin >= 0 ? 0 : 1]);
      data.init.push([p[0], p[1], p[2], pin]);
      data.rest.push([u, v, clothId, 0]);
    }

  // Stretch triangles and the hinges between them.
  const edges = new Map();
  const restCoord = (c) => [c[0] * du, c[1] * dv];
  const firstTri = data.stretch.length;
  for (let t = 0; t < tris.length; ++t) {
    const tri = tris[t];
    const ids = tri.grid.map((g) => particleOf[g]);
    const p0 = restCoord(tri.coords[0]), p1 = restCoord(tri.coords[1]), p2 = restCoord(tri.coords[2]);
    const e1 = [p1[0] - p0[0], p1[1] - p0[1]], e2 = [p2[0] - p0[0], p2[1] - p0[1]];
    const det = cross2(e1, e2);
    const area = 0.5 * Math.abs(det);
    // Dm^-1 for Dm = [e1 e2]: Fu = s1 (x1-x0) + s2 (x2-x0), Fv = t1 (x1-x0) + t2 (x2-x0).
    data.stretch.push({ idx: [ids[0], ids[1], ids[2], clothId], grad: [e2[1] / det, -e1[1] / det, -e2[0] / det, e1[0] / det], area, local: t });
    for (const id of ids) data.rest[id][3] += area / 3;
    for (let e = 0; e < 3; ++e) {
      const a = ids[e], b = ids[(e + 1) % 3];
      const key = a < b ? a * 16777216 + b : b * 16777216 + a;
      let entry = edges.get(key);
      if (!entry) edges.set(key, (entry = []));
      entry.push({ tri: firstTri + t, opposite: (e + 2) % 3 });
    }
  }

  // Isometric bending model (Bergou et al. 2006), written as a linear vector constraint.
  const cot = (u, v) => (u[0] * v[0] + u[1] * v[1]) / Math.abs(cross2(u, v));
  const sub2 = (a, b) => [a[0] - b[0], a[1] - b[1]];
  for (const refs of edges.values()) {
    if (refs.length !== 2) continue;
    const sa = data.stretch[refs[0].tri], sb = data.stretch[refs[1].tri];
    const ta = tris[refs[0].tri - firstTri], tb = tris[refs[1].tri - firstTri];
    const oa = refs[0].opposite, ob = refs[1].opposite;
    const x2 = sa.idx[oa], x3 = sb.idx[ob], x0 = sa.idx[(oa + 1) % 3], x1 = sa.idx[(oa + 2) % 3];
    const coordOf = (tri, s, particle) => {
      for (let k = 0; k < 3; ++k) if (s.idx[k] === particle) return restCoord(tri.coords[k]);
      throw new Error('Bad bending stencil');
    };
    const a0 = coordOf(ta, sa, x0), a1 = coordOf(ta, sa, x1), a2 = coordOf(ta, sa, x2);
    const b0 = coordOf(tb, sb, x0), b1 = coordOf(tb, sb, x1), b3 = coordOf(tb, sb, x3);
    const c01 = cot(sub2(a1, a0), sub2(a2, a0)), c03 = cot(sub2(a0, a1), sub2(a2, a1));
    const c02 = cot(sub2(b1, b0), sub2(b3, b0)), c04 = cot(sub2(b0, b1), sub2(b3, b1));
    const edge = sub2(a1, a0);
    data.bend.push({ idx: [x0, x1, x2, x3], K: [c03 + c04, c01 + c02, -c01 - c03, -c02 - c04],
      extra: [(sa.area + sb.area) / 3, (edge[1] * edge[1]) / (edge[0] * edge[0] + edge[1] * edge[1]), clothId, 0] });
  }

  // Long range strain limit along the yarns, at geometric steps. Two parities per phase.
  let level = 0;
  for (let step = 4; step < Math.max(nu, nv); step *= 4, ++level)
    for (let axis = 0; axis < 2; ++axis) {
      const phase = level * 2 + axis;
      while (data.pendingLimits.length <= phase) data.pendingLimits.push([[], []]);
      for (let j = 0; j < nv; ++j)
        for (let i = 0; i < nu; ++i) {
          const i2 = axis === 0 ? i + step : i, j2 = axis === 0 ? j : j + step;
          if (i2 >= nu || j2 >= nv) continue;
          const a = particleOf[j * nu + i], b = particleOf[j2 * nu + i2];
          if (a < 0 || b < 0) continue;
          const parity = Math.floor((axis === 0 ? i : j) / step) % 2;
          data.pendingLimits[phase][parity].push([a, b, step * (axis === 0 ? du : dv), clothId]);
        }
    }

  // Long range attachments: nearest pin in material space.
  const pins = [];
  for (let j = 0; j < nv; ++j)
    for (let i = 0; i < nu; ++i) {
      const p = particleOf[j * nu + i];
      if (p >= 0 && data.init[p][3] >= 0) pins.push([p, i * du, j * dv]);
    }
  while (data.attachments.length < data.positions.length) data.attachments.push([0xffffffff, 0]);
  if (pins.length) {
    data.hasAttachments = true;
    const period = desc.wrapU ? desc.width : 0;
    for (let j = 0; j < nv; ++j)
      for (let i = 0; i < nu; ++i) {
        const p = particleOf[j * nu + i];
        if (p < 0 || data.init[p][3] >= 0) continue;
        const u = i * du, v = j * dv;
        let best = 1e18, bestPin = 0;
        for (const pin of pins) {
          let a = Math.abs(pin[1] - u);
          if (period > 0) a = Math.min(a, period - a);
          const b = pin[2] - v, d2 = a * a + b * b;
          if (d2 < best) {
            best = d2;
            bestPin = pin[0];
          }
        }
        data.attachments[p] = [bestPin, Math.sqrt(best)];
      }
  }

  // Wide stencil for the aerodynamic normal (about 4 cm), and the adjacent grid vertices.
  const reach = Math.min(Math.max(Math.round(0.04 / Math.max(du, dv)), 2), 24);
  const walk = (i, j, di, dj) => {
    for (let s = reach; s >= 1; --s) {
      let a = i + di * s;
      const b = j + dj * s;
      if (desc.wrapU) a = (a + nu) % nu;
      if (a < 0 || b < 0 || a >= nu || b >= nv) continue;
      if (particleOf[b * nu + a] >= 0) return particleOf[b * nu + a];
    }
    return particleOf[j * nu + i];
  };
  for (let j = 0; j < nv; ++j)
    for (let i = 0; i < nu; ++i) {
      const p = particleOf[j * nu + i];
      if (p < 0) continue;
      data.aeroStencil[p] = [walk(i, j, 1, 0), walk(i, j, -1, 0), walk(i, j, 0, 1), walk(i, j, 0, -1)];
      const adjacent = (di, dj) => {
        let a = i + di;
        const b = j + dj;
        if (desc.wrapU) a = (a + nu) % nu;
        if (a < 0 || b < 0 || a >= nu || b >= nv || particleOf[b * nu + a] < 0) return p;
        return particleOf[b * nu + a];
      };
      data.gridNeighbors[p] = [adjacent(1, 0), adjacent(-1, 0), adjacent(0, 1), adjacent(0, -1)];
    }

  // Render mesh: a tube needs duplicated vertices along the seam for continuous material coordinates.
  const renderU = desc.wrapU ? nu + 1 : nu;
  const renderOf = new Int32Array(renderU * nv).fill(-1);
  const firstRender = data.renderVertices.length;
  for (let j = 0; j < nv; ++j)
    for (let i = 0; i < renderU; ++i) {
      const p = particleOf[j * nu + (i % nu)];
      if (p < 0) continue;
      renderOf[j * renderU + i] = data.renderVertices.length - firstRender;
      data.renderVertices.push([i * du, j * dv, p]);
    }
  const firstIndex = data.indices.length;
  for (const tri of tris) for (const c of tri.coords) data.indices.push(firstRender + renderOf[c[1] * renderU + c[0]]);
  data.ranges.push({ firstIndex, indexCount: data.indices.length - firstIndex, firstParticle, particleCount: data.positions.length - firstParticle,
    spacing: Math.max(du, dv), fabric: desc.fabric, cutRadius: desc.cutRadius, cutCenter: [desc.width * 0.5, desc.height * 0.5] });
  data.maxSpacing = Math.max(data.maxSpacing, Math.max(du, dv));
}

export function finalizeCloths(data) {
  const n = data.positions.length;
  const st = colourAndSort(data.stretch, (s) => [s.idx[0], s.idx[1], s.idx[2]], n);
  data.stretch = st.items;
  data.stretchBatches = st.batches;
  const bd = colourAndSort(data.bend, (b) => b.idx, n);
  data.bend = bd.items;
  data.bendBatches = bd.batches;

  data.limits = [];
  data.limitPhases = [];
  for (const phase of data.pendingLimits) {
    const batches = [];
    for (const edges of phase) {
      if (!edges.length) continue;
      batches.push({ first: data.limits.length, count: edges.length });
      for (const e of edges) data.limits.push(e);
    }
    if (batches.length) data.limitPhases.push(batches);
  }
  delete data.pendingLimits;

  // Particle -> incident triangles (after sorting, so that indices match the GPU buffer).
  const offsets = new Uint32Array(n + 1);
  for (const s of data.stretch) for (let k = 0; k < 3; ++k) offsets[s.idx[k] + 1]++;
  for (let i = 0; i < n; ++i) offsets[i + 1] += offsets[i];
  const vertexTris = new Uint32Array(offsets[n]);
  const cursor = offsets.slice(0, n);
  data.stretch.forEach((s, t) => {
    for (let k = 0; k < 3; ++k) vertexTris[cursor[s.idx[k]]++] = t;
  });
  data.vertexTriOffsets = offsets;
  data.vertexTris = vertexTris;
}

export function buildClothData(scene) {
  const data = createClothData();
  scene.cloths.forEach((c, i) => addCloth(data, c, i));
  finalizeCloths(data);
  return data;
}
