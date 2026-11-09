// WGSL of the renderer: cloth with a procedural weave, colliders, floor and sky.
export const COMMON = /* wgsl */ `
const PI = 3.14159265359;

struct View {
  viewProj: mat4x4f, invViewProj: mat4x4f, shadowVP: mat4x4f,
  camPos: vec4f,      // xyz, w: time
  l0Dir: vec4f,       // xyz: towards the key light, w: tan of its half angle
  l0Col: vec4f,
  l1Dir: vec4f,
  l1Col: vec4f,
  env: vec4f,         // x: type (0 studio, 1 daylight), y: intensity, z: exposure, w: shadow texel size
  floorCol: vec4f,    // rgb, w: roughness
  floorFade: vec4f,   // x: start, y: end
  shadowInfo: vec4f,  // x: depth range (m), y: width (m), z: bias
};
struct Draw { model: mat4x4f, color: vec4f, params: vec4f };
struct ClothMat {
  warpColor: vec4f, weftColor: vec4f, backColor: vec4f, sheen: vec4f, rough: vec4f, shade: vec4f, trans: vec4f,
  weave: vec4f,   // threads per metre along u and v, weave id, thread colour id
  extra: vec4f,   // slub, colour variation, motif id, motif size
  cut: vec4f,     // cut radius, cut centre (material coordinates), unused
};

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<uniform> draw: Draw;
@group(0) @binding(2) var<uniform> mats: array<ClothMat, 20>;
@group(0) @binding(3) var shadowMap: texture_depth_2d;
@group(0) @binding(4) var shadowSampler: sampler_comparison;

fn hash1(n: u32) -> f32 {
  var x = n;
  x = x ^ (x >> 16u); x = x * 0x7feb352du; x = x ^ (x >> 15u); x = x * 0x846ca68bu; x = x ^ (x >> 16u);
  return f32(x) / 4294967295.0;
}
fn hash2(i: i32, j: i32) -> f32 { return hash1((bitcast<u32>(i) * 73856093u) ^ (bitcast<u32>(j) * 19349663u)); }
fn ign(p: vec2f) -> f32 { return fract(52.9829189 * fract(dot(p, vec2f(0.06711056, 0.00583715)))); }

// Environment light: a studio with soft boxes in the direction of the lights, or a daylight sky.
fn envRadiance(dir: vec3f, rough: f32) -> vec3f {
  var c: vec3f;
  let y = dir.y;
  if (view.env.x < 0.5) {
    c = mix(vec3f(0.035, 0.035, 0.04), vec3f(0.22, 0.22, 0.23), smoothstep(-0.3, 0.9, y));
    let sharp0 = mix(60.0, 3.0, rough);
    c += view.l0Col.rgb * 0.08 * pow(max(dot(dir, view.l0Dir.xyz), 0.0), sharp0) * (sharp0 + 2.0) / 8.0;
    c += view.l1Col.rgb * 0.06 * pow(max(dot(dir, view.l1Dir.xyz), 0.0), sharp0 * 0.6) * (sharp0 * 0.6 + 2.0) / 8.0;
  } else {
    let sky = mix(vec3f(0.62, 0.72, 0.85), vec3f(0.18, 0.36, 0.72), clamp(y, 0.0, 1.0));
    let ground = vec3f(0.16, 0.15, 0.12);
    c = mix(ground, sky, smoothstep(-0.08, 0.05, y)) * 0.9;
    let sharp = mix(400.0, 4.0, rough);
    c += view.l0Col.rgb * 0.05 * pow(max(dot(dir, view.l0Dir.xyz), 0.0), sharp) * (sharp + 2.0) / 16.0;
  }
  return c * view.env.y;
}

fn background(dir: vec3f) -> vec3f {
  if (view.env.x < 0.5) {
    let glow = exp(-pow(max(dir.y - 0.02, 0.0) * 2.6, 2.0));
    let below = smoothstep(0.0, -0.3, dir.y);
    return mix(vec3f(0.024, 0.024, 0.026) + vec3f(0.10, 0.10, 0.104) * glow, vec3f(0.12), below) * view.env.y;
  }
  return envRadiance(dir, 0.0);
}

// Percentage-closer soft shadows: blocker search, penumbra estimate, filtering.
fn shadowVisibility(p: vec3f, n: vec3f, frag: vec2f) -> f32 {
  let sp = view.shadowVP * vec4f(p + n * 0.01, 1.0);
  let ndc = sp.xyz / sp.w;
  let uv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
  if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0)) || ndc.z >= 1.0) { return 1.0; }
  let z = ndc.z - view.shadowInfo.z;
  let phi = ign(frag) * 2.0 * PI;
  let size = vec2f(textureDimensions(shadowMap));
  let texel = 1.0 / size.x;
  let search = max(view.l0Dir.w * 2.0 * 2.0 / view.shadowInfo.y, 2.0 * texel);
  var blockers = 0.0;
  var blockerSum = 0.0;
  for (var i = 0; i < 8; i++) {
    let r = sqrt((f32(i) + 0.5) / 8.0);
    let a = f32(i) * 2.39996 + phi;
    let s = clamp(uv + r * vec2f(cos(a), sin(a)) * search, vec2f(0.0), vec2f(0.999));
    let d = textureLoad(shadowMap, vec2i(s * size), 0);
    if (d < z) { blockers += 1.0; blockerSum += d; }
  }
  if (blockers == 0.0) { return 1.0; }
  let blockerDepth = blockerSum / blockers;
  let penumbra = clamp((ndc.z - blockerDepth) * view.shadowInfo.x * view.l0Dir.w * 2.0 / view.shadowInfo.y, 1.5 * texel, search);
  var lit = 0.0;
  for (var i = 0; i < 12; i++) {
    let r = sqrt((f32(i) + 0.5) / 12.0);
    let a = f32(i) * 2.39996 + phi;
    lit += textureSampleCompareLevel(shadowMap, shadowSampler, uv + r * vec2f(cos(a), sin(a)) * penumbra, z);
  }
  return lit / 12.0;
}

fn distributionGgx(NoH: f32, a: f32) -> f32 {
  let a2 = a * a;
  let d = NoH * NoH * (a2 - 1.0) + 1.0;
  return a2 / (PI * d * d + 1e-7);
}
fn visibilitySmith(NoV: f32, NoL: f32, a: f32) -> f32 {
  let a2 = a * a;
  let lv = NoL * sqrt(NoV * NoV * (1.0 - a2) + a2);
  let ll = NoV * sqrt(NoL * NoL * (1.0 - a2) + a2);
  return 0.5 / max(lv + ll, 1e-5);
}
fn fresnel(f0: vec3f, VoH: f32) -> vec3f { return f0 + (vec3f(1.0) - f0) * pow(1.0 - VoH, 5.0); }

// AgX tone curve (Troy Sobotka, fit by Benjamin Wrensch), then sRGB and a dither against banding.
fn agxContrast(x: vec3f) -> vec3f {
  let x2 = x * x; let x4 = x2 * x2;
  return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}
fn finish(colorIn: vec3f, frag: vec2f) -> vec3f {
  let toAgx = mat3x3f(0.842479062253094, 0.0423282422610123, 0.0423756549057051, 0.0784335999999992, 0.878468636469772, 0.0784336,
                      0.0792237451477643, 0.0791661274605434, 0.879142973793104);
  let fromAgx = mat3x3f(1.19687900512017, -0.0528968517574562, -0.0529716355144438, -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
                        -0.0990297440797205, -0.0989611768448433, 1.15107367264116);
  let minEv = -12.47393; let maxEv = 4.026069;
  var c = toAgx * (colorIn * view.env.z);
  c = clamp(log2(max(c, vec3f(1e-10))), vec3f(minEv), vec3f(maxEv));
  c = (c - minEv) / (maxEv - minEv);
  c = agxContrast(c);
  let luma = dot(c, vec3f(0.2126, 0.7152, 0.0722));
  c = pow(max(c, vec3f(0.0)), vec3f(1.25));
  c = luma + 1.15 * (c - luma);
  c = pow(max(fromAgx * c, vec3f(0.0)), vec3f(2.2));
  let lin = clamp(c, vec3f(0.0), vec3f(1.0));
  let srgb = select(1.055 * pow(lin, vec3f(1.0 / 2.4)) - 0.055, lin * 12.92, lin < vec3f(0.0031308));
  return srgb + (ign(frag) - 0.5) / 255.0;
}
`;

// ---------------------------------------------------------------- cloth

export const CLOTH = /* wgsl */ `
@group(1) @binding(0) var<storage, read> positions: array<vec4f>;
@group(1) @binding(1) var<storage, read> normals: array<vec4f>;
@group(1) @binding(2) var<storage, read> tangents: array<vec4f>;

struct VOut {
  @builtin(position) clip: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
  @location(2) warp: vec3f,
  @location(3) uv: vec2f,
};

@vertex fn vs_cloth(@location(0) uv: vec2f, @location(1) particle: u32) -> VOut {
  var o: VOut;
  let p = positions[particle].xyz;
  o.world = p;
  o.normal = normals[particle].xyz;
  o.warp = tangents[particle].xyz;
  o.uv = uv;
  o.clip = view.viewProj * vec4f(p, 1.0);
  return o;
}

@vertex fn vs_cloth_shadow(@location(0) uv: vec2f, @location(1) particle: u32) -> @builtin(position) vec4f {
  return view.shadowVP * vec4f(positions[particle].xyz, 1.0);
}

// ---- the weave: which yarn is on top in each crossing, as a loom draft would say
fn warpUp(weave: i32, i: i32, j: i32) -> bool {
  switch weave {
    case 2: { return ((i - j) & 3) < 2; }                       // 2/2 twill
    case 3: { return ((i - j) & 3) != 3; }                      // 3/1 twill
    case 4: { return ((i % 5) + 5) % 5 != ((j * 2) % 5 + 5) % 5; }   // 5 harness satin, warp faced
    case 5: { return ((i % 8) + 8) % 8 != ((j * 3) % 8 + 8) % 8; }   // 8 harness satin, warp faced
    case 6: {                                                   // herringbone
      let k = ((i % 16) + 16) % 16;
      let m = select(15 - k, k, k < 8);
      return ((m + j) & 3) < 2;
    }
    case 7: { return ((i & 3) >> 1u) == ((j & 3) >> 1u); }     // basket
    case 8: { return ((i & 3) >> 1u) == (j & 1); }             // oxford
    default: { return ((i + j) & 1) == 0; }                     // plain, gauze, knit, tulle
  }
}

fn yarnWidths(weave: i32) -> vec2f {
  switch weave {
    case 1: { return vec2f(0.5, 0.5); }
    case 4, 5: { return vec2f(1.0, 0.95); }
    case 8: { return vec2f(0.82, 1.0); }
    case 10: { return vec2f(0.28, 0.28); }
    default: { return vec2f(0.95, 0.95); }
  }
}

// Share of the warp in what is seen, averaged over a repeat: used where a pixel covers many threads.
fn warpShare(weave: i32) -> f32 {
  switch weave {
    case 3: { return 0.75; }
    case 4: { return 0.8; }
    case 5: { return 0.875; }
    case 8: { return 0.45; }
    default: { return 0.5; }
  }
}

fn srgbToLinear(c: vec3f) -> vec3f { return pow(c, vec3f(2.2)); }

// Thread colours set by their order, as on a loom: tartan, houndstooth, gingham, stripes.
fn threadColor(kind: i32, index: i32, isWarp: bool) -> vec3f {
  if (kind == 1) {
    var runs = array<i32, 12>(40, 4, 30, 4, 30, 4, 30, 4, 30, 4, 40, 36);
    var cols = array<vec3f, 12>(vec3f(0.62, 0.07, 0.09), vec3f(0.03, 0.03, 0.04), vec3f(0.05, 0.25, 0.15), vec3f(0.03, 0.03, 0.04),
      vec3f(0.07, 0.11, 0.26), vec3f(0.85, 0.72, 0.25), vec3f(0.07, 0.11, 0.26), vec3f(0.03, 0.03, 0.04), vec3f(0.05, 0.25, 0.15),
      vec3f(0.03, 0.03, 0.04), vec3f(0.62, 0.07, 0.09), vec3f(0.07, 0.11, 0.26));
    var k = ((index % 256) + 256) % 256;
    for (var r = 0; r < 12; r++) {
      if (k < runs[r]) { return srgbToLinear(cols[r]); }
      k -= runs[r];
    }
    return vec3f(0.0);
  }
  if (kind == 2) { return srgbToLinear(select(vec3f(0.86, 0.83, 0.76), vec3f(0.05, 0.05, 0.06), ((index % 8) + 8) % 8 < 4)); }
  if (kind == 3) { return srgbToLinear(select(vec3f(0.93, 0.93, 0.9), vec3f(0.08, 0.2, 0.5), ((index % 32) + 32) % 32 < 16)); }
  if (kind == 4) {
    if (!isWarp) { return srgbToLinear(vec3f(0.04, 0.07, 0.2)); }
    return srgbToLinear(select(vec3f(0.86, 0.82, 0.72), vec3f(0.04, 0.07, 0.2), ((index % 64) + 64) % 64 < 56));
  }
  return vec3f(1.0);
}

fn threadColorAverage(kind: i32, isWarp: bool) -> vec3f {
  if (kind == 1) { return srgbToLinear(vec3f(0.30, 0.13, 0.15)); }
  if (kind == 2) { return srgbToLinear(vec3f(0.62, 0.60, 0.55)); }
  if (kind == 3) { return srgbToLinear(vec3f(0.69, 0.71, 0.76)); }
  if (kind == 4) { return srgbToLinear(select(vec3f(0.04, 0.07, 0.2), vec3f(0.36, 0.36, 0.38), isWarp)); }
  return vec3f(1.0);
}

// Flower motifs on a half-drop lattice: damask face change, printed flowers, lace.
fn flowerMask(uv: vec2f, size: f32) -> vec2f {
  let g = uv / size;
  let row = floor(g.y);
  let cellId = vec2f(floor(g.x + 0.5 * row), row);
  let p = vec2f(fract(g.x + 0.5 * row), fract(g.y)) - 0.5;
  let r = length(p);
  let a = atan2(p.y, p.x) + hash2(i32(cellId.x), i32(cellId.y)) * 6.28;
  let petals = 0.28 + 0.1 * cos(a * 5.0);
  let edge = fwidth(r) * 1.5 + 1e-4;
  let inside = smoothstep(petals + edge, petals - edge, r);
  let heart = smoothstep(0.09 + edge, 0.09 - edge, r);
  return vec2f(inside, heart);
}

@fragment fn fs_cloth(vin: VOut, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  let ci = u32(draw.params.w);
  let m = mats[ci];
  // A circular cut is exact per pixel; the mesh only follows the grid.
  var cutAlpha = 1.0;
  if (draw.params.x > 0.0) {
    let sd = draw.params.x - length(vin.uv - draw.params.yz);
    cutAlpha = clamp(sd / max(fwidth(sd), 1e-6) + 0.5, 0.0, 1.0);
    if (cutAlpha <= 0.0) { discard; }
  }

  let N0 = normalize(vin.normal);
  let warpDir = normalize(vin.warp - N0 * dot(vin.warp, N0));
  let weftDir = cross(warpDir, N0);
  let Ng = select(-N0, N0, front);
  let V = normalize(view.camPos.xyz - vin.world);

  // ---- weave
  let weave = i32(m.weave.z);
  let threads = vin.uv * m.weave.xy;
  let footprint = max(length(dpdx(threads)), length(dpdy(threads)));
  let detail = 1.0 - smoothstep(0.35, 1.1, footprint);  // 1 when a pixel sees single yarns
  let i = i32(floor(threads.x));
  let j = i32(floor(threads.y));
  let f = fract(threads) - 0.5;
  let widths = yarnWidths(weave);
  let onWarp = abs(f.x) < widths.x * 0.5;
  let onWeft = abs(f.y) < widths.y * 0.5;
  var up = warpUp(weave, i, j);
  var motif = vec2f(0.0);
  let motifKind = i32(m.extra.z);
  if (motifKind > 0) { motif = flowerMask(vin.uv, m.extra.w); }
  if (motifKind == 1 && motif.x > 0.5) { up = !up; }  // damask: the loom swaps the weave face in the motif

  var coverage = 1.0;
  var warpVisible = 0.5;
  var micro = vec2f(0.0);
  if (weave == 10) {
    // Tulle: a fine hexagonal net.
    let q = vec2f(threads.x + 0.5 * f32(j & 1), threads.y * 1.15);
    let e = abs(fract(q) - 0.5);
    let line = max(e.x, e.y);
    coverage = mix(0.42, smoothstep(0.30, 0.38, line), detail);
  } else if (weave == 11) {
    // Pile: tufts in every direction, seen end on.
    micro = (vec2f(hash2(i, j), hash2(j, i + 17)) - 0.5) * 0.6 * detail;
  } else {
    let pickWarp = (onWarp && onWeft && up) || (onWarp && !onWeft);
    let pickWeft = (onWarp && onWeft && !up) || (onWeft && !onWarp);
    let yarnCoverage = select(0.0, 1.0, pickWarp || pickWeft);
    let avgCoverage = 1.0 - (1.0 - widths.x) * (1.0 - widths.y);
    coverage = mix(avgCoverage, yarnCoverage, detail);
    warpVisible = mix(warpShare(weave), select(0.0, 1.0, pickWarp), detail);
    if (weave == 9) { warpVisible = 0.5; }
    // Round yarns: the normal turns across each one.
    let across = select(f.y / (widths.y * 0.5), f.x / (widths.x * 0.5), pickWarp);
    let tilt = clamp(across, -1.0, 1.0) * 0.55;
    micro = select(vec2f(0.0, tilt), vec2f(tilt, 0.0), pickWarp) * detail * m.shade.w;
  }
  if (motifKind == 3) { coverage = max(coverage, motif.x); }

  // ---- colours of the yarns
  var warpColor = m.warpColor.rgb;
  var weftColor = m.weftColor.rgb;
  let lut = i32(m.weave.w);
  if (lut > 0) {
    // Averaged only once a pixel covers the narrowest run of colour, not a single thread.
    let narrowest = select(select(select(4.0, 16.0, lut == 3), 8.0, lut == 4), 4.0, lut <= 2);
    let colourDetail = 1.0 - smoothstep(narrowest * 0.5, narrowest * 1.5, footprint);
    warpColor *= mix(threadColorAverage(lut, true), threadColor(lut, i, true), colourDetail);
    weftColor *= mix(threadColorAverage(lut, false), threadColor(lut, j, false), colourDetail);
  }
  let variation = m.extra.y;
  warpColor *= 1.0 + (hash2(i, 7) - 0.5) * variation * detail;
  weftColor *= 1.0 + (hash2(13, j) - 0.5) * variation * detail;
  if (!front) {
    warpColor *= m.backColor.rgb;
    weftColor *= m.backColor.rgb;
    if (m.backColor.a > 0.5) { warpVisible = 1.0 - warpVisible; }
    micro = -micro;
  }
  // Round yarns hide each other: looking across the warp shows more warp.
  let acrossWarp = dot(V, weftDir); let alongWarp = dot(V, warpDir);
  let shift = 0.9 * (acrossWarp * acrossWarp - alongWarp * alongWarp);
  let warpWeight = clamp(warpVisible + shift * warpVisible * (1.0 - warpVisible) * 2.0, 0.0, 1.0);
  let weftWeight = 1.0 - warpWeight;
  var albedo = warpColor * warpWeight * (1.0 - m.warpColor.a) + weftColor * weftWeight * (1.0 - m.weftColor.a);
  if (motifKind == 2) {
    let ink = select(vec3f(0.62, 0.10, 0.18), vec3f(0.95, 0.70, 0.15), motif.y > 0.5);
    albedo = mix(albedo, srgbToLinear(ink), motif.x * 0.92);
  }

  let nz = sqrt(max(1.0 - dot(micro, micro), 0.01));
  var N = normalize(weftDir * micro.x + warpDir * micro.y + Ng * nz);
  if (dot(N, V) < 0.02) { N = normalize(N + V * (0.02 - dot(N, V))); }

  // ---- lighting: two anisotropic GGX lobes (warp, weft), Charlie sheen, Oren-Nayar, transmission
  let widen = (1.0 - detail) * 0.06;
  let NoV = max(dot(N, V), 1e-3);
  let sheenAlpha = max(m.sheen.a * m.sheen.a, 1e-3);
  let sheenColor = m.sheen.rgb;
  let sheenScale = 1.0 - max(sheenColor.r, max(sheenColor.g, sheenColor.b)) * 0.4;
  let metallic = clamp(m.warpColor.a * warpWeight + m.weftColor.a * weftWeight, 0.0, 1.0);
  let dielectric = vec3f(m.shade.x);
  let f0Warp = mix(dielectric * mix(vec3f(1.0), warpColor / max(max(warpColor.r, max(warpColor.g, warpColor.b)), 1e-3), m.shade.y), warpColor, m.warpColor.a);
  let f0Weft = mix(dielectric * mix(vec3f(1.0), weftColor / max(max(weftColor.r, max(weftColor.g, weftColor.b)), 1e-3), m.shade.y), weftColor, m.weftColor.a);
  let aWT = clamp(sqrt(m.rough.x * m.rough.x + widen), 0.03, 1.0); let aWB = clamp(sqrt(m.rough.y * m.rough.y + widen), 0.03, 1.0);
  let aFT = clamp(sqrt(m.rough.z * m.rough.z + widen), 0.03, 1.0); let aFB = clamp(sqrt(m.rough.w * m.rough.w + widen), 0.03, 1.0);
  let Tw = normalize(warpDir - N * dot(warpDir, N)); let Bw = cross(N, Tw);
  let Tf = normalize(weftDir - N * dot(weftDir, N)); let Bf = cross(N, Tf);
  let transmission = m.trans.rgb * (1.0 - metallic);
  let sigma = m.shade.z;
  let s2 = sigma * sigma;
  let onA = 1.0 - 0.5 * s2 / (s2 + 0.33);
  let onB = 0.45 * s2 / (s2 + 0.09);

  var direct = vec3f(0.0);
  for (var li = 0; li < 2; li++) {
    let L = select(view.l1Dir.xyz, view.l0Dir.xyz, li == 0);
    let radiance = select(view.l1Col.rgb, view.l0Col.rgb, li == 0);
    let NoLs = dot(N, L);
    let NgoL = dot(Ng, L);
    if (NgoL > -0.05 && NoLs > 0.0) {
      var vis = 1.0;
      if (li == 0) { vis = shadowVisibility(vin.world, Ng, vin.clip.xy); }
      let NoL = max(NoLs, 1e-3);
      let H = normalize(V + L);
      let VoH = max(dot(V, H), 0.0);
      let NoH = max(dot(N, H), 0.0);
      let LoV = dot(L, V);
      let sOn = LoV - NoL * NoV;
      let tOn = select(1.0, max(NoL, NoV), sOn > 0.0);
      let diffuse = albedo * (onA + onB * sOn / max(tOn, 1e-3)) / PI * sheenScale;
      // anisotropic GGX, one lobe per yarn set
      let dW = aWT * aWB; let vW = vec3f(aWB * dot(Tw, H), aWT * dot(Bw, H), dW * NoH); let w2W = dW / max(dot(vW, vW), 1e-8);
      let DW = dW * w2W * w2W / PI;
      let GW = 0.5 / max(NoL * length(vec3f(aWT * dot(Tw, V), aWB * dot(Bw, V), NoV)) + NoV * length(vec3f(aWT * dot(Tw, L), aWB * dot(Bw, L), NoL)), 1e-5);
      let dF = aFT * aFB; let vF = vec3f(aFB * dot(Tf, H), aFT * dot(Bf, H), dF * NoH); let w2F = dF / max(dot(vF, vF), 1e-8);
      let DF = dF * w2F * w2F / PI;
      let GF = 0.5 / max(NoL * length(vec3f(aFT * dot(Tf, V), aFB * dot(Bf, V), NoV)) + NoV * length(vec3f(aFT * dot(Tf, L), aFB * dot(Bf, L), NoL)), 1e-5);
      let spec = warpWeight * DW * GW * fresnel(f0Warp, VoH) + weftWeight * DF * GF * fresnel(f0Weft, VoH);
      // Charlie sheen
      let inv = 1.0 / sheenAlpha;
      let sin2 = max(1.0 - NoH * NoH, 7.8e-3);
      let DS = (2.0 + inv) * pow(sin2, inv * 0.5) / (2.0 * PI);
      let VS = 1.0 / max(4.0 * (NoL + NoV - NoL * NoV), 1e-3);
      direct += (diffuse + spec * sheenScale + sheenColor * DS * VS) * radiance * NoL * vis;
    } else if (NoLs < 0.0) {
      var vis = 1.0;
      if (li == 0) { vis = shadowVisibility(vin.world, L, vin.clip.xy); }
      let back = max(-dot(N, L), 0.0);
      direct += transmission * radiance * (back * 0.8 + 0.2) / PI * vis;
    }
  }
  let R = reflect(-V, N);
  var ambient = albedo * envRadiance(N, 1.0) * sheenScale;
  ambient += transmission * envRadiance(-N, 1.0) * 0.8;
  let roughMean = sqrt(sqrt(aWT * aWB) * warpWeight + sqrt(aFT * aFB) * weftWeight);
  let f0Mean = f0Warp * warpWeight + f0Weft * weftWeight;
  ambient += envRadiance(R, roughMean) * (f0Mean + (vec3f(1.0) - f0Mean) * pow(1.0 - NoV, 5.0) * (1.0 - roughMean)) * sheenScale;
  ambient += sheenColor * envRadiance(N, 0.7) * pow(1.0 - NoV, 2.0) * 1.2;

  var alpha = select(1.0, coverage * m.trans.a, m.trans.a < 0.999 || weave == 10) * cutAlpha;
  if (m.trans.a < 0.999 || weave == 10) {
    let grazing = pow(1.0 - clamp(abs(dot(N0, V)), 0.0, 1.0), 2.5);
    alpha = mix(alpha, cutAlpha, grazing);
  }
  return vec4f(finish(direct + ambient, vin.clip.xy), clamp(alpha, 0.0, 1.0));
}
`;

// ---------------------------------------------------------------- colliders, floor and sky

export const MESH = /* wgsl */ `
struct MOut {
  @builtin(position) clip: vec4f,
  @location(0) world: vec3f,
  @location(1) normal: vec3f,
};

@vertex fn vs_mesh(@location(0) p: vec3f, @location(1) n: vec3f) -> MOut {
  var o: MOut;
  let w = draw.model * vec4f(p, 1.0);
  o.world = w.xyz;
  o.normal = (draw.model * vec4f(n, 0.0)).xyz;
  o.clip = view.viewProj * w;
  return o;
}

@vertex fn vs_mesh_shadow(@location(0) p: vec3f, @location(1) n: vec3f) -> @builtin(position) vec4f {
  return view.shadowVP * (draw.model * vec4f(p, 1.0));
}

@fragment fn fs_mesh(vin: MOut) -> @location(0) vec4f {
  let isFloor = draw.params.y > 0.5;
  var albedo = draw.color.rgb;
  var rough = draw.color.a;
  let metallic = draw.params.x;
  if (isFloor) { albedo = view.floorCol.rgb; rough = view.floorCol.a; }
  let N = normalize(vin.normal);
  let V = normalize(view.camPos.xyz - vin.world);
  let NoV = max(dot(N, V), 1e-3);
  let a = max(rough * rough, 0.02);
  let f0 = mix(vec3f(0.04), albedo, metallic);
  let diffuseColor = albedo * (1.0 - metallic);
  var direct = vec3f(0.0);
  for (var li = 0; li < 2; li++) {
    let L = select(view.l1Dir.xyz, view.l0Dir.xyz, li == 0);
    let radiance = select(view.l1Col.rgb, view.l0Col.rgb, li == 0);
    let NoL = dot(N, L);
    if (NoL <= 0.0) { continue; }
    var vis = 1.0;
    if (li == 0) { vis = shadowVisibility(vin.world, N, vin.clip.xy); }
    let H = normalize(V + L);
    let spec = distributionGgx(max(dot(N, H), 0.0), a) * visibilitySmith(NoV, NoL, a) * fresnel(f0, max(dot(V, H), 0.0));
    direct += (diffuseColor / PI + spec) * radiance * NoL * vis;
  }
  let R = reflect(-V, N);
  var color = direct + diffuseColor * envRadiance(N, 1.0) + envRadiance(R, rough) * (f0 + (vec3f(1.0) - f0) * pow(1.0 - NoV, 5.0) * (1.0 - rough));
  if (isFloor) {
    let dist = length(vin.world.xz - view.camPos.xz);
    let fade = smoothstep(view.floorFade.x, view.floorFade.y, dist);
    color = mix(color, background(-V), fade);
  }
  return vec4f(finish(color, vin.clip.xy), 1.0);
}

struct SOut { @builtin(position) clip: vec4f, @location(0) ndc: vec2f };

@vertex fn vs_sky(@builtin(vertex_index) i: u32) -> SOut {
  var o: SOut;
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0;
  o.clip = vec4f(p, 1.0, 1.0);
  o.ndc = p;
  return o;
}

@fragment fn fs_sky(vin: SOut) -> @location(0) vec4f {
  let far = view.invViewProj * vec4f(vin.ndc, 1.0, 1.0);
  let dir = normalize(far.xyz / far.w - view.camPos.xyz);
  return vec4f(finish(background(dir), vin.clip.xy), 1.0);
}
`;
