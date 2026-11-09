# WebGPU Cloth

Real-time cloth simulation and rendering in the browser, with all of the physics on the GPU in WebGPU compute shaders.

**Live:** https://netolcc06.github.io/webgpu-cloth/

Pull the cloth with the mouse, switch between eight scenes and 19 fabrics, and change the mesh resolution from 64 × 64 to 256 × 256.

## Requirements

A browser with WebGPU: Chrome or Edge 113 or later, or Safari 26. On Linux, Chrome may need WebGPU enabled in `chrome://flags`. A discrete GPU is recommended for resolutions above 128 × 128.

## Controls

| Input | Action |
|---|---|
| Left drag | Pull the cloth (dragging empty space orbits the camera) |
| Right drag or Ctrl + drag | Orbit the camera |
| Shift + drag | Pan |
| Wheel | Zoom |
| Space | Pause |
| R | Reset |

The URL accepts `scene` (0 to 7), `fabric` (0 to 18) and `res`, for example `?scene=4&res=192`.

## Physics

The solver is small-step XPBD (Macklin et al. 2019): every substep of 1/1440 s makes a single pass over the constraints, and each frame runs as many substeps as fit in the time since the last one, so the motion follows the display at any frame rate.

- **Membrane:** strain-based triangle constraints (Mueller et al. 2014), with separate compliance for warp, weft and shear, softened under compression so that fabric buckles.
- **Bending:** the isometric model of Bergou et al. (2006) as a linear vector constraint, exact in XPBD and free of singularities near the flat state.
- **Parallel Gauss-Seidel:** constraints are coloured greedily so that each colour runs as one dispatch with no write conflicts.
- **Strain limiting:** one-sided long range edges along the yarns, and long range attachments to pins (Kim et al. 2012), so that high resolution cloth does not look like rubber.
- **Colliders:** analytic signed distance fields, with Coulomb and static friction relative to moving surfaces.
- **Cloth-cloth collision:** vertex against triangle through a spatial hash. Each particle remembers which side of each nearby triangle it belongs to, contacts are summed with fixed point atomics and normalised by their Jacobi load. Candidate lists are only rebuilt when something moved, through indirect dispatches decided on the GPU.
- **Aerodynamics:** pressure and skin drag over a wide stencil normal, in a turbulent wind field.
- **Stability:** a weak membrane viscosity removes the mesh scale oscillations of a single pass per substep, which would otherwise defeat static friction.

Collision reduces crossings but does not prevent all of them: there is no continuous collision detection and no edge-edge test.

## Rendering

Forward rendering with 4x MSAA and alpha to coverage for real holes in sheer fabrics.

- Two anisotropic GGX lobes, one per yarn set, with the visible share of warp and weft shifting with the view direction.
- Charlie sheen (Estevez and Kulla 2017), Oren-Nayar diffuse and diffuse transmission.
- A procedural weave evaluated per pixel from loom drafts (plain, twill, satin, herringbone, basket, Oxford, knit, tulle, pile), faded to its average where a pixel covers many threads.
- Thread colour orders for tartan, houndstooth, gingham and stripes, a damask motif that swaps the weave face, a floral print and lace.
- Percentage-closer soft shadows from the key light, an analytic studio or daylight environment and the AgX tone curve.

## Running locally

There is no build step. Serve the folder with any static file server and open it, for example:

```
python -m http.server 8000
```

then open http://localhost:8000.

## Code

| File | Contents |
|---|---|
| `js/simShaders.js` | WGSL compute shaders of the solver |
| `js/sim.js` | GPU buffers, pipelines and the recording of a frame of simulation |
| `js/mesh.js` | Cloth meshes, constraints and graph colouring, on the CPU |
| `js/renderShaders.js` | WGSL of the renderer |
| `js/render.js` | Shadow and forward passes, collider meshes |
| `js/scenes.js` | The eight scenes |
| `js/fabrics.js` | The 19 fabrics: mechanics in SI units and appearance |
| `js/main.js` | WebGPU setup, input, interface and frame loop |

## Known limitations

- Meshes are rectangular or tubular grids, with an optional circular cut.
- Crossings between cloths are reduced, not prevented; heaps of crumpled cloth keep some.
- Atomic operations make runs non-deterministic.
- The weave is procedural and simpler than a baked, mip-filtered texture: very fine patterns can shimmer when the camera moves.
- There is no screen-space ambient occlusion and no shell rendering for pile; velvet relies on sheen alone.
