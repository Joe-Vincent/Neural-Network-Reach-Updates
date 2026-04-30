# Fixed points of a diffusion denoiser via RPM

Companion experiment to the half-moon diffusion fixed-point study. The original
plan lives at `~/.claude/plans/in-my-paper-see-cuddly-dove.md`.

## Research question

Do diffusion models learn fixed points that approximately correspond to training
data samples?

This frames diffusion as **projection onto the data manifold**: a denoiser
applied to a clean point should leave it (nearly) unchanged. If that is true,
points on the manifold are fixed points of the denoiser map. Concretely, for a
trained denoiser `f(x, t)` and a low analysis time `t`, do the fixed points
`{x : f(x, t) = x}` cluster on the training manifold?

The 2D half-moon distribution is small enough that RPM
(`compute_reach` in `reach.jl`) can enumerate every PWA cell of the network
exactly, then `find_fixed_points_robust` solves `(I − C) x = d` per cell. No
gradient descent, no sampling — every fixed point in the input box is found.

## Setup

### Predict-`x_0` parameterization

The network is trained with the predict-`x_0` parameterization: the target of
the regression is the clean sample, not the noise. Both predict-noise and
predict-`x_0` are standard DDPM variants and learn the same data distribution
(they are linearly related: `x_0 = (x_t − σ(t) ε) / α(t)`).

We pick predict-`x_0` because it makes fixed-point semantics align with the
research question:

- predict-`x_0`: output is in input space. `f(x, t) = x` literally means "the
  network thinks `x` is already clean". That is the manifold-projection claim.
- predict-noise: output is noise. A fixed point of the raw network has no clean
  interpretation. We would have to wrap the network as `g(x) = x − σ(t) ε̂(x,t)`
  before feeding it to RPM — workable, but it adds an output-side affine layer
  and offers no advantage for this experiment.

### Freeze-`t` trick

RPM analyzes plain MLPs `R^d → R^d`. A time-conditioned denoiser is
`R^(d+1) → R^d` because of the time input. `freeze_time_input(weights, t_value)`
folds a fixed `t` into the first-layer bias and drops the time column, producing
a pure `R^2 → R^2` augmented network that RPM can ingest without modification.
The math is detailed at the top of `diffusion_helpers.jl`.

### `chain_varied` for composed steps

DDPM sampling uses a *decreasing* `t` schedule; each step has its own frozen
network. `chain_varied(weights_list)` extends the existing `chain_net` (which
chains identical copies, `load_networks.jl:84-105`) to allow a different weight
matrix per copy. The composed network can then be passed to `compute_reach`
unchanged. Both helpers come with eval-net round-trip checks at the top of the
caller scripts so a bug in the surgery cannot silently corrupt downstream
results.

## Findings

Inter-data baseline (median nearest-neighbor distance among 8000 training
samples): **0.0058**. This is the spatial scale below which "close to data" is
a non-trivial claim.

| Setting                                | Cells | Stable fps | Unstable fps | Stable-fp → data (median) | Stable-fp → data (max) | % stable fps within 0.05 of data | % data within 0.10 of a stable fp |
|----------------------------------------|------:|-----------:|-------------:|--------------------------:|----------------------:|---------------------------------:|----------------------------------:|
| One-step at `t = 0.05`                 |  1586 |          5 |            6 |                    0.0037 |                0.0094 |                             100% |                             12.7% |
| 3-step composed `t = [0.5, 0.3, 0.1]`  |  3457 |          1 |            0 |                    0.2798 |                0.2798 |                               0% |                              0.0% |

**One-step interpretation.** Every stable fixed point sits within `~1.6×` the
inter-data nearest-neighbor scale of an actual training sample — they sit *on*
the half-moon manifold. The 6 unstable fixed points sit between the two moons
and at the moon-arm boundaries, acting as saddles separating the basins of the
two arms. The manifold-projection hypothesis holds for the one-step denoiser.

**3-step composed interpretation (the surprise).** Cell-wise fixed-point
analysis collapses to a single attractor at `≈(0, 0)` — the centroid of the
data, well off the manifold. The 9 fixed points seen in an earlier
under-trained model were largely artifacts of network roughness; the better-
trained network is smoother (cells `7017 → 3457`, ~50% fewer linear pieces) and
its cell-local maps near the data are very close to the identity. With `C ≈ I`
the linear system `(I − C) x = d` has a near-singular kernel; what solutions do
exist generically land *outside* their own cell and get filtered (3456 of 3457
candidates are filtered for this reason). The single surviving fp is the global
attractor of the high-t composed contraction — at `t = 0.5` the network's
optimal `E[x_0 | x_t]` for high-noise inputs is the data mean, and three
compositions pull everything toward it.

This means cell-wise exact fp analysis becomes a *less* sensitive probe of the
manifold-projection structure as composition makes the map smoother and more
nearly identity on the data. It does not mean the data manifold isn't
approximately invariant under the composed map — only that "approximately
invariant" doesn't show up as cellwise fixed points. Verifying invariance
directly (iterating `x_{n+1} = f(x_n)` from clean points and checking
displacement) is on the next-steps list below.

The "% data within 0.10 of a stable fp" column is intentionally not 100% in
the one-step case: 5 stable fps cannot cover thousands of data samples evenly.
What matters is the converse — no stable fp drifts off the manifold.

### Plots

- `fps_onestep.png` — light-grey PWA cell boundaries from the one-step
  denoiser at `t = 0.05`, blue training points, green stable fixed points
  on the moon arms, red unstable fixed points between the moons.
- `fps_kstep.png` — same overlay for the 3-step composed network. A single
  green fp at the data centroid; no unstable fps survive the cell filter.

Both plots are gitignored (regenerable). Run `diffusion_replot.jl` to render
them from the saved `.jld2` without re-running RPM.

## How to reproduce

All commands run from the repo root with the project active. The scripts live
at the repo root next to `quadratic.jl`, `pendulum.jl`, etc.

```julia
julia --project=.

# Phase 1 — train the denoiser. Python; uses the venv at ./venv.
# (Skip if weights.npz, norm_params.npz, training_data.npy already exist.)
shell> source venv/bin/activate && python diffusion_train.py

# Phase 2 — one-step analysis at t = 0.05
julia> include("diffusion_fixed_points.jl")
# produces: fps_onestep.jld2, fps_onestep.png

# Phase 3 — 3-step composed analysis at t = [0.5, 0.3, 0.1]
julia> include("diffusion_multistep.jl")
# produces: fps_kstep.jld2, fps_kstep.png

# Phase 4 — nearest-neighbor stats
julia> include("diffusion_compare.jl")
# prints the inter-data baseline + per-stage tables to stdout

# Utility — re-render plots from saved jld2 without re-running compute_reach
julia> include("diffusion_replot.jl")
```

The Julia scripts each take a few seconds to a few minutes; the 3-step
`compute_reach` is the slowest (cell count is ~3.5k for the current model,
~30s on this machine). Nothing here needs MATLAB.

### Files

Tracked (small, reproducibility-pinning):
- `weights.npz`, `norm_params.npz` — trained denoiser, in the augmented format
  `pytorch_net` (`load_networks.jl:144-177`) expects.
- `training_data.npy` — the half-moon samples used at train time. Pinned so the
  comparison stats are reproducible.

Gitignored (derived from the above):
- `fps_onestep.{jld2,png}`, `fps_kstep.{jld2,png}` — RPM output and plots.
- `samples_sanity.png` — Phase-1 reverse-process check from `diffusion_train.py`.

## Potential next steps

- **Sweep `t_low`** for the one-step analysis and plot `median(stable-fp → data)`
  vs `t_low`. Expected curve: distance grows as `t_low → 0.5` (denoiser
  output regresses toward the data mean) and shrinks toward zero as
  `t_low → 0` (denoiser becomes the identity). The "elbow" of that curve is the
  effective analysis time.
- **Iterated dynamics check** (priority: high after the composed-case
  surprise). Two questions to answer empirically: (1) from random starts
  inside the box, does `x_{n+1} = f(x_n)` for the one-step network converge to
  the 5 stable cell-wise fps and only to them? Confirms the eigenvalue-based
  stability classification matches actual basin behavior. (2) For the composed
  3-step network, does iteration from clean data points stay near the data?
  If yes, the data manifold is approximately invariant even though no cell-wise
  fp lies on it — meaning cell-wise fp analysis underestimates manifold
  invariance for smooth composed maps. If no, the composed map really does
  pull data toward the centroid, and the high-t step is too aggressive.
- **Predict-noise vs predict-`x_0` side-by-side.** Train the same architecture
  with predict-noise, wrap as `g(x) = x − σ(t) ε̂(x, t)` (an output-side affine),
  feed *that* to RPM. Same fixed-point set if the parameterizations are
  equivalent, but a useful regression test on the wrap math.
- **Harder distributions.** 3-component Gaussian mixture or swiss roll. Tests
  whether the stable-fp-on-manifold pattern survives non-trivial topology.
- **Width / depth sweep.** Vary network size and track fixed-point count and
  cell count. Connects fixed-point geometry to the network's expressive
  capacity.

## Files in this experiment

Code (at repo root, mirrors `quadratic.jl` / `pendulum.jl` convention):

| File                          | Role                                                                 |
|-------------------------------|----------------------------------------------------------------------|
| `diffusion_train.py`          | Phase 1: train the predict-`x_0` MLP `[3,16,16,16,16,16,2]`          |
| `diffusion_helpers.jl`        | Shared: `freeze_time_input`, `chain_varied`, `find_fixed_points_robust`, `plot_diffusion_result` |
| `diffusion_fixed_points.jl`   | Phase 2: one-step analysis at `t = 0.05`                             |
| `diffusion_multistep.jl`      | Phase 3: 3-step composed analysis                                    |
| `diffusion_compare.jl`        | Phase 4: nearest-neighbor stats                                      |
| `diffusion_replot.jl`         | Utility: re-render plots from saved `.jld2`                          |

Inputs (this directory, tracked): `weights.npz`, `norm_params.npz`,
`training_data.npy`.

Outputs (this directory, gitignored): `fps_onestep.{jld2,png}`,
`fps_kstep.{jld2,png}`, `samples_sanity.png`.
