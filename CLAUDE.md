# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Implementation of the **Reachable Polyhedral Marching (RPM)** algorithm for exact forward and backward reachability of fully connected ReLU neural networks. Companion code for two papers (IEEE 2021 conference + arXiv:2210.08339 journal). Given a ReLU network and an input polytope, RPM enumerates the network's piecewise-affine (PWA) cells and propagates polytopic sets through them.

Primarily Julia (1.10.4); a few MATLAB scripts use the [MPT3](https://www.mpt3.org/) toolbox (3.2.1) for invariant-set computations that RPM does not perform itself. The `pytorch_.py` / `requirements.txt` venv is only needed to retrain the example networks — analysis runs purely in Julia.

The `paper/` directory holds the LaTeX source of the journal paper (arXiv:2210.08339) that this repo accompanies. Mathematical derivations and notation conventions for the algorithm live there — consult it when the code references results, lemmas, or symbols whose definitions aren't obvious from the source alone.

## Running

This project has no test suite or build step. The "tests" are the per-example scripts at the repo root, run from the Julia REPL:

```julia
julia                       # in the repo root
julia> ]                    # enter pkg mode
(@v1.10) pkg> activate .
(Neural-Network-Reach) pkg> instantiate    # one-time
julia> include("random.jl")           # or pendulum.jl, acas.jl, quadratic.jl,
                                      # vanderpol_roa.jl, pendulum_controlled.jl,
                                      # taxinet_roa.jl, pwa_back_reach.jl, invariance.jl
```

Always `include` from the REPL rather than `julia file.jl` from the shell — every example loads heavy plotting/optimization stacks (Plots, JuMP, GLPK, LazySets, Polyhedra, CDDLib) and benefits from REPL warm-up; many scripts also leave plot handles in scope intentionally.

There is no lint/format command and no CI. `@time begin ... end` blocks around `compute_reach` calls are the only built-in benchmarking.

## Architecture

### The core abstraction: activation patterns (APs)

The whole codebase is organized around an **activation pattern** = `Vector{BitVector}`, one BitVector per hidden layer indicating which ReLU neurons are active. Each unique AP corresponds to one affine cell of the PWA function the network represents. Almost every dictionary key in this codebase is an AP. When you see `ap2input`, `ap2output`, `ap2map`, `ap2backward`, `ap2neighbors` — these are all keyed by the same AP type and represent different views of the same cell:

- `ap2input :: Dict{Vector{BitVector}, (A, b)}` — H-rep of the cell in input space
- `ap2output :: Dict{Vector{BitVector}, (A', b')}` — H-rep of the forward image
- `ap2map :: Dict{Vector{BitVector}, (C, d)}` — local affine map `y = Cx + d` valid inside the cell
- `ap2backward :: Vector{Dict{...}}` — one entry per output set; each is a Dict from AP to the preimage polytope (BRS slice)
- `ap2neighbors :: Dict{Vector{BitVector}, Vector{Vector{BitVector}}}` — adjacency graph over cells

All weight matrices in this codebase are augmented by an extra column for the bias and an extra row that propagates a constant `1` (see `nnet_load`, `pytorch_net`, `chain_net`). Layer maps are therefore plain matrix multiplies on `[x; 1]`. `local_map` strips the last column to recover `(C, d)`.

### `reach.jl` — the algorithm

`reach.jl` is the heart of the project (~750 lines). The two public entry points:

- `compute_reach(weights, Aᵢ, bᵢ, Aₒ, bₒ; fp, reach, back, connected, graph, check_aps)` — main marcher. `Aₒ`/`bₒ` are **vectors of** matrices/vectors so a single run can compute BRSs of multiple unsafe sets simultaneously. Flags: `reach=true` for forward image, `back=true` for backward reachable set, `connected=true` to restrict marching to the connected component containing `fp` (used for ROA computations), `graph=true` to also return `ap2neighbors`.
- `verify_safety(weights, Aᵢ, bᵢ, Aₒ, bₒ)` — early-terminating cell enumeration that returns the first unsafe AP per output set instead of computing reachable sets.

The marcher works by: pick a seed input → compute its AP → for that cell, build constraints from the active neurons (`get_constraints`), prune redundant ones (`remove_redundant` does a bounding-box LP heuristic first, then per-constraint LPs in `exact_lp_remove`/`exact_lp_remove_feas`), then "step over" each non-redundant facet to enumerate neighbor APs (`add_neighbor_aps`). The numerical tolerance `ϵ = 1e-15` defined at the top of `reach.jl` is used everywhere; LPs are solved with GLPK via JuMP.

`poly_intersection` lives in `pwa_back_reach.jl` but is used inside `compute_reach`/`verify_safety`, so don't `include` `reach.jl` alone if you call those — most example scripts include both indirectly.

### Loading networks (`load_networks.jl`)

Three loaders produce the augmented-weights format the rest of the code expects:

- `nnet_load(path)` / `acas_net_nnet(a, b)` — Stanford `.nnet` files (uses `nnet.jl`, vendored from the Stanford NNet project).
- `pendulum_net(path, copies)` — MATLAB `.mat` with `X_mean/X_std/Y_mean/Y_std/weights/biases`.
- `pytorch_net(weights.npz, norm_params.npz, copies)` — exported from `pytorch_.py`.
- `taxinet_cl(copies)` / `taxinet_2input_resid()` — manually stitch the Stanford taxinet perception net to the dynamics net to form a closed-loop network.

`copies` is the multi-step composition count: `chain_net` glues `copies` copies of a 1-step network together so a single `compute_reach` call analyzes `t`-step dynamics. `chain_net` accounts for the augmented `[x;1]` row when wiring the output of one copy to the input of the next.

### Example scripts ↔ paper figures

Each script at the repo root implements one experiment. They share a stereotyped layout that is worth knowing when adding a new one:

1. `input_constraints_<name>(...)` returns `(Aᵢ, bᵢ)`.
2. `output_constraints_<name>(...)` returns `(Aₒ, bₒ)` — note the comment in `pendulum.jl` about un-normalizing: output constraints written against `yₒᵤₜ = Aₒᵤₜy + bₒᵤₜ` must be transformed if you want them on raw network outputs.
3. `plot_hrep_<name>(...)` plots a dict of polytopes.
4. A "scripting" block at the bottom that wires those together and calls `compute_reach`.

`invariance.jl` is the ROA toolkit shared across `vanderpol_roa.jl`, `pendulum_controlled.jl`, `taxinet_roa.jl`: fixed-point search (`find_fixed_points`, `find_attractor`), homeomorphism check (`is_homeomorphism`), local Lyapunov via `lyapd`, and the seed-polytope SDP (`polytope_roa_sdp` → `solve_sdp_jump` with COSMO). `find_roa(...)` is a one-call pipeline that runs RPM, finds an attractor, solves the seed SDP, and does backward reachability.

`pwa_back_reach.jl` is a separate, faster backward-reachability loop that **assumes the PWA function is a homeomorphism** and uses a precomputed `ap2neighbors` graph rather than re-marching the network. It loads `pwa_dict["ap2map"/"ap2input"/"ap2neighbors"]` from a saved JLD2. This is what was used for the multi-step taxinet BRSs in the journal paper. Keep this distinction in mind: `compute_reach(..., back=true)` re-derives cells from network weights; `pwa_back_reach.jl` walks an already-enumerated PWA representation.

### Julia ↔ MATLAB bridge

Some experiments hand off to MATLAB for invariant sets / Lyapunov work that MPT3 does well:

- `merge_poly.jl` ⇄ `merge_poly.m` — convert polytope sets between `.jld2` and `.mat`.
- `mpt_invariant.jl` exports a saved `pwa_dict` (`.jld2`) to a `.mat` for MPT.
- `mat_invariant_pendulum.m`, `mat_invariant_vanderpol.m`, `mat_roa.m`, `plot_3D.m` — MATLAB-side scripts (run MATLAB as Administrator per the README).

The `MAT`, `MATLAB`, and `NPZ` Julia packages are the bridges; `MATLAB.jl` requires a working MATLAB install.

### Vendored / unusual code

- `nnet.jl` is a vendored Stanford NNet reader; treat it as upstream code.
- `unique_custom.jl` is a copy of `Base.unique` modified to return the unique-row index map. Used by `get_constraints` to dedupe redundant hyperplanes from the AP. Don't replace it with `unique` — the index mapping is load-bearing.

## Things to know before editing

- **Don't add a test suite**: there is no `test/` directory and no `Project.toml` `[targets]`. Examples are the regression suite — if you change `reach.jl`, re-run `random.jl`, `pendulum.jl`, and `quadratic.jl` from the REPL and eyeball the plot/`length(ap2input)` output.
- **`Manifest.toml` is checked in** and pinned to Julia 1.10.4. Don't `Pkg.update()` casually — the SDP/LP solver stack (COSMO, SCS, Convex, JuMP, MathOptInterface) is sensitive to version drift.
- **Large model artifacts are gitignored**: `models/taxinet/taxinet_pwa_map_5_15.jld2` and `models/vanderpol/BRSs.jld2` are not in the repo; scripts that load them will fail until those files are regenerated locally.
- **`error_states/` is referenced but not committed** (`save("error_states/latest_error_ap.jld2", ...)` in `exact_lp_remove`). If an LP fails on bad geometry, mkdir it first or the error path will throw a different error.
- **Unicode identifiers everywhere**: `Aᵢ, bᵢ, Aₒ, bₒ, ϵ, σᵢ, μᵢ`, etc. Match the existing convention rather than ASCII-izing.
