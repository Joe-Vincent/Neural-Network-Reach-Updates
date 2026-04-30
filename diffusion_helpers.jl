# Shared helpers for the diffusion fixed-point experiments.
# Included by diffusion_fixed_points.jl (one-step) and diffusion_multistep.jl (composed).
# Assumes reach.jl and invariance.jl are already included by the caller.

using Plots, LinearAlgebra


### CONSTRAINTS ###
function input_constraints_diffusion(type::String)
    # x ∈ R^2 (the t input is frozen separately by freeze_time_input)
    if type == "box"
        in_dim = 2
        A_pos = Matrix{Float64}(I, in_dim, in_dim)
        A_neg = Matrix{Float64}(-I, in_dim, in_dim)
        A = vcat(A_pos, A_neg)
        b = [2.5, 2.5, 2.5, 2.5]   # half-moons live in ~[-1.65, 1.6]^2; this gives margin
    else
        error("Invalid input constraint specification: $type")
    end
    return A, b
end


function output_constraints_diffusion(type::String)
    # Dummy: we don't use back-reachability here, but compute_reach requires this argument.
    if type == "dummy"
        A = [1. 0.; -1. 0.; 0. 1.; 0. -1.]
        b = [100., 100., 100., 100.]
    else
        error("Invalid output constraint specification: $type")
    end
    return A, b
end


### NETWORK SURGERY ###

# Fold a fixed value of one input column into the first-layer bias and drop that column.
#
# pytorch_net returns weights[1] in augmented form: shape (h+1, in_dim+1) where columns
# 1..in_dim are input feature columns, column in_dim+1 is the bias, and the bottom row
# is the propagator [0, ..., 0, 1] for the constant-1 channel.
#
# For our network in_dim = 3, t is at column index 3, so we add t_value * weights[1][:, 3]
# into the bias column (last column) and drop column 3. The propagator row's t column is
# already 0, so dropping it leaves [0, 0, 1] which is still a valid propagator for the
# new (in_dim=2) augmented input [x; 1].
function freeze_time_input(weights, t_value; t_col_idx::Int=3)
    new_weights = [copy(w) for w in weights]
    W1 = new_weights[1]
    new_W1 = copy(W1)
    new_W1[:, end] = new_W1[:, end] + t_value * new_W1[:, t_col_idx]
    keep_cols = [j for j in 1:size(new_W1, 2) if j != t_col_idx]
    new_W1 = new_W1[:, keep_cols]
    new_weights[1] = new_W1
    return new_weights
end


# Chain K weight vectors of equal length, each shaped R^d → R^d, into a single composed
# network. Generalizes chain_net (load_networks.jl:84-105) which assumes all K copies are
# identical. For our use case, each copy is the same base network with a different value
# of t frozen in via freeze_time_input, matching how DDPM sampling uses a decreasing t
# schedule.
function chain_varied(weights_list)
    K = length(weights_list)
    @assert K >= 1 "Need at least one set of weights"
    num_layers = length(weights_list[1])
    @assert all(length(w) == num_layers for w in weights_list) "All weight vectors must have same length"

    total_len = K * num_layers - (K - 1)
    weights = Vector{Matrix{Float64}}(undef, total_len)
    # Junctions are the global layer indices that merge copy c's last layer with copy c+1's
    # first layer. Mirrors `merged_layers` in chain_net but excludes the trailing one
    # (which is just the final output layer of the last copy).
    junction_indices = [c * num_layers - (c - 1) for c in 1:(K - 1)]

    copy_idx = 1
    w_idx = 1
    for k in 1:total_len
        if k == 1
            weights[k] = weights_list[1][1]
            w_idx = 2
        elseif k == total_len
            weights[k] = weights_list[K][end]
        elseif k in junction_indices
            w_prev = weights_list[copy_idx]
            w_next = weights_list[copy_idx + 1]
            w̄ₒ = vcat(w_prev[end], reshape(zeros(size(w_prev[end], 2)), 1, :))
            w̄ₒ[end, end] = 1
            weights[k] = w_next[1] * w̄ₒ
            copy_idx += 1
            w_idx = 2
        else
            weights[k] = weights_list[copy_idx][w_idx]
            w_idx += 1
        end
    end
    return weights
end


### FIXED POINTS ###

# Like find_fixed_points (invariance.jl:31-56) but skips cells where (I - C) is rank-
# deficient or near-singular instead of erroring. For diffusion denoisers near low t,
# many cells have local maps very close to identity; (I - C) is then near-singular and
# the original function would throw "Non-unique fixed point!".
function find_fixed_points_robust(state2map, state2input, weights; tol::Float64=1e-8)
    dim = size(weights[1], 2) - 1
    fixed_points = Vector{Vector{Float64}}(undef, 0)
    fp_dict = Dict{Vector{Float64}, Vector{Tuple{Matrix{Float64},Vector{Float64}}}}()
    n_skipped_singular = 0
    n_outside_polytope = 0
    n_failed_sanity = 0
    for ap in keys(state2map)
        C, d = state2map[ap]
        A, b = state2input[ap]
        M = Matrix{Float64}(I, dim, dim) - C
        s = svdvals(M)
        if s[end] < tol * max(s[1], one(eltype(s)))
            n_skipped_singular += 1
            continue
        end
        fp = M \ d
        if !in_polytope(fp, A, b)
            n_outside_polytope += 1
            continue
        end
        if !(eval_net(fp, weights, 1) ≈ fp)
            n_failed_sanity += 1
            continue
        end
        push!(fixed_points, fp)
        fp_dict[fp] = [(A, b), (C, d)]
    end
    println("find_fixed_points_robust: kept $(length(fixed_points)) fixed points")
    println("  skipped $n_skipped_singular cells with near-singular (I − C)")
    println("  $n_outside_polytope candidates fell outside their cell")
    println("  $n_failed_sanity candidates failed eval_net sanity check")
    return fixed_points, fp_dict
end


# Classify a fixed point as stable if all eigenvalues of its cell's C have |λ| < 1.
function is_stable(fp, fp_dict)
    C, _ = fp_dict[fp][2]
    return all(norm.(eigen(C).values) .< 1)
end


### PLOTTING ###
function plot_diffusion_result(ap2input, fps, fp_dict, training_data;
                                xlims=(-2.5, 2.5), ylims=(-2.5, 2.5), title="")
    plt = plot(reuse=false, legend=:topright, aspect_ratio=:equal,
               xlims=xlims, ylims=ylims, xlabel="x₁", ylabel="x₂", title=title)
    # Plot cell boundaries directly to avoid Plots/LazySets default vertex markers.
    # We use Polyhedra.jl to compute vertices, then close the polygon manually.
    for ap in keys(ap2input)
        A, b = ap2input[ap]
        reg = HPolytope(constraints_list(A, b))
        if isempty(reg); continue; end
        verts = vertices_list(reg)
        if length(verts) < 2; continue; end
        # Order vertices by polar angle around centroid so we get a clean polygon
        cx = sum(v[1] for v in verts) / length(verts)
        cy = sum(v[2] for v in verts) / length(verts)
        sorted_verts = sort(verts, by = v -> atan(v[2] - cy, v[1] - cx))
        xs = [v[1] for v in sorted_verts]; push!(xs, sorted_verts[1][1])
        ys = [v[2] for v in sorted_verts]; push!(ys, sorted_verts[1][2])
        plot!(plt, xs, ys, linecolor=:lightgray, linewidth=0.3,
              markershape=:none, label=false)
    end
    scatter!(plt, training_data[:, 1], training_data[:, 2],
             markersize=2, markeralpha=0.5, markercolor=:steelblue,
             markerstrokewidth=0, label="training data")
    stable_fps = [fp for fp in fps if is_stable(fp, fp_dict)]
    unstable_fps = [fp for fp in fps if !is_stable(fp, fp_dict)]
    if !isempty(stable_fps)
        S = reduce(hcat, stable_fps)
        scatter!(plt, S[1, :], S[2, :],
                 markersize=6, markercolor=:green, markershape=:circle,
                 markerstrokewidth=0.5, label="stable fp")
    end
    if !isempty(unstable_fps)
        U = reduce(hcat, unstable_fps)
        scatter!(plt, U[1, :], U[2, :],
                 markersize=6, markercolor=:red, markershape=:diamond,
                 markerstrokewidth=0.5, label="unstable fp")
    end
    return plt
end
