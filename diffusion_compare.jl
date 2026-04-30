using FileIO, JLD2, NPZ, LinearAlgebra, Statistics, Printf
include("reach.jl")
include("invariance.jl")
include("diffusion_helpers.jl")


# Pairwise distance helpers
function pairwise_min_distance(query_pts, ref_pts)
    """For each query point, return the distance to its nearest neighbor among ref_pts."""
    n_query = size(query_pts, 1)
    out = Vector{Float64}(undef, n_query)
    for i in 1:n_query
        q = view(query_pts, i, :)
        best = Inf
        for j in 1:size(ref_pts, 1)
            r = view(ref_pts, j, :)
            d = norm(q .- r)
            if d < best
                best = d
            end
        end
        out[i] = best
    end
    return out
end


function inter_data_nn_distance(data)
    """Mean nearest-neighbor distance within a point set (excluding self-matches)."""
    n = size(data, 1)
    out = Vector{Float64}(undef, n)
    for i in 1:n
        best = Inf
        for j in 1:n
            i == j && continue
            d = norm(view(data, i, :) .- view(data, j, :))
            if d < best
                best = d
            end
        end
        out[i] = best
    end
    return out
end


function summarize(label, fps, fp_dict, training_data)
    if isempty(fps)
        println("\n=== $label ===")
        println("  no fixed points found.")
        return
    end
    fp_mat = reduce(hcat, fps)'   # (n_fp, 2)
    stable_flags = [is_stable(fp, fp_dict) for fp in fps]
    n_stable = sum(stable_flags)
    n_unstable = length(fps) - n_stable

    fp_to_data = pairwise_min_distance(fp_mat, training_data)
    data_to_fp = pairwise_min_distance(training_data, fp_mat)

    if n_stable > 0
        stable_fp_mat = reduce(hcat, fps[stable_flags])'
        stable_to_data = pairwise_min_distance(stable_fp_mat, training_data)
        data_to_stable = pairwise_min_distance(training_data, stable_fp_mat)
    else
        stable_to_data = Float64[]
        data_to_stable = Float64[]
    end

    println("\n=== $label ===")
    @printf("  fixed points: %d total (%d stable, %d unstable)\n", length(fps), n_stable, n_unstable)
    @printf("  fp → nearest data:    median = %.4f,  mean = %.4f,  max = %.4f\n",
            median(fp_to_data), mean(fp_to_data), maximum(fp_to_data))
    if n_stable > 0
        @printf("  stable fp → data:     median = %.4f,  mean = %.4f,  max = %.4f\n",
                median(stable_to_data), mean(stable_to_data), maximum(stable_to_data))
        for δ in (0.05, 0.1, 0.2)
            frac_fp_close = mean(stable_to_data .<= δ)
            frac_data_covered = mean(data_to_stable .<= δ)
            @printf("    δ = %.2f:  %.0f%% of stable fps within δ of data;  %.1f%% of data within δ of a stable fp\n",
                    δ, 100*frac_fp_close, 100*frac_data_covered)
        end
    end
    return (fp_to_data=fp_to_data, data_to_fp=data_to_fp,
            stable_to_data=stable_to_data, data_to_stable=data_to_stable,
            n_stable=n_stable, n_unstable=n_unstable)
end


# Load
println("Loading training data and fixed-point sets...")
training_data = npzread("models/diffusion/training_data.npy")
println("  training_data: ", size(training_data))

onestep = load("models/diffusion/fps_onestep.jld2")
fps_1, fp_dict_1 = onestep["fps"], onestep["fp_dict"]

kstep_path = "models/diffusion/fps_kstep.jld2"
fps_k, fp_dict_k = nothing, nothing
if isfile(kstep_path)
    kstep = load(kstep_path)
    fps_k, fp_dict_k = kstep["fps"], kstep["fp_dict"]
    println("  Loaded $(length(fps_1)) one-step fps and $(length(fps_k)) k-step fps")
else
    println("  Loaded $(length(fps_1)) one-step fps  (no k-step file yet)")
end

# Baseline: how spread out is the data?
inter_nn = inter_data_nn_distance(training_data)
println("\n--- Inter-data baseline ---")
@printf("  nearest-neighbor distance among training samples:\n")
@printf("    median = %.4f,  mean = %.4f,  max = %.4f\n",
        median(inter_nn), mean(inter_nn), maximum(inter_nn))

# Per-stage stats
summarize("One-step denoiser (t = $(onestep["t_low"]))", fps_1, fp_dict_1, training_data)
if !isnothing(fps_k)
    summarize("Composed $(length(load("models/diffusion/fps_kstep.jld2")["t_schedule"]))-step (t = $(load("models/diffusion/fps_kstep.jld2")["t_schedule"]))",
              fps_k, fp_dict_k, training_data)
end

println("\nDone.")
