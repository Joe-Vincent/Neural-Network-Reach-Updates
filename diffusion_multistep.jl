using Plots, FileIO, JLD2, NPZ, LinearAlgebra
include("reach.jl")
include("invariance.jl")
include("diffusion_helpers.jl")


### SCRIPTING (multi-step composed analysis) ###

# Load the full time-conditioned network (R^3 → R^2)
nn_weights = "models/diffusion/weights.npz"
nn_params  = "models/diffusion/norm_params.npz"
weights_full = pytorch_net(nn_weights, nn_params, 1)
println("Loaded weights_full: ", length(weights_full), " layers, weights_full[1] size = ", size(weights_full[1]))

# DDPM sampling uses a decreasing t schedule. We compose the per-step frozen networks.
# Start with k=3 steps; if compute_reach blows up, drop to k=2.
const t_schedule = [0.5, 0.3, 0.1]
println("\nt_schedule = ", t_schedule, " (K = $(length(t_schedule)) composed steps)")

weights_list = [freeze_time_input(weights_full, t) for t in t_schedule]
println("  built $(length(weights_list)) frozen-t weight vectors, each $(length(weights_list[1])) layers")

# Sanity-check chain_varied: evaluating the chained network on a few inputs should equal
# manually composing the per-step networks.
weights_k = chain_varied(weights_list)
println("  composed network length = ", length(weights_k))
println("  weights_k[1] size = ", size(weights_k[1]))
println("  weights_k[end] size = ", size(weights_k[end]))

let
    println("\nchain_varied round-trip check (5 random inputs):")
    max_err = 0.0
    for _ in 1:5
        x = (2 .* rand(2) .- 1) .* 2.0
        # Manual composition: apply each frozen step in sequence
        x_manual = copy(x)
        for w in weights_list
            x_manual = eval_net(x_manual, w, 1)
        end
        x_chained = eval_net(x, weights_k, 1)
        err = norm(x_manual - x_chained)
        max_err = max(max_err, err)
        println("  x = $(round.(x, digits=4)), |manual - chained| = $(round(err, sigdigits=6))")
    end
    println("  max error = $(round(max_err, sigdigits=6))")
    @assert max_err < 1e-9 "chain_varied round-trip failed"
end

# Set up domain + dummy output constraints
Aᵢ, bᵢ = input_constraints_diffusion("box")
Aₒ, bₒ = output_constraints_diffusion("dummy")

# Run RPM on the composed network
println("\nRunning compute_reach on $(length(t_schedule))-step composed network...")
@time begin
    ap2input_k, ap2output_k, ap2map_k, ap2backward_k =
        compute_reach(weights_k, Aᵢ, bᵢ, [Aₒ], [bₒ])
end
println("  # cells = ", length(ap2input_k))

# Find fixed points
println("\nFinding fixed points of composed map...")
fps_k, fp_dict_k = find_fixed_points_robust(ap2map_k, ap2input_k, weights_k)

# Save
training_data = npzread("models/diffusion/training_data.npy")
save("models/diffusion/fps_kstep.jld2",
     Dict("fps" => fps_k,
          "fp_dict" => fp_dict_k,
          "ap2input" => ap2input_k,
          "ap2map" => ap2map_k,
          "t_schedule" => collect(t_schedule)))
println("\nSaved models/diffusion/fps_kstep.jld2")

plt = plot_diffusion_result(ap2input_k, fps_k, fp_dict_k, training_data;
                            title="$(length(t_schedule))-step composed denoiser fps  (t = $t_schedule)")
savefig(plt, "models/diffusion/fps_kstep.png")
println("Saved models/diffusion/fps_kstep.png")
