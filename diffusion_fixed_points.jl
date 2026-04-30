using Plots, FileIO, JLD2, NPZ, LinearAlgebra
include("reach.jl")
include("invariance.jl")
include("diffusion_helpers.jl")


### SCRIPTING (one-step analysis) ###

# Load full time-conditioned network (R^3 → R^2)
nn_weights = "models/diffusion/weights.npz"
nn_params  = "models/diffusion/norm_params.npz"
weights_full = pytorch_net(nn_weights, nn_params, 1)
println("Loaded weights_full: ", length(weights_full), " layers")
println("  weights_full[1] size = ", size(weights_full[1]))
println("  weights_full[end] size = ", size(weights_full[end]))

# Sanity-check freeze_time_input correctness via eval_net round-trip.
let
    t_test = 0.05
    weights_frozen = freeze_time_input(weights_full, t_test)
    println("\nfreeze_time_input round-trip check (5 random inputs at t=$t_test):")
    max_err = 0.0
    for _ in 1:5
        x = (2 .* rand(2) .- 1) .* 2.0
        y_full = eval_net(vcat(x, t_test), weights_full, 1)
        y_frozen = eval_net(x, weights_frozen, 1)
        err = norm(y_full - y_frozen)
        max_err = max(max_err, err)
        println("  x = $(round.(x, digits=4)), |y_full - y_frozen| = $(round(err, sigdigits=6))")
    end
    println("  max error = $(round(max_err, sigdigits=6))")
    @assert max_err < 1e-10 "freeze_time_input round-trip failed"
end

# Pick a low t for "near-clean" analysis and freeze it
const t_low = 0.05
weights_1 = freeze_time_input(weights_full, t_low)
println("\nweights_1 (t = $t_low frozen): ", length(weights_1), " layers")
println("  weights_1[1] size = ", size(weights_1[1]))

# Set up domain + dummy output constraints
Aᵢ, bᵢ = input_constraints_diffusion("box")
Aₒ, bₒ = output_constraints_diffusion("dummy")

# Run RPM
println("\nRunning compute_reach on one-step denoiser at t = $t_low...")
@time begin
    ap2input, ap2output, ap2map, ap2backward = compute_reach(weights_1, Aᵢ, bᵢ, [Aₒ], [bₒ])
end
println("  # cells = ", length(ap2input))

# Find fixed points
println("\nFinding fixed points...")
fps, fp_dict = find_fixed_points_robust(ap2map, ap2input, weights_1)

# Save
training_data = npzread("models/diffusion/training_data.npy")
save("models/diffusion/fps_onestep.jld2",
     Dict("fps" => fps,
          "fp_dict" => fp_dict,
          "ap2input" => ap2input,
          "ap2map" => ap2map,
          "t_low" => t_low))
println("\nSaved models/diffusion/fps_onestep.jld2")

# Plot
plt = plot_diffusion_result(ap2input, fps, fp_dict, training_data;
                            title="One-step denoiser fixed points (t = $t_low)")
savefig(plt, "models/diffusion/fps_onestep.png")
println("Saved models/diffusion/fps_onestep.png")
