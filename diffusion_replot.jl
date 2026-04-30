# Re-render the fp plots from saved jld2 without re-running compute_reach.
using Plots, FileIO, JLD2, NPZ, LinearAlgebra
include("reach.jl")
include("invariance.jl")
include("diffusion_helpers.jl")

training_data = npzread("models/diffusion/training_data.npy")

println("Re-rendering one-step plot...")
onestep = load("models/diffusion/fps_onestep.jld2")
plt1 = plot_diffusion_result(onestep["ap2input"], onestep["fps"], onestep["fp_dict"],
                              training_data;
                              title="One-step denoiser fixed points  (t = $(onestep["t_low"]))")
savefig(plt1, "models/diffusion/fps_onestep.png")
println("  wrote models/diffusion/fps_onestep.png")

println("\nRe-rendering 3-step plot...")
kstep = load("models/diffusion/fps_kstep.jld2")
plt2 = plot_diffusion_result(kstep["ap2input"], kstep["fps"], kstep["fp_dict"],
                              training_data;
                              title="3-step composed denoiser fps  (t = $(kstep["t_schedule"]))")
savefig(plt2, "models/diffusion/fps_kstep.png")
println("  wrote models/diffusion/fps_kstep.png")
