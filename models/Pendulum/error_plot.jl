include("../../load_networks.jl")

# torque-controlled damped pendulum #
# dynamics from http://control.asu.edu/Publications/2018/Colbert_CDC_2018.pdf


RK_f(x, u; m=1., l=1., b=1.) = [x[2], (u - b*x[2] - m*9.81*l*sin(x[1]))/(m*l^2)]


function RK_update(x, u, dt)
	k1 = RK_f(x, u)
	k2 = RK_f(x + dt*0.5*k1, u)
	k3 = RK_f(x + dt*0.5*k2, u)
	k4 = RK_f(x + dt*k3, u)
	return x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
end

bound_r(a,b) = (b-a)*(rand()-1) + b # Generates a uniformly random number on [a,b]

# generates data where each X[i,:] is an input and each corresponding Y[i,:] is an output
function gen_data(n)
	dt = 0.1
	X = hcat([[bound_r(-2*π/3, 2*π/3), bound_r(-π, π), bound_r(-2,2)] for i in 1:n]...)'
	Y = hcat([RK_update(X[i,1:2], X[i,3], dt) for i in 1:n]...)'
	npzwrite("models/Pendulum/X_controlled.npy", X)
	npzwrite("models/Pendulum/Y_controlled.npy", Y)
	return nothing
end

copies = 1 # copies = 1 is original network
nn_weights = "models/Pendulum/weights_controlled.npz"
nn_params = "models/Pendulum/norm_params_controlled.npz"
weights = pytorch_net(nn_weights, nn_params, copies)

n = 100_000
errors = Matrix{Float64}(undef, 2, n)
for i in 1:n
	xₒ = [bound_r(-2*π/3, 2*π/3), bound_r(-π, π)]
	u = bound_r(-2,2)
	x′_true = rad2deg.(RK_update(xₒ, u, 0.1))
	x′_est = rad2deg.(eval_net([xₒ; u], weights, 1))
	errors[:,i] = x′_est - x′_true
end

# plot errors
plt = plot(reuse=false, legend=false, title="Error", xlabel="x₁: Angle (°)", ylabel="x₂: Angular Velocity (°/s)")
scatter!(plt, errors[1,:], errors[2,:])