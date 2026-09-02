#=
Round-trip experiment support: enumerate a random ReLU network's PWA cells with
RPM and export them for the Python PWA -> ReLU routes in ../pwa2nn/experiments.

The point of the round trip is ground truth.  Every PWA -> ReLU result so far was
measured on a random mesh, where nobody knows the minimal network, so "is this
near-optimal?" had no answer.  Starting from a network of N hidden neurons
supplies a witness: an exact network of size N provably exists.

Run from the REPL (never `julia export_pwa.jl` -- see CLAUDE.md):

	julia> include("export_pwa.jl")          # smallest case, writes exports/
	julia> run_sweep()                       # the full Stage 1 sweep

reach.jl is not modified; everything here is additive.
=#

include("reach.jl")
using NPZ, Random, Printf

###############################################################################
######## NETWORK GENERATION ###################################################
###############################################################################

"""
Zero-mean He-initialised net in the same augmented format random_net produces.

random_net (load_networks.jl:22) draws entries as sqrt(2/515)*(2*rand - rand)
with TWO independent uniform draws, so entries have mean +0.031 rather than 0.
Measured over the "box" domain in 2D, only 9-14 of 40 neurons ever switch at
hdim=10,layers=5, and gradients come out at 3e-04.  That would test the N^n cell
estimate against a net that behaves as if N were 11, and 3e-04 gradients also
collide under the Python side's distinct-gradient dedupe.

This generator fixes both.  random_net is kept as a baseline (gen=:repo) so the
difference is measured rather than assumed.
"""
function random_net_he(in_d, out_d, hdim, layers; bias_std=0.5)
	layers >= 2 || error("need at least one hidden layer")
	Weights = Vector{Array{Float64,2}}(undef, layers)

	r_weight = sqrt(2/in_d)*randn(hdim, in_d)
	r_bias   = bias_std*randn(hdim, 1)
	Weights[1] = vcat(hcat(r_weight, r_bias), reshape(zeros(1+in_d),1,:))
	Weights[1][end,end] = 1

	for i in 2:layers-1
		r_weight = sqrt(2/hdim)*randn(hdim, hdim)
		r_bias   = bias_std*randn(hdim, 1)
		Weights[i] = vcat(hcat(r_weight, r_bias), reshape(zeros(1+hdim),1,:))
		Weights[i][end,end] = 1
	end

	r_weight = sqrt(2/hdim)*randn(out_d, hdim)
	r_bias   = bias_std*randn(out_d, 1)
	Weights[end] = hcat(r_weight, r_bias)
	return Weights
end

"""H-rep of the box [-span, span]^in_d, matching the `box` branch of input_constraints_random."""
function input_constraints_box(in_d, span)
	Aᵢ = vcat(Matrix{Float64}(I, in_d, in_d), Matrix{Float64}(-I, in_d, in_d))
	bᵢ = span*ones(2*in_d)
	return Aᵢ, bᵢ
end

###############################################################################
######## FACET PROVENANCE #####################################################
###############################################################################

"""
Label each stored facet row of a cell with the neuron that produced it.

compute_reach stores ap2input[ap] = (A,b) as vcat(A[essential,:], Aᵢ[essentialᵢ,:])
(reach.jl:193), dropping the neuron index behind each row.  But remove_redundant
selects rows *verbatim* out of the matrix get_constraints built, and
get_constraints normalises every row to unit norm and canonicalises -0.0
(positive_zeros), so re-deriving that matrix and matching rows recovers the
provenance to well under 1e-12.  reach.jl stays untouched.

Returns (labels, worst, n_ambiguous):
  labels[r] = global neuron index (1-based; get_layer_neuron(labels[r], ap)
              gives (layer, neuron)), or -1 for an input-box facet, or -2 for
              no match at all.
  n_ambiguous counts rows matched by neurons in more than one layer, which is
  the only case where the label is not well defined.
"""
function facet_labels(A_s, b_s, A_all, b_all, Aᵢ, bᵢ, ap; tol=1e-9)
	labels = fill(-2, size(A_s,1))
	worst, n_ambiguous = 0.0, 0

	for r in 1:size(A_s,1)
		best_d, best_i, ties = Inf, 0, Int[]
		for i in 1:size(A_all,1)
			d = norm(A_all[i,:] - A_s[r,:]) + abs(b_all[i] - b_s[r])
			d < best_d && ((best_d, best_i) = (d, i))
			d < tol && push!(ties, i)
		end
		box_d = Inf
		for i in 1:size(Aᵢ,1)
			box_d = min(box_d, norm(Aᵢ[i,:] - A_s[r,:]) + abs(bᵢ[i] - b_s[r]))
		end

		if best_d < tol && best_d <= box_d
			labels[r] = best_i
			if length(ties) > 1
				lyrs = unique([get_layer_neuron(t, ap)[1] for t in ties])
				length(lyrs) > 1 && (n_ambiguous += 1)
			end
			worst = max(worst, best_d)
		elseif box_d < tol
			labels[r] = -1
			worst = max(worst, box_d)
		end
	end
	return labels, worst, n_ambiguous
end

"""Global neuron index -> (layer, neuron), and the inverse."""
global_index(layer_sizes, l, j) = sum(layer_sizes[1:l-1]) + j

###############################################################################
######## EXPORT ###############################################################
###############################################################################

"""
Enumerate the cells of one network and write them to exports/<tag>.npz.

reach=false: the forward image is not needed and affine_map is expensive.
graph=true: ap2neighbors is the cell adjacency the polytopal DC LP needs.
"""
function export_case(; in_d=2, out_d=1, hdim=4, layers=3, span=2.0, seed=1,
					   gen=:he, bias_std=0.5, outdir="exports", tol=1e-9,
					   save=true, tag=nothing)
	out_d == 1 || error("the Python pipeline is scalar throughout; use out_d = 1")
	tag === nothing && (tag = @sprintf("%s_n%d_h%d_L%d_s%d", gen, in_d, hdim, layers, seed))

	Random.seed!(seed)
	weights = gen === :he   ? random_net_he(in_d, out_d, hdim, layers; bias_std=bias_std) :
			  gen === :repo ? random_net(in_d, out_d, hdim, layers) :
			  error("unknown generator $gen (use :he or :repo)")

	Aᵢ, bᵢ = input_constraints_box(in_d, span)
	Aₒ = Matrix{Float64}(undef,0,0)
	bₒ = Vector{Float64}()

	println("\n=== $tag :  in_d=$in_d hdim=$hdim layers=$layers span=$span ===")
	t0 = time()
	ap2input, ap2output, ap2map, ap2backward, ap2neighbors =
		compute_reach(weights, Aᵢ, bᵢ, [Aₒ], [bₒ]; reach=false, graph=true)
	rpm_s = time() - t0

	aps    = collect(keys(ap2input))          # OrderedDict: insertion order
	M      = length(aps)
	ap2idx = Dict(ap => k for (k, ap) in enumerate(aps))
	layer_sizes = [length(aps[1][l]) for l in eachindex(aps[1])]
	num_neurons = sum(layer_sizes)

	# ---- activation patterns and local maps ---------------------------------
	ap_bits = zeros(Int8, M, num_neurons)
	Cs = zeros(Float64, M, in_d)
	ds = zeros(Float64, M)
	for (k, ap) in enumerate(aps)
		i = 1
		for l in eachindex(ap), j in 1:length(ap[l])
			ap_bits[k, i] = ap[l][j]; i += 1
		end
		C, d = ap2map[ap]
		Cs[k, :] = vec(C); ds[k] = d[1]
	end

	# ---- cells, with facet provenance ---------------------------------------
	cellA = Vector{Matrix{Float64}}(undef, M)
	cellb = Vector{Vector{Float64}}(undef, M)
	labels = Vector{Vector{Int}}(undef, M)
	zerow_counts = zeros(Int32, M)
	worst_match, n_ambiguous, n_unmatched = 0.0, 0, 0

	for (k, ap) in enumerate(aps)
		A_s, b_s = ap2input[ap]
		A_all, b_all, _, zerows, _ = get_constraints(weights, ap, num_neurons)
		lab, w, namb = facet_labels(A_s, b_s, A_all, b_all, Aᵢ, bᵢ, ap; tol=tol)
		cellA[k], cellb[k], labels[k] = A_s, b_s, lab
		zerow_counts[k] = length(zerows)
		worst_match = max(worst_match, w)
		n_ambiguous += namb
		n_unmatched += count(==(-2), lab)
	end

	rows = [size(A,1) for A in cellA]
	cell_off = Int32.(vcat(0, cumsum(rows)))
	cellA_flat = vcat(cellA...)
	cellb_flat = vcat(cellb...)
	facet_src  = Int32.(vcat(labels...))

	# ---- adjacency ----------------------------------------------------------
	# add_neighbor_aps (reach.jl:429) inserts the reverse edge for a neighbour AP
	# before that cell is known to be non-empty inside the box, so ap2neighbors
	# can name cells that never made it into ap2input.  Drop those.
	pairs = Set{Tuple{Int,Int}}()
	n_dangling = 0
	for (ap, nbrs) in ap2neighbors
		haskey(ap2idx, ap) || (n_dangling += length(nbrs); continue)
		k = ap2idx[ap]
		for nb in nbrs
			if haskey(ap2idx, nb)
				l = ap2idx[nb]
				k != l && push!(pairs, (min(k,l), max(k,l)))
			else
				n_dangling += 1
			end
		end
	end
	pairs = sort(collect(pairs))

	# ---- per-edge shared facet, gradient jump, rank-1 residual --------------
	E = length(pairs)
	edges = zeros(Int32, E, 2)
	edge_a = zeros(Float64, E, in_d)
	edge_b = zeros(Float64, E)
	edge_gamma = zeros(Float64, E)
	edge_resid = fill(NaN, E)
	edge_src = fill(Int32(-2), E)
	n_nofacet, n_zerojump, max_resid = 0, 0, 0.0

	for (e, (k, l)) in enumerate(pairs)
		Ak, bk = cellA[k], cellb[k]
		Al, bl = cellA[l], cellb[l]
		# the shared facet: row i of k and row j of l with a_i = -a_j, β_i = -β_j
		best_d, bi = Inf, 0
		for i in 1:size(Ak,1), j in 1:size(Al,1)
			d = norm(Ak[i,:] + Al[j,:]) + abs(bk[i] + bl[j])
			d < best_d && ((best_d, bi) = (d, i))
		end
		if best_d >= tol
			n_nofacet += 1
			edges[e,:] = [k-1, l-1]      # 0-based for numpy
			continue
		end
		a, β = Ak[bi,:], bk[bi]           # cell k satisfies a·x <= β, so l is on the + side
		dC = Cs[l,:] - Cs[k,:]
		γ  = dot(dC, a) / dot(a, a)
		res = norm(dC - γ*a)

		edges[e,:]    = [k-1, l-1]
		edge_a[e,:]   = a
		edge_b[e]     = β
		edge_gamma[e] = γ
		edge_resid[e] = res
		edge_src[e]   = labels[k][bi]
		max_resid = max(max_resid, res)
		abs(γ) < tol && (n_zerojump += 1)
	end

	# ---- census -------------------------------------------------------------
	const_rows = Int32[global_index(layer_sizes, l, layer_sizes[l]) for l in eachindex(layer_sizes)]
	live = falses(num_neurons)
	for i in 1:num_neurons
		col = @view ap_bits[:, i]
		live[i] = any(==(Int8(0)), col) && any(==(Int8(1)), col)
	end
	n_hidden = num_neurons - length(const_rows)      # constant-1 rows are not neurons
	n_live = count(live) - count(i -> live[i], const_rows)
	distinct_C = length(unique([Cs[k,:] for k in 1:M]))
	grad_scale = maximum(abs, Cs)

	@printf("cells M = %d   edges E = %d   RPM %.2f s\n", M, E, rpm_s)
	@printf("neurons: %d nominal, %d live (%.0f%%)   distinct gradients: %d\n",
			n_hidden, n_live, 100*n_live/n_hidden, distinct_C)
	@printf("max|grad| = %.3e\n", grad_scale)
	@printf("census: worst facet match %.2e, ambiguous %d, unmatched %d,\n",
			worst_match, n_ambiguous, n_unmatched)
	@printf("        edges without a shared facet %d, zero-jump edges %d, dangling nbrs %d\n",
			n_nofacet, n_zerojump, n_dangling)
	@printf("        max rank-1 residual %.2e, zerows/cell %d..%d\n",
			max_resid, minimum(zerow_counts), maximum(zerow_counts))

	# ---- write --------------------------------------------------------------
	data = Dict{String,Any}(
		"n"              => Int32[in_d],
		"out_d"          => Int32[out_d],
		"hdim"           => Int32[hdim],
		"n_layers"       => Int32[layers],
		"M"              => Int32[M],
		"span"           => Float64[span],
		"seed"           => Int32[seed],
		"rpm_seconds"    => Float64[rpm_s],
		"ap_layer_sizes" => Int32.(layer_sizes),
		"const_rows"     => const_rows,
		"box_A"          => Aᵢ,
		"box_b"          => bᵢ,
		"ap_bits"        => ap_bits,
		"C"              => Cs,
		"d"              => ds,
		"cellA_flat"     => cellA_flat,
		"cellb_flat"     => cellb_flat,
		"cell_off"       => cell_off,
		"facet_src"      => facet_src,
		"edges"          => edges,
		"edge_a"         => edge_a,
		"edge_b"         => edge_b,
		"edge_gamma"     => edge_gamma,
		"edge_resid"     => edge_resid,
		"edge_src"       => edge_src,
		"zerow_counts"   => zerow_counts,
		"census"         => Int32[n_ambiguous, n_unmatched, n_nofacet, n_zerojump,
								  n_dangling, n_live, n_hidden, distinct_C],
	)
	for (i, W) in enumerate(weights)
		data["W$i"] = W
	end

	if save
		isdir(outdir) || mkdir(outdir)
		path = joinpath(outdir, tag * ".npz")
		npzwrite(path, data)
		println("wrote $path")
	end
	return data
end

###############################################################################
######## SCRIPTING ############################################################
###############################################################################

"""The Stage 1 sweep: smallest first, both generators, 3 seeds."""
function run_sweep(; seeds=1:3, outdir="exports")
	archs = [(2, 4, 3), (2, 10, 3), (2, 10, 5), (3, 6, 3)]
	for (in_d, hdim, layers) in archs, gen in (:he, :repo), seed in seeds
		try
			export_case(in_d=in_d, hdim=hdim, layers=layers, seed=seed,
						gen=gen, outdir=outdir)
		catch err
			println("FAILED  n=$in_d hdim=$hdim layers=$layers gen=$gen seed=$seed: $err")
		end
	end
end

# Default include() runs only the smallest case, per the "start SMALL" constraint.
data = export_case(in_d=2, hdim=4, layers=3, seed=1, gen=:he)
println("\nrun_sweep() for the full Stage 1 sweep.")
