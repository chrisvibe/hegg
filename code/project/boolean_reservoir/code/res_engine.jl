using PythonCall
using LinearAlgebra

mutable struct ReservoirEngine
    nb_vals::Vector{Int64}              # CSR: flattened 1-indexed neighbor indices
    nb_offsets::Vector{Int64}           # CSR: nb_offsets[n]:nb_offsets[n+1]-1 are neighbors of n
    nb_shifts::Vector{Int64}            # pre-computed bit-shifts (k_i - pos) per neighbor
    no_neighbours_indices::Vector{Int64}# 1-indexed isolated nodes
    lut_flat::Vector{UInt8}             # jagged LUT: all nodes' entries concatenated
    lut_offsets::Vector{Int64}          # 1-indexed offsets: node n's LUT starts at lut_offsets[n]
    new_states::Matrix{UInt8}           # double-buffer (batch_size, N_total)
    N_total::Int
    max_k::Int
    # Cached per-model constants (avoids pyconvert on every forward call)
    w_bi::Matrix{UInt8}                 # (c*k, N_I) — input connection weights
    ticks::Vector{UInt8}                # (c,) — reservoir ticks per chunk
    chunk_masks::Vector{Vector{Bool}}   # per-chunk input-node masks (precomputed from w_bi + pert)
    input_nodes::Vector{Int64}          # 1-indexed global positions of I-nodes (from selector)
    N_I::Int                            # number of input nodes (for chunk_masks/w_bi sizing)
    c::Int                              # number of input chunks
    k_bits::Int                         # bits per chunk
    pert::Int                           # perturbation mode: xor=0, override=1, and=2, or=3
    # Readout layer (weights pushed from Python after ridge solve; updated in-place for FORCE)
    output_indices::Vector{Int64}       # 1-indexed positions of output nodes in state vector
    readout_W::Matrix{Float32}          # (n_out, n_r) — readout weight matrix
    readout_b::Vector{Float32}          # (n_out,)    — readout bias
end

"""
Convert Python's flat jagged LUT + 0-based offsets to native Julia arrays.
lut_offsets is shifted to 1-based so hot-path access is `lut_flat[lut_offsets[n] + idx]`.
"""
function build_lut_jagged(lut_flat_py, lut_offsets_py)
    lut_flat    = pyconvert(Vector{UInt8}, lut_flat_py)
    lut_offsets = pyconvert(Vector{Int64}, lut_offsets_py) .+ Int64(1)  # 0-based → 1-based
    return lut_flat, lut_offsets
end

"""
Build CSR adjacency (nb_vals, nb_offsets, nb_shifts) from a padded adj_list + mask.
nb_shifts use per-node relative bit positions: shift = k_i - pos (MSB-first within
each node's actual degree k_i), so LUT index is always in [0, 2^k_i - 1].
"""
function build_csr_adj(adj_list, adj_mask, N_total::Int, max_k::Int)
    nb_vals    = Int64[]
    nb_offsets = Int64[1]
    nb_shifts  = Int64[]
    for n in 1:N_total
        k_i = sum(view(adj_mask, n, :))   # actual in-degree
        pos = 0
        for k in 1:max_k
            if adj_mask[n, k]
                pos += 1
                push!(nb_vals,   adj_list[n, k])
                push!(nb_shifts, k_i - pos)    # per-node MSB-first
            end
        end
        push!(nb_offsets, length(nb_vals) + 1)
    end
    return nb_vals, nb_offsets, nb_shifts
end

"""
    init_engine(adj_list_py, adj_list_mask_py, no_neighbours_py,
                lut_flat_py, lut_offsets_py,
                w_bi_py, ticks_py, input_nodes_py, output_mask_py,
                N_total, max_k, batch_size, N_I, c, k_bits, pert, n_out)

Called once from Python at model init. Copies reference arrays into Julia memory
and pre-allocates the double-buffer. Readout weights initialised to zeros; call
set_readout_b after ridge solve to install trained weights.
"""
function init_engine(adj_list_py, adj_list_mask_py, no_neighbours_py,
                     lut_flat_py, lut_offsets_py,
                     w_bi_py, ticks_py, input_nodes_py, output_mask_py,
                     N_total::Int, max_k::Int, batch_size::Int,
                     N_I::Int, c::Int, k_bits::Int, pert::Int, n_out::Int)
    adj_list      = pyconvert(Matrix{Int64}, adj_list_py) .+ Int64(1)
    adj_mask      = pyconvert(Matrix{Bool},  adj_list_mask_py)
    no_nb         = pyconvert(Vector{Int64}, no_neighbours_py) .+ Int64(1)
    input_nodes   = pyconvert(Vector{Int64}, input_nodes_py)   .+ Int64(1)
    output_mask   = pyconvert(Vector{Bool},  output_mask_py)

    lut_flat, lut_offsets          = build_lut_jagged(lut_flat_py, lut_offsets_py)
    nb_vals, nb_offsets, nb_shifts = build_csr_adj(adj_list, adj_mask, N_total, max_k)

    w_bi  = pyconvert(Matrix{UInt8}, w_bi_py)
    ticks = pyconvert(Vector{UInt8}, ticks_py)

    needs_mask = (pert == 1 || pert == 2)
    chunk_masks = Vector{Vector{Bool}}(undef, c)
    row = 1
    for ci in 1:c
        row_end = row + k_bits - 1
        if needs_mask
            mask = Vector{Bool}(undef, N_I)
            for n in 1:N_I
                connected = false
                for r in row:row_end
                    if w_bi[r, n] != 0x00
                        connected = true
                        break
                    end
                end
                mask[n] = connected
            end
            chunk_masks[ci] = mask
        else
            chunk_masks[ci] = trues(N_I)
        end
        row += k_bits
    end

    new_states     = zeros(UInt8, batch_size, N_total)
    output_indices = findall(output_mask)   # 1-indexed
    n_r            = length(output_indices)
    readout_W      = zeros(Float32, n_out, n_r)
    readout_b      = zeros(Float32, n_out)

    return ReservoirEngine(nb_vals, nb_offsets, nb_shifts, no_nb, lut_flat, lut_offsets,
                           new_states, N_total, max_k, w_bi, ticks, chunk_masks,
                           input_nodes, N_I, c, k_bits, pert,
                           output_indices, readout_W, readout_b)
end

"""
    set_readout_b(engine, W_py, b_py)

Push trained readout weights from Python into the Julia engine.
Call after ridge solve. Also the update point for future FORCE learning.
Python access: jl.set_readout_b(engine, W, b)
"""
function set_readout!(engine::ReservoirEngine, W_py, b_py)
    engine.readout_W = pyconvert(Matrix{Float32}, W_py)
    engine.readout_b = pyconvert(Vector{Float32}, b_py)
    return nothing
end

"""
    apply_readout_b(engine, states_py, m, bipolar)

Extract output nodes from states, apply float conversion, bipolar shift if needed,
then compute W @ o + b. Returns (m, n_out) Float32 matrix.
For classification mode (final state).
Python access: jl.apply_readout_b(engine, states, m, bipolar)
"""
function apply_readout!(engine::ReservoirEngine, states_py::AbstractMatrix{UInt8},
                          m::Int, bipolar::Bool)
    n_r   = length(engine.output_indices)
    n_out = length(engine.readout_b)
    o = Matrix{Float32}(undef, m, n_r)
    @inbounds for (ri, ni) in enumerate(engine.output_indices)
        @simd for s in 1:m
            o[s, ri] = Float32(states_py[s, ni])
        end
    end
    if bipolar
        @. o = o * 2.0f0 - 1.0f0
    end
    # (m, n_r) * (n_r, n_out) + (n_out,) broadcast → (m, n_out)
    return o * engine.readout_W' .+ engine.readout_b'
end

"""
    apply_readout_timeseries_b(engine, step_buffer_py, m, s, bipolar)

Time-series readout: apply W @ o_step + b for each step, sum over steps.
step_buffer_py shape: (s, m, N_total). Returns (m, n_out) Float32.
Python access: jl.apply_readout_timeseries_b(engine, step_buffer, m, s, bipolar)
"""
function apply_readout_timeseries!(engine::ReservoirEngine, step_buffer_py, m::Int, s::Int, bipolar::Bool)
    step_buffer = pyconvert(Array{UInt8, 3}, step_buffer_py)  # (s, m, N_total)
    n_r   = length(engine.output_indices)
    n_out = length(engine.readout_b)
    outputs = zeros(Float32, m, n_out)
    o = Matrix{Float32}(undef, m, n_r)
    for si in 1:s
        @inbounds for (ri, ni) in enumerate(engine.output_indices)
            @simd for samp in 1:m
                o[samp, ri] = Float32(step_buffer[si, samp, ni])
            end
        end
        if bipolar
            @. o = o * 2.0f0 - 1.0f0
        end
        outputs += o * engine.readout_W' .+ engine.readout_b'
    end
    return outputs
end

"""
    reservoir_tick!(engine, states_py, m)

Synchronous Boolean update for m samples in parallel.
"""
function reservoir_tick!(engine::ReservoirEngine, states_py::AbstractMatrix{UInt8}, m::Int)
    N = engine.N_total

    @inbounds for n in 1:N
        lo = engine.nb_offsets[n]
        hi = engine.nb_offsets[n + 1] - 1
        for s in 1:m
            idx = 0
            for ki in lo:hi
                idx |= Int(states_py[s, engine.nb_vals[ki]]) << engine.nb_shifts[ki]
            end
            engine.new_states[s, n] = engine.lut_flat[engine.lut_offsets[n] + idx]
        end
    end

    for n in engine.no_neighbours_indices
        @simd for s in 1:m
            engine.new_states[s, n] = states_py[s, n]
        end
    end

    @inbounds for n in 1:N
        @simd for s in 1:m
            states_py[s, n] = engine.new_states[s, n]
        end
    end
end

"""
    warmup_b(engine, states_py, m, ticks)

Run `ticks` reservoir ticks without input to wash out the init state.
Python access: jl.warmup_b(engine, states_py, m, ticks)
"""
function warmup!(engine::ReservoirEngine, states_py::AbstractMatrix{UInt8}, m::Int, ticks::Int)
    for _ in 1:ticks
        reservoir_tick!(engine, states_py, m)
    end
end

"""
    forward_sequence_b(engine, states_py, x_py, history_buffer, step_buffer)

Full Boolean forward pass (s × c × t loop) in Julia.
- states_py     : AbstractMatrix{UInt8} (batch_size, N_total), modified in-place
- x_py          : Fortran-order Array{UInt8,4} (m, s, c, k)
- history_buffer: PyArray{UInt8,3} (n_entries, m, N_total) or Python None
- step_buffer   : PyArray{UInt8,3} (s, m, N_total) or Python None (for time-series readout)

Python access: jl.forward_sequence_b(engine, states, x, history_buffer, step_buffer)
"""
function forward_sequence!(engine::ReservoirEngine, states_py::AbstractMatrix{UInt8},
                              x_py, history_buffer, step_buffer)
    x = pyconvert(Array{UInt8, 4}, x_py)
    m = size(x, 1)
    s = size(x, 2)

    has_history  = !pyis(history_buffer, pybuiltins.None)
    has_stepbuf  = !pyis(step_buffer,   pybuiltins.None)

    pert_bits = zeros(UInt8, m)
    entry_idx = 1

    for si in 1:s
        a = 1
        for ci in 1:engine.c
            has_conn = engine.chunk_masks[ci]

            @inbounds for ni in eachindex(engine.input_nodes)
                n = engine.input_nodes[ni]
                has_conn[ni] || continue

                fill!(pert_bits, 0x00)
                for bit in 1:engine.k_bits
                    wb = engine.w_bi[a + bit - 1, ni]
                    wb == 0x00 && continue
                    @simd for samp in 1:m
                        pert_bits[samp] |= x[samp, si, ci, bit]
                    end
                end

                for samp in 1:m
                    old = states_py[samp, n]
                    pb  = pert_bits[samp]
                    states_py[samp, n] = engine.pert == 0 ? old ⊻ pb :
                                         engine.pert == 1 ? pb         :
                                         engine.pert == 2 ? old & pb   :
                                                            old | pb
                end
            end
            a += engine.k_bits

            if has_history
                @inbounds for n in 1:engine.N_total
                    @simd for samp in 1:m
                        history_buffer[entry_idx, samp, n] = states_py[samp, n]
                    end
                end
                entry_idx += 1
            end

            t = Int(engine.ticks[ci])
            for ti in 1:t
                reservoir_tick!(engine, states_py, m)

                is_last = (si == s) && (ci == engine.c) && (ti == t)
                if has_history && !is_last
                    @inbounds for n in 1:engine.N_total
                        @simd for samp in 1:m
                            history_buffer[entry_idx, samp, n] = states_py[samp, n]
                        end
                    end
                    entry_idx += 1
                end
            end
        end

        if has_stepbuf
            @inbounds for n in 1:engine.N_total
                @simd for samp in 1:m
                    step_buffer[si, samp, n] = states_py[samp, n]
                end
            end
        end
    end

    if has_history
        @inbounds for n in 1:engine.N_total
            @simd for samp in 1:m
                history_buffer[entry_idx, samp, n] = states_py[samp, n]
            end
        end
    end

    return states_py
end
