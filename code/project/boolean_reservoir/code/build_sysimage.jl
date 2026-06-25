# Build a Julia sysimage that includes PythonCall + pre-compiled res_engine.jl methods.
# Run once on the target machine; thereafter workers load the sysimage instead of JIT-compiling.
#
# Usage (from the project root, with Julia in PATH):
#   julia project/boolean_reservoir/code/build_sysimage.jl
#   # → writes res_engine_sysimage.so next to this file (~5-10 min first time)
#
# Then point juliacall at it before running the grid search:
#   export PYTHON_JULIACALL_SYSIMAGE=/abs/path/to/res_engine_sysimage.so
#
# On SLURM, add that export to your job script before the python invocation.
# The sysimage is machine-specific — rebuild if you move to a different node type.

using PackageCompiler

script_dir = @__DIR__
sysimage_path = joinpath(script_dir, "res_engine_sysimage.so")

# Precompile script: exercises the hot methods via Python-free Julia calls so
# PackageCompiler can snapshot the compiled method instances.
precompile_script = joinpath(script_dir, "_precompile_res_engine.jl")
write(precompile_script, raw"""
using PythonCall

# res_engine.jl defines ReservoirEngine + init_engine + reservoir_tick! etc.
# We can't call init_engine from Julia directly (it takes PyArrays), so we
# construct a ReservoirEngine manually and exercise the hot inner methods.
include(joinpath(@__DIR__, "res_engine.jl"))

let
    N=128; K=4; B=64; NI=16; C=2; kb=4

    # Minimal valid CSR adjacency (each node connects to node 1)
    nb_vals    = ones(Int64, N * K)
    nb_offsets = collect(Int64, 1:K:N*K+1)
    nb_shifts  = repeat(collect(Int64, K-1:-1:0), N)
    no_nb      = Int64[]
    lut        = zeros(UInt8, 2^K, N)
    new_states = zeros(UInt8, B, N)
    w_bi       = zeros(UInt8, C*kb, NI)
    ticks      = ones(UInt8, C)
    masks      = [trues(NI) for _ in 1:C]

    engine = ReservoirEngine(nb_vals, nb_offsets, nb_shifts, no_nb, lut, new_states,
                             N, K, w_bi, ticks, masks, NI, C, kb, 0)

    states = zeros(UInt8, B, N)

    # Exercise the hot paths so their compiled forms are captured in the sysimage
    reservoir_tick!(engine, states, B)
    warmup!(engine, states, B, 4)

    # forward_sequence! requires a PyArray for x — skip it here; PythonCall's
    # pyconvert specialisations are captured via the `using PythonCall` above.
end
""")

create_sysimage(
    ["PythonCall"];
    sysimage_path = sysimage_path,
    precompile_execution_file = precompile_script,
)

rm(precompile_script; force=true)
println("\nSysimage written to: ", sysimage_path)
println("Export before running: PYTHON_JULIACALL_SYSIMAGE=$(sysimage_path)")
