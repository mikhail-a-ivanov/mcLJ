# mcLJ
A parallel Metropolis Monte Carlo code for simulating a Lennard-Jones fluid

This is my first program in Julia!

# How to run:
1. Set the input parameters in the `LJ-init.in` file
2. Run the simulation: `julia -p n mcLJ.jl LJ-init.in` (where `n` is the number of available cores)

## Required Julia packages
**Julia version: 1.7**

Core packages:
- `Distributed`
- `StaticArrays`
- `RandomNumbers`
- `LinearAlgebra`

Other packages:
- `Dates`
- `Printf`
- `Statistics`
