# mcLJ
**A parallel Metropolis Monte Carlo code for simulating a Lennard-Jones fluid**

This is my first program in Julia!

Even with 1 CPU core, this program runs almost **10x** faster compared to my previous Python implementation (with `numpy` and `numba` optimizations). 

Moreover, the program runs in parallel by launching separate Monte Carlo simulations on every available processor and combining the results in the end, enabling almost linear scaling.

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
