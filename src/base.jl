using Printf
using RandomNumbers
using StaticArrays

include("distances.jl")

"""
ljlattice(parameters)

Generates a cubic latice of LJ atoms
separated by scaled Rm distance,
and the periodic box vectors in reduced units
"""
function ljlattice(parameters)
    lattice = [convert(SVector{3, Float32}, [i, j, k]) 
        for i in 0:parameters.latticePoints-1 
            for j in 0:parameters.latticePoints-1 
                for k in 0:parameters.latticePoints-1]
    scaling::Float64 = (2^(1/6)) * (parameters.latticeScaling)
    lattice = lattice .* scaling
    return(lattice)
end

"""
totalenergy(distanceMatrix, parameters)

Computes the total potential energy in reduced units
for a given distance matrix
"""
function totalenergy(distanceMatrix, parameters)
    E::Float64 = 0.
    @inbounds for i in 1:parameters.N
        @inbounds @fastmath for j in 1:i-1
            r6 = (1/distanceMatrix[i,j])^6
            E += 4 * (r6^2 - r6)
        end
    end
    return(E)
end

"""
particleenergy(distanceVector)

Computes the potential energy of one particle
from a given distance vector
"""
function particleenergy(distanceVector)
    E::Float64 = 0.
    @inbounds @fastmath for i in 1:length(distanceVector)
        if distanceVector[i] != 0
            r6 = (1/distanceVector[i])^6
            E += 4 * (r6^2 - r6)
        end
    end
    return(E)
end

"""
hist!(distanceMatrix, hist, parameters)

Accumulates pair distances in a histogram
"""
function hist!(distanceMatrix, hist, parameters)
    @inbounds for i in 1:parameters.N
        @inbounds @fastmath for j in 1:i-1
            histIndex = floor(Int, 0.5 + distanceMatrix[i,j]/parameters.binWidth)
            if histIndex <= parameters.Nbins
                hist[histIndex] += 1
            end
        end
    end
    return(hist)
end

"""
mcmove!(conf, parameters, distanceMatrix, E, rng)

Performs a Metropolis Monte Carlo
displacement move
"""
function mcmove!(conf, parameters, distanceMatrix, E, rng)
    # Pick a particle at random and calculate its energy
    pointIndex = rand(rng, Int32(1):Int32(length(conf)))
    distanceVector = distanceMatrix[:, pointIndex]
    E1 = particleenergy(distanceVector)

    # Displace the particle
    dr = SVector{3, Float32}(parameters.delta*(rand(rng, Float32) - 0.5), 
                             parameters.delta*(rand(rng, Float32) - 0.5), 
                             parameters.delta*(rand(rng, Float32) - 0.5))
    
    conf[pointIndex] += dr

    # Update the distance vector and calculate energy
    updatedistance!(conf, parameters.box, distanceVector, pointIndex)
    E2 = particleenergy(distanceVector)

    # Get energy difference
    ΔE = E2 - E1
    # Acceptance counter
    accepted = 0

    if rand(rng, Float64) < exp(-ΔE*parameters.β)
        accepted += 1
        E += ΔE
        # Update distance matrix
        distanceMatrix[pointIndex, :] = distanceVector
        distanceMatrix[:, pointIndex] = distanceVector
    else
        conf[pointIndex] -= dr
    end
    return(conf, E, accepted, distanceMatrix)
end

"""
mcrun(parameters)

Runs the Monte Carlo simulation for a given
set of input parameters
"""
function mcrun(parameters)
    # Get the worker id and the output filenames
    if nprocs() == 1
        id = myid()
    else
        id = myid() - 1
    end
    idString = lpad(id, 3, '0')
    energyFile = "energies-p$(idString).dat"
    trajFile = "mctraj-p$(idString).xyz"

    # Generate LJ lattice
    conf = ljlattice(parameters)

    # Initialize the distance histogram
    hist = zeros(Int, parameters.Nbins)

    # Initialize RNG
    rng_xor = RandomNumbers.Xorshifts.Xoroshiro128Plus()

    # Build distance matrix
    distanceMatrix = builddistanceMatrix(conf, parameters.box)

    # Initialize the total energy
    E = totalenergy(distanceMatrix, parameters)

    # Save initial configuration and energy
    if parameters.outlevel >= 2
        writeenergies(E, 0, false, energyFile)
    end

    # Start writing MC trajectory
    if parameters.outlevel == 3
        writexyz(conf, 0, parameters, false, trajFile)
    end

    # Acceptance counters
    acceptedTotal = 0
    acceptedIntermediate = 0

    # Run MC simulation
    @inbounds @fastmath for i in 1:parameters.steps
        conf, E, accepted, distanceMatrix = mcmove!(conf, parameters, distanceMatrix, E, rng_xor)
        acceptedTotal += accepted
        acceptedIntermediate += accepted

            # MC output
            if i % parameters.outfreq == 0 && i > parameters.Eqsteps && parameters.outlevel >= 1
                hist = hist!(distanceMatrix, hist, parameters)
            end

            if i % parameters.outfreq == 0 && parameters.outlevel >= 2
                writeenergies(E, i, true, energyFile)
            end

            if i % parameters.xyzout == 0 && parameters.outlevel == 3
                writexyz(conf, i, parameters, true, trajFile)
            end
            # Perform MC step adjustment during the equilibration
            if parameters.stepAdjustFreq > 0 && i % parameters.stepAdjustFreq == 0 && i < parameters.Eqsteps
                stepAdjustment!(parameters, acceptedIntermediate)
                acceptedIntermediate = 0
            end
    end

    if parameters.outlevel >= 1
        # Normalize the histogram to the number of frames
        Nframes = (parameters.steps - parameters.Eqsteps) / parameters.outfreq
        hist /= Nframes
    end

    acceptanceRatio = acceptedTotal / parameters.steps
    
    return(hist, acceptanceRatio)
end

"""
function stepAdjustment!(parameters, acceptedIntermediate)

MC step length adjustment
"""
function stepAdjustment!(parameters, acceptedIntermediate)
    acceptanceRatio = acceptedIntermediate / parameters.stepAdjustFreq
    #println("Current acceptance ratio = $(round(acceptanceRatio, digits=4))")
    parameters.delta = acceptanceRatio * parameters.delta / parameters.targetAR
    #println("New maximum displacement length = $(round((parameters.delta * parameters.sigma), digits=4)) Å")
    return(parameters)
end

"""
writexyz(conf, currentStep, parameters, append, outname, atomtype="Ar")

Writes a configuration to an XYZ file
"""
function writexyz(conf, currentStep, parameters, append, outname, atomtype="Ar")
    if append
        io = open(outname, "a")
    else
        io = open(outname, "w")
    end
    print(io, parameters.N, "\n")
    print(io, "Step = ", @sprintf("%d", currentStep), "\n")
    for i in 1:parameters.N
        print(io, atomtype, " ")
        for j in 1:3
            print(io, @sprintf("%10.3f", conf[i][j]*parameters.sigma), " ")
            if j == 3
                print(io, "\n")
            end
        end
    end
    close(io)
end

"""
writeenergies(energy, currentStep, append, outname)

Writes the total energy to an output file
"""
function writeenergies(energy, currentStep, append, outname)
    if append
        io = open(outname, "a")
    else
        io = open(outname, "w")
    end
    print(io, "# Total energy in reduced units \n")
    print(io, "# Step = ", @sprintf("%d", currentStep), "\n")
    print(io, @sprintf("%10.3f", energy), "\n")
    print(io, "\n")
    close(io)
end

"""
writeRDF(outname, hist, parameters)

Normalizes the histogram to RDF and writes it into a file
"""
function writeRDF(outname, hist, parameters)
    # Normalize the historgram
    V = (parameters.box[1])^3
    Npairs::Int = parameters.N*(parameters.N-1)/2
    bins = [bin*parameters.binWidth for bin in 1:parameters.Nbins]
    rdfNorm = [(V/Npairs) * 1/(4*π*parameters.binWidth*bins[i]^2) for i in 2:length(bins)]
    RDF = hist[2:end] .* rdfNorm

    # Convert bins to Å
    bins .*= parameters.sigma

    # Write the data
    io = open(outname, "w")
    print(io, "# RDF data \n")
    print(io, "# r, Å; g(r); Histogram \n")
    print(io, @sprintf("%6.3f %12.3f %12.3f", 0, 0, hist[1]), "\n")
    for i in 2:length(hist)
        print(io, @sprintf("%6.3f %12.3f %12.3f", bins[i], RDF[i-1], hist[i]), "\n")
    end
    close(io)
end

"""
mutable struct inputParms

Fields:
latticePoints: number of LJ lattice points
N: number of particles
atommass: atom mass [amu]
sigma: sigma LJ parameter [Å]
epsilon: epsilon LJ parameter, ϵ/kB [K]
T: temperature [K]
β: ϵ/(kB*T), reciprocal reduced temperature
density: target density [kg/m3]
densityRm: initial density [kg/m3]
latticeScaling: lattice scaling factor
box: box vector, σ
delta: max displacement [σ]
steps: total number of steps
Eqsteps: equilibration steps
stepAdjustFreq: frequency of MC step adjustment
targetAR: target acceptance ratio
binWidth: histogram bin width [σ]
Nbins: number of histogram bins
xyzout: XYZ output frequency
outfreq: output frequency
outlevel: output level (0: no output, 1: +RDF, 2: +energies, 3: +trajectories)

"""
mutable struct inputParms
    latticePoints::Int
    N::Int
    atommass::Float64
    sigma::Float32
    epsilon::Float64
    T::Float64
    β::Float64
    density::Float64
    densityRm::Float64
    latticeScaling::Float64
    box::SVector{3, Float32}
    delta::Float32  
    steps::Int
    Eqsteps::Int
    stepAdjustFreq::Int
    targetAR::Float64
    binWidth::Float32
    Nbins::Int
    xyzout::Int 
    outfreq::Int
    outlevel::Int
end

"""
readinput(inputname)

Reads an input file for mcLJ run
and saves the data into the
inputParms struct
"""
function readinput(inputname)
    # Constants
    kB::Float64 = 1.38064852E-23 # [J/K]
    amu::Float64 = 1.66605304E-27 # [kg]

    # Input values for later conversion
    latticePoints::Int = 0
    σ::Float32 = 0. # [Å]
    ϵ::Float64 = 0. # [K]
    atommass::Float64 = 0. # [amu]
    
    # Read the input file
    file = open(inputname, "r")
    lines = readlines(file)
    splittedLines = [split(line) for line in lines]

    # Make a list of field names
    fields = [String(field) for field in fieldnames(inputParms)]

    vars = [] # Array with input variables
    # Loop over fieldnames and fieldtypes and over splitted lines
    for (field, fieldtype) in zip(fields, fieldtypes(inputParms))
        for line in splittedLines
            if length(line) != 0 && field == line[1]      
                if field == "latticePoints"
                    latticePoints = parse(fieldtype, line[3])
                    N = Int(latticePoints^3)
                    append!(vars, latticePoints)
                    append!(vars, N)
                elseif field == "atommass"
                    atommass = parse(fieldtype, line[3])
                    append!(vars, atommass)
                elseif field == "sigma"
                    σ = parse(fieldtype, line[3])
                    append!(vars, σ)
                elseif field == "epsilon"
                    ϵ = parse(fieldtype, line[3])
                    ϵ *= kB
                    append!(vars, ϵ)
                elseif field == "T"
                    T = parse(fieldtype, line[3])
                    β = ϵ/(T*kB)
                    append!(vars, T)  
                    append!(vars, β)
                elseif field == "density"
                    density = parse(fieldtype, line[3])
                    densityRm = (amu*atommass / (2^(1/6) * σ * 1E-10)^3)
                    latticeScaling = (densityRm / density)^(1/3)
                    # Generate PBC box vectors
                    boxSide::Float32 = latticePoints * 2^(1/6) * latticeScaling
                    box::SVector{3, Float32} = [boxSide, boxSide, boxSide]
                    append!(vars, density)
                    append!(vars, densityRm)
                    append!(vars, latticeScaling)
                    append!(vars, [box])
                elseif field == "delta"
                    Δ = parse(fieldtype, line[3])
                    append!(vars, Δ/σ)
                elseif field == "binWidth"
                    binWidth = parse(fieldtype, line[3])
                    append!(vars, binWidth/σ)
                else
                    if fieldtype != String
                        append!(vars, parse(fieldtype, line[3]))
                    else
                        append!(vars, [line[3]])
                    end
                end
            end
        end
    end

    # Save parameters into the inputParms struct
    parameters = inputParms(vars...)
    return(parameters)
end