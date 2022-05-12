using Printf
using RandomNumbers
using StaticArrays
using Chemfiles

include("distances.jl")

"""
ljlattice(parameters)

Generates a cubic latice of LJ atoms
separated by scaled Rm distance,
and the periodic box vectors in reduced units
"""
function ljlattice(parameters)
    lattice = [convert(Vector{Float64}, [i, j, k]) 
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
    dr = [parameters.Δ*(rand(rng, Float64) - 0.5), 
          parameters.Δ*(rand(rng, Float64) - 0.5), 
          parameters.Δ*(rand(rng, Float64) - 0.5)]
    
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
    trajFile = "mctraj-p$(idString).xtc"
    pdbFile = "confout-p$(idString).pdb"

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
    writeenergies(E, 0, energyFile, "w")

    # Start writing MC trajectory
    writetraj(conf, parameters, trajFile, 'w')

    # Acceptance counters
    acceptedTotal = 0
    acceptedIntermediate = 0

    # Run MC simulation
    @inbounds @fastmath for i in 1:parameters.steps
        conf, E, accepted, distanceMatrix = mcmove!(conf, parameters, distanceMatrix, E, rng_xor)
        acceptedTotal += accepted
        acceptedIntermediate += accepted

        # Perform MC step adjustment during the equilibration
        if parameters.stepAdjustFreq > 0 && i % parameters.stepAdjustFreq == 0 && i < parameters.Eqsteps
            stepAdjustment!(parameters, acceptedIntermediate)
            acceptedIntermediate = 0
        end

        # Energy output
        if i % parameters.outfreq == 0
            writeenergies(E, i, energyFile, "a")
        end

        # MC trajectory output
        if i % parameters.trajout == 0
            writetraj(conf, parameters, trajFile, 'a')
        end

        # Distance histogram calculation
        if i % parameters.outfreq == 0 && i > parameters.Eqsteps
            hist = hist!(distanceMatrix, hist, parameters)
        end
    end

    # Normalize the histogram to the number of frames
    Nframes = (parameters.steps - parameters.Eqsteps) / parameters.outfreq
    hist /= Nframes

    # Compute the final acceptance ratio
    acceptanceRatio = acceptedTotal / parameters.steps
    
    # Write the final configuration to a PDB file
    writetraj(conf, parameters, pdbFile, 'w')

    return(hist, acceptanceRatio)
end

"""
function stepAdjustment!(parameters, acceptedIntermediate)

MC step length adjustment
"""
function stepAdjustment!(parameters, acceptedIntermediate)
    acceptanceRatio = acceptedIntermediate / parameters.stepAdjustFreq
    #println("Current acceptance ratio = $(round(acceptanceRatio, digits=4))")
    parameters.Δ = acceptanceRatio * parameters.Δ / parameters.targetAR
    #println("New maximum displacement length = $(round((parameters.Δ * parameters.σ), digits=4)) Å")
    return(parameters)
end

"""
function writetraj(conf, parameters, outname, mode='w', atomtype="Ar")

Writes a wrapped configuration into a trajectory file (Depends on Chemfiles)
"""
function writetraj(conf, parameters, outname, mode='w', atomtype="Ar")
    # Create an empty Frame object
    frame = Frame() 
    # Set PBC vectors
    box = parameters.box .* parameters.σ
    boxCenter = box ./ 2
    set_cell!(frame, UnitCell(box))
    # Add wrapped atomic coordinates to the frame
    for i in 1:parameters.N
        wrappedAtomCoords = wrap!(UnitCell(frame), conf[i] .* parameters.σ) .+ boxCenter
        add_atom!(frame, Atom(atomtype), wrappedAtomCoords)
    end
    # Write to file
    Trajectory(outname, mode) do traj
        write(traj, frame)
    end
    return
end

"""
function writexyz(conf, currentStep, parameters, outname, mode="w", atomtype="Ar")

Writes a configuration to an XYZ file (legacy function)
"""
function writexyz(conf, currentStep, parameters, outname, mode="w", atomtype="Ar")
    io = open(outname, mode)
    print(io, parameters.N, "\n")
    print(io, "Step = ", @sprintf("%d", currentStep), "\n")
    for i in 1:parameters.N
        print(io, atomtype, " ")
        for j in 1:3
            print(io, @sprintf("%10.3f", conf[i][j]*parameters.σ), " ")
            if j == 3
                print(io, "\n")
            end
        end
    end
    close(io)
end

"""
function writeenergies(energy, currentStep, outname, mode="w")

Writes the total energy to an output file
"""
function writeenergies(energy, currentStep, outname, mode="w")
    io = open(outname, mode)
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
    rdfNorm = [(V/Npairs) * 1/(4*π*parameters.binWidth*bins[i]^2) for i in 1:length(bins)]
    RDF = hist .* rdfNorm

    # Convert bins to Å
    bins .*= parameters.σ

    # Write the data
    io = open(outname, "w")
    print(io, "# RDF data \n")
    print(io, "# r, Å; g(r); Histogram \n")
    for i in 1:length(hist)
        print(io, @sprintf("%6.3f %12.3f %12.3f", bins[i], RDF[i], hist[i]), "\n")
    end
    close(io)
end

"""
mutable struct inputParms

Fields:
latticePoints: number of LJ lattice points
N: number of particles
μ: atom mass [amu]
σ: σ LJ parameter [Å]
ϵ: ϵ LJ parameter, ϵ/kB [K]
T: temperature [K]
β: ϵ/(kB*T), reciprocal reduced temperature
ρ: target ρ [kg/m3]
ρRm: initial ρ [kg/m3]
latticeScaling: lattice scaling factor
box: box vector, σ
Δ: max displacement [σ]
steps: total number of steps
Eqsteps: equilibration steps
stepAdjustFreq: frequency of MC step adjustment
targetAR: target acceptance ratio
binWidth: histogram bin width [σ]
Nbins: number of histogram bins
trajout: XYZ output frequency
outfreq: output frequency
"""
mutable struct inputParms
    latticePoints::Int
    N::Int 
    μ::Float64
    σ::Float64
    ϵ::Float64
    T::Float64
    β::Float64
    ρ::Float64 
    ρRm::Float64
    latticeScaling::Float64
    box::Vector{Float64}
    Δ::Float64  
    steps::Int
    Eqsteps::Int
    stepAdjustFreq::Int
    targetAR::Float64
    binWidth::Float64
    Nbins::Int
    trajout::Int 
    outfreq::Int
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
    σ::Float64 = 0. # [Å]
    ϵ::Float64 = 0. # [K]
    μ::Float64 = 0. # [amu]
    
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
                elseif field == "μ"
                    μ = parse(fieldtype, line[3])
                    append!(vars, μ)
                elseif field == "σ"
                    σ = parse(fieldtype, line[3])
                    append!(vars, σ)
                elseif field == "ϵ"
                    ϵ = parse(fieldtype, line[3])
                    ϵ *= kB
                    append!(vars, ϵ)
                elseif field == "T"
                    T = parse(fieldtype, line[3])
                    β = ϵ/(T*kB)
                    append!(vars, T)  
                    append!(vars, β)
                elseif field == "ρ"
                    ρ = parse(fieldtype, line[3])
                    ρRm = (amu*μ / (2^(1/6) * σ * 1E-10)^3)
                    latticeScaling = (ρRm / ρ)^(1/3)
                    # Generate PBC box vectors
                    boxSide::Float64 = latticePoints * 2^(1/6) * latticeScaling
                    box = [boxSide, boxSide, boxSide]
                    append!(vars, ρ)
                    append!(vars, ρRm)
                    append!(vars, latticeScaling)
                    append!(vars, [box])
                elseif field == "Δ"
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