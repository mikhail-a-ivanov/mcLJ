using Dates
using LinearAlgebra
using Distributed
using Statistics
@everywhere include("src/base.jl")
BLAS.set_num_threads(1)

"""
Main function for running MC simulation
"""
function main()
    # Total number of workers
    np = length(workers())

    # Start the timer and read the input file
    startTime = Dates.now()

    # Read input data
    inputname = ARGS[1]
    parameters = readinput(inputname)

    println("Running MC simulation on $(np) rank(s)...\n")
    println("The simulated system contains $(parameters.N) $(parameters.atomname) atoms")
    println("Total number of steps: $(parameters.steps * np / 1E6)M")
    println("Number of equilibration steps per rank: $(parameters.Eqsteps / 1E6)M")
    println("Trajectory output every $(parameters.trajout / 1E6)M steps")
    println("Box vectors: ", [round(boxSide*parameters.σ, digits=3) for boxSide in parameters.box], " Å")
    println("Density: $(parameters.ρ) kg/m3")
    println("Temperature: $(parameters.T) K")
    println("Starting at: ", startTime)

    # Run MC simulation
    inputs = [parameters for worker in workers()]
    outputs = pmap(mcrun, inputs)

    # Write the final RDF        
    meanHist = mean([output[1] for output in outputs])
    rdfName = "rdf-mean-p$(np).dat"
    writeRDF(rdfName, meanHist, parameters)
    
    # Report the mean acceptance ratio
    meanAcceptanceRatio = mean([output[2] for output in outputs])
    println("Mean acceptance ratio = ", round(meanAcceptanceRatio, digits=3))
    
    # Stop the timer
    stopTime = Dates.now()
    wallTime = Dates.canonicalize(stopTime - startTime)
    println("Stopping at: ", stopTime, "\n")
    println("Walltime: ", wallTime)
end

"""
Run the main() function
"""

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
