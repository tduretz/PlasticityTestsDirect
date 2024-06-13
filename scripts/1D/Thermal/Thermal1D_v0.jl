using Plots, SparseArrays, Symbolics, SparseDiffTools, Printf, CSV, DataFrames, IfElse, ExactFieldSolutions
import LinearAlgebra: norm

function Residual!(F, T, T0, k, ρ, Cp, Δx, Δt, x, t)
    ncx = length(F)
    params = (T0=1.0, K=1.0, σ=0.1)
    for i in eachindex(T)
        # West boundary
        if i==1
            sol   = Diffusion1D_Gaussian([x.min-Δx/2.0; t]; params )
            Ta = sol.u
            qxW = -k*(T[i] - Ta)/Δx
        end
        # East boundary
        if i==ncx
            sol   = Diffusion1D_Gaussian([x.min-Δx/2.0; t]; params )
            Ta = sol.u
            qxE = -k*(Ta - T[i])/Δx
        end
        # West flux
        if i>1 # Left flux 
            qxW = -k*(T[i] - T[i-1])/Δx
        end
        # East
        if i<ncx # Right flux
            qxE = -k*(T[i+1] - T[i])/Δx
        end
        # Balance
        F[i] = ρ*Cp*(T[i] - T0[i])/Δt + (qxE - qxW)/Δx 
    end
    return nothing
end

function main()

    # Parameters
    x   = (min=-1.0, max=1.0)
    ncx = 100
    nt  = 1
    Δx  = (x.max-x.min)/ncx
    xc  = LinRange(x.min+Δx/2., x.max-Δx/2., ncx)
    ρ   = 1.0
    Cp  = 1.0
    k   = 1.0
    Δt  = 1e-7
    t   = 0.

    # Allocations
    T   = zeros(ncx)
    T0  = zeros(ncx)
    F   = zeros(ncx)
    δT  = zeros(ncx)
    Ta  = zeros(ncx)

    # Sparsity pattern
    input       = rand(ncx)
    output      = similar(input)
    Res_closed! = (F, T) -> Residual!(F, T, T0, k, ρ, Cp, Δx, Δt, x, t)
    sparsity    = Symbolics.jacobian_sparsity(Res_closed!, output, input)
    J           = Float64.(sparse(sparsity))

    # Makes coloring
    colors      = matrix_colors(J)

    # Initial condition: Evaluate exact initial solution
    params = (T0=1.0, K=1.0, σ=0.1)
    for i in eachindex(T)
        sol   = Diffusion1D_Gaussian([xc[i]; 0]; params )
        T[i] = sol.u
    end
    
    # Time loop
    for it=1:nt
        T0 .= T
        t += Δt 
        @printf("########### Step %06d ###########\n", it)

        for iter=1:5
            @printf("## Iteration %06d ##\n", iter)

            # Residual evaluation: T is found if F = 0
            Residual!(F, T, T0, k, ρ, Cp, Δx, Δt, x, t)
            Res_closed! = (F, T) -> Residual!(F, T, T0, k, ρ, Cp, Δx, Δt, x, t)
            @info norm(F)/ncx

            # Jacobian assembly
            forwarddiff_color_jacobian!(J, Res_closed!, T, colorvec = colors)

            # Solve
            δT   .= .-J\F

            # update
            T    .+= δT
        end
    end

    # Evaluate exact solution
    for i in eachindex(T)
        sol   = Diffusion1D_Gaussian([xc[i]; 0]; params )
        Ta[i] = sol.u
    end

    # error
    @show ϵ = norm(T .- Ta)/ncx

    p = plot(title="error = $(ϵ)", xlabel="x", ylabel="T")
    p = plot!(xc, Ta, label="Exact solution")
    p = scatter!(xc, T, label="numerics")

end

main()