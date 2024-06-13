using Plots, SparseArrays, Symbolics, SparseDiffTools, Printf, CSV, DataFrames, IfElse, ExactFieldSolutions
import LinearAlgebra: norm
import Statistics: mean

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
            sol   = Diffusion1D_Gaussian([x.min+Δx/2.0; t]; params )
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

function main(ncx, nt)

    # Parameters
    x   = (min=-1., max=1.)
    Δx  = (x.max-x.min)/ncx
    xc  = LinRange(x.min+Δx/2., x.max-Δx/2., ncx)
    ρ   = 1.0
    Cp  = 1.0
    k   = 1.0
    t   = 0.
    Δt  = 0.01 ./ nt

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
        sol   = Diffusion1D_Gaussian([xc[i]; t]; params )
        T[i] = sol.u
    end
    
    # Time loop
    for it=1:nt
        T0 .= T
        t += Δt 
        @printf("########### Step %06d ###########\n", it)

        for iter=1:10

            # Residual evaluation: T is found if F = 0
            Residual!(F, T, T0, k, ρ, Cp, Δx, Δt, x, t)
            Res_closed! = (F, T) -> Residual!(F, T, T0, k, ρ, Cp, Δx, Δt, x, t)
            r = norm(F)/ncx
            @printf("## Iteration %06d: r = %1.2e ##\n", iter, r)
            if r < 1e-10 break end
                
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
        sol   = Diffusion1D_Gaussian([xc[i]; t]; params )
        Ta[i] = sol.u
    end
    
    # Error
    return mean(abs.(T .- Ta))
end

function ConvergenceAnalysis()

    # Time
    ncx = 1000
    Nt  = [10, 20, 40, 80, 160, 320, 640]  
    Δtv = 1e-2./[ 40, 80, 160, 320, 640]  
    ϵt  = zeros(size(Nt))
    for i in eachindex(Nt)
        ϵt[i] = main(ncx, Nt[i])
    end
    @show ϵt

     # Space
     nt  = 5000
     Ncx = [ 40, 80, 160, 320, 640]  
     Δxv = 2.0./[ 40, 80, 160, 320, 640]  
     ϵx  = zeros(size(Ncx))
     for i in eachindex(Ncx)
        ϵx[i] = main(Ncx[i], nt)
     end
     @show ϵx

    p1 = plot(xlabel="log10(1/Δx)", ylabel="log10(ϵx)")
    p1 = scatter!( log10.( 1.0./Δxv ), log10.(ϵx), label="ϵ")
    p1 = plot!( log10.( 1.0./Δxv ), log10.(ϵx[1]) .- 1.0* ( log10.( 1.0./Δxv ) .- log10.( 1.0./Δxv[1] ) ), label="O1"  ) 
    p1 = plot!( log10.( 1.0./Δxv ), log10.(ϵx[1]) .- 2.0* ( log10.( 1.0./Δxv ) .- log10.( 1.0./Δxv[1] ) ), label="O2"  ) 

    p2 = plot(xlabel="log10(1/Δt)", ylabel="log10(ϵt)")
    p2 = scatter!( log10.( 1.0./Δtv ), log10.(ϵt), label="ϵ")
    p2 = plot!( log10.( 1.0./Δtv ), log10.(ϵt[1]) .- 1.0* ( log10.( 1.0./Δtv ) .- log10.( 1.0./Δtv[1] ) ), label="O1" ) 
    display(plot(p1, p2))
end

ConvergenceAnalysis()