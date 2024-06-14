using Plots, SparseArrays, Symbolics, SparseDiffTools, Printf, CSV, DataFrames, IfElse
import LinearAlgebra: norm

# Commented version 😄

include("Functions.jl")

# Main action
function main(σ0)

    params = (
        #----------------------#
        K   = 10e6,            # Bulk modulus [Pa]
        G   = 10e6,            # Shear modulus [Pa]
        lc  = 0.01,            # Cosserat lengthscale [m] 
        ν   = 0.,              # Poisson ratio [-] (UNUSED since K is defined above)
        c   = 0.0e4,           # C ohesion [Pa]
        ϕ   = 40/180*π,        # Friction angle [∘]
        ψ   = 10/180*π,        # Dilation angle [∘]
        θt  = 25/180*π,        # angle for smoothing Mohr-Coulomb [∘] (UNUSED)
        ρ   = 2000.,           # Density [kg/m3]
        ηvp = 0e7,             # Viscoplasticity: Kelvin element viscosity [Pa.s] 
        γ̇xy = 0.00001,         # Applied shear strain rate [1/s]
        Δt  = 1,               # Time step [s]
        nt  = 250*4*8,         # Number of time steps
        ncy = 30,              # Number of grid cells
        law = :MC_Vermeer1990, # UNUSED
        oop = :Vermeer1990,    # UNUSED
        pl  = true)            # Activates plasticity

    # Internal scaling
    sc   = (σ = params.G, L = 1.0, t = 1.0/params.γ̇xy) 

    # Boundary shear strain rate scaling
    ε̇0   = params.γ̇xy/(1/sc.t)
    # Values of initial stresses and scaling
    σxxi = σ0.xx/sc.σ
    σyyi = σ0.yy/sc.σ
    σzzi = 0.5*(σxxi + σyyi)
    Pi   = -(σxxi + σyyi)/2.0
    τxxi = Pi + σxxi
    τyyi = Pi + σyyi
    τzzi = Pi + σzzi
    τxyi = 0.0
    σ_BC = (yy = σyyi,)

    # Data from Vermeer (1990) for reference
    if σxxi>σyyi
        V90 = (
            x = convert(Array, CSV.read("./data/Vermeer1990_Friction_CaseA.csv", DataFrame, header =[:x, :y])[:,1]),
            y = convert(Array, CSV.read("./data/Vermeer1990_Friction_CaseA.csv", DataFrame, header =[:x, :y])[:,2]),
        )
    else
        V90 = (
            x = convert(Array, CSV.read("./data/Vermeer1990_Friction_CaseB.csv", DataFrame, header =[:x, :y])[:,1]),
            y = convert(Array, CSV.read("./data/Vermeer1990_Friction_CaseB.csv", DataFrame, header =[:x, :y])[:,2]),
        )
    end
    y    = (min=-0.5/sc.L, max=0.5/sc.L)

    # Structure that stores boundary condition information
    V_BC = ( 
        x = ( 
            S = ε̇0*y.min,
            N = ε̇0*y.max,
        ),
        y = (
            S = 0.0,
            N = 0.0,
        )
    )
  
    # Spatial resolution, grid spacing and coordinate arrays
    Ncy = params.ncy
    Nt  = params.nt
    Δy  = (y.max-y.min)/Ncy
    Δt  = params.Δt/sc.t
    yc  = LinRange(y.min+Δy/2, y.max-Δy/2, Ncy  )
    yv  = LinRange(y.min,      y.max,      Ncy+1)

    # Tuple that contains arrays of rheological parameters 
    rheo = (
        c    = params.c/sc.σ*ones(Ncy+1).+250/sc.σ,
        ψ    = params.ψ      *ones(Ncy+1),
        ϕ    = params.ϕ      *ones(Ncy+1),
        ν    = params.ν      *ones(Ncy+1),
        G    = params.G/sc.σ *ones(Ncy+1),
        K    = params.K/sc.σ *ones(Ncy+1),
        ηvp  = params.ηvp    *ones(Ncy+1)/(sc.σ*sc.t),
        lc   = params.lc/sc.L*ones(Ncy+1),
    )
    rheo.c[Int64(ceil(Ncy/2)+1)] = params.c/sc.σ

    # Arrays that are necessary to store the solutions of the model
    τxx  = τxxi*ones(Ncy+1)
    τyy  = τyyi*ones(Ncy+1)
    τzz  = τzzi*ones(Ncy+1)
    τxy  = τxyi*ones(Ncy+1)
    P    =   Pi*ones(Ncy+1)
    Rz   =    zeros(Ncy+1)
    myz  =    zeros(Ncy+1)
    fc   = zeros(Ncy+1)
    pl   = zeros(Ncy+1)
    Pc   = zeros(Ncy+1)
    Vx   = collect(ε̇0.*yc)
    Vy   = zeros(Ncy+0)
    Vx0  = collect(ε̇0.*yc)
    Vy0  = zeros(Ncy+0)
    ω̇z   =    zeros(Ncy+0)
    ρ    = params.ρ/(sc.L*sc.σ*sc.t^2) * ones(Ncy+0)
    τxx0 = zeros(Ncy+1)
    τyy0 = zeros(Ncy+1)
    τzz0 = zeros(Ncy+1)
    τxy0 = zeros(Ncy+1)
    Rz0  =    zeros(Ncy+1)
    myz0 =    zeros(Ncy+1)
    εyyt = zeros(Ncy+1)
    P0   = zeros(Ncy+1)

    # Arrays and structures necessary to store/build the system of equations
    N    = 3*(Ncy+0) + Ncy+1
    F    = zeros(N)
    x    = zeros(N)
    ind   = (x=Ncy, y=2*(Ncy), p=2*(Ncy)+(Ncy+1))  
    NumV = (x=1:Ncy, y=Ncy+1:2*Ncy)
    
    # Sparsity pattern
    input       = rand(N)
    output      = similar(input)
    Res_closed! = (F,x) -> Res!(F, x, P, P0, τxx0, τyy0, τzz0, τxy0, Rz0, myz0, ρ, Vx0, Vy0, V_BC, σ_BC, rheo, ind, NumV, Δy, Δt )
    sparsity    = Symbolics.jacobian_sparsity(Res_closed!, output, input)
    J           = Float64.(sparse(sparsity))

    # Makes coloring
    colors   = matrix_colors(J) 

    # Globalisation
    LS = (
        α = [0.01 0.05 0.1 0.25 0.5 0.75 1.0], 
        F = zeros(7),
    )

    # Probes will be used to monitor solutions (e.g. stress at ONE POINT as a function of time)
    probes = (
        fric = zeros(Nt),
        σxx  = zeros(Nt),
        θσ3  = zeros(Nt),
        εyy  = zeros(Nt),
    )

    # Maps stress stores some fields as function of time (e.g. velocity space-time map)
    maps = ( 
        Vx = zeros(Nt,Ncy+0),
        Vy = zeros(Nt,Ncy+0),
        P  = zeros(Nt,Ncy+1),
    )
    
    # Time integration loop 
    for it=1:Nt

        @printf("########### Step %06d ###########\n", it)

        # From previous time step
        τxx0 .= τxx
        τyy0 .= τyy
        τzz0 .= τzz
        τxy0 .= τxy
        P0   .= P
        Vx0  .= Vx 
        Vy0  .= Vy
        myz0 .= myz
        Rz0  .= Rz

        # Vx  .= collect(ε̇0.*yc) 
        # Vy  .= 0*Vy

        # Populate global solution array
        x[1:ind.x]       .= Vx
        x[ind.x+1:ind.y] .= Vy
        x[ind.y+1:ind.p] .= P
        x[ind.p+1:end  ] .= ω̇z

        ϵglob = 1e-13

        # Newton iterations
        for iter=1:100

            # Residual
            Res!(F, x, P, P0, τxx0, τyy0, τzz0, τxy0, Rz0, myz0, ρ, Vx0, Vy0, V_BC, σ_BC, rheo, ind, NumV, Δy, Δt )
            @show (iter, norm(F))
            if norm(F)/length(F)<ϵglob break end

            # Jacobian assembly
            forwarddiff_color_jacobian!(J, Res_closed!, x, colorvec = colors)

            # Solve
            δx    = J\F

            # Line search and update of global solution array
            i_opt = LineSearch(Res_closed!, F, x, δx, LS) 
            x   .-= LS.α[i_opt]*δx
           
        end
        if norm(F)/length(F)>ϵglob error("Diverged!") end

        # Extract fields from global solution array
        Vx .= x[1:ind.x]
        Vy .= x[ind.x+1:ind.y]
        P  .= x[ind.y+1:ind.p]
        ω̇z .= x[ind.p+1:end  ]

        # Compute stress for postprocessing
        ComputeStress!(τxx, τyy, τzz, τxy, Pc, Rz, myz, fc, pl, εyyt, x, P, P0, τxx0, τyy0, τzz0, τxy0, Rz0, myz0, V_BC, σ_BC, rheo, ind, NumV, Δy, Δt )
        P .= Pc #!!!!!!!!!

        # Postprocessing
        @printf("σyy_BC = %2.4e, max(f) = %2.4e\n", (τyy[end]-P[end])*sc.σ, maximum(fc)*sc.σ)
        probes.fric[it] = -τxy[end]/(τyy[end]-P[end])
        probes.εyy[it]  = εyyt[end]
        maps.Vx[it,:]  .= Vx
        maps.Vy[it,:]  .= Vy
        maps.P[it,:]   .= P

        # Visualisation
        nout = 1000
        if mod(it, nout)==0 || it==1
            # p1 = plot()
            # p1 = plot!(Vx, yc, label="Vx")
            # p1 = plot!(Vy, yc, label="Vy")
            # p1 = plot!(P,  yv, label="P")
            p1 = heatmap((1:Nt)*ε̇0*Δt*100, yc, maps.Vx[1:Nt,:]', title="Vx", xlabel="strain", ylabel="y") #, clim=(0,1)
            # p2 = heatmap((1:Nt)*ε̇0*Δt*100, yc, maps.Vy[1:Nt,:]', title="Vy", xlabel="strain", ylabel="y") #, clim=(0,1/2)
            # p3 = heatmap((1:Nt)*ε̇0*Δt*100, yv, maps.P[1:Nt,:]',  title="P",  xlabel="strain", ylabel="y")
            p2 = plot(ω̇z, yc)
            p3 = scatter(V90.x, V90.y)
            p3 = plot!((1:it)*ε̇0*Δt*100, probes.fric[1:it])
            # p4 = plot((1:it)*ε̇0*Δt*100, probes.εyy[1:it]*100)
            p4 = plot(εyyt[1:end-1], yv[1:end-1])
            p4 = scatter!(εyyt[pl.==1], yv[pl.==1])
            display(plot(p1,p2,p3,p4))
        end
    end 
   
end

# σ0 = (xx= -25e3, yy=-100e3) # Case A
σ0 = (xx=-400e3, yy=-100e3) # Case B 
main(σ0)