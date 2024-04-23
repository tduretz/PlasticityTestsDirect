using Plots, SparseArrays, Symbolics, SparseDiffTools, Printf, CSV, DataFrames, IfElse
import LinearAlgebra: norm

function main()

    params=(
        #---------------#
        K   = 10e6, # K = 3/2*Gv in Vermeer (1990)
        G   = 10e6,
        ν   = 0., 
        c   = 0.0e4,
        ϕ   = 40/180*π,
        ψ   = 10/180*π,
        θt  = 25/180*π,
        ηvp = 0.,
        γ̇xy = 0.00001,
        Δt  = 20,
        nt  = 400,
        law = :MC_Vermeer1990,
        oop = :Vermeer1990,
        pl  = true)

    sc   = (σ = params.G, L = 1.0, t = 1.0/params.γ̇xy)

    ε̇0   = params.γ̇xy/(1/sc.t)
    σxxi = -25e3/sc.σ
    σyyi = -100e3/sc.σ
    σzzi       = 0.5*(σxxi + σyyi)
    Pi         = -(σxxi + σyyi)/2.0
    τxxi       = Pi + σxxi
    τyyi       = Pi + σyyi
    τzzi       = Pi + σzzi
    τxyi       = 0.0
    σ_BC       = (yy = σyyi,)

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
    y    = (min=0.0/sc.L, max=1.0/sc.L)

    V_BC = ( 
        x = ( 
            S = 0.0,
            N = ε̇0*y.max,
        ),
        y = (
            S = 0.0,
            N = 0.0,
        )
    )

    Ncy = 1
    Nt  = params.nt
    Δy  = (y.max-y.min)/Ncy
    Δt  = params.Δt/sc.t
    yc  = LinRange(-y.min+Δy/2, y.max-Δy/2, Ncy  )
    yv  = LinRange(-y.min,      y.max,      Ncy+1)

    rheo = (
        c    = params.c/sc.σ*ones(Ncy+1),
        ψ    = params.ψ     *ones(Ncy+1),
        ϕ    = params.ϕ     *ones(Ncy+1),
        ν    = params.ν     *ones(Ncy+1),
        G    = params.G/sc.σ*ones(Ncy+1),
        K    = params.K/sc.σ*ones(Ncy+1),
    )
    # @. rheo.K = 2/3 .* rheo.G.*(1 .+ rheo.ν) ./ (1 .- 2 .* rheo.ν)
    ε̇xx  =     zeros(Ncy+1)
    ε̇yy  =     zeros(Ncy+1)
    ε̇zz  =     zeros(Ncy+1)
    ε̇xy  =     zeros(Ncy+1)

    ε̇xxt =     zeros(Ncy+1)
    ε̇yyt =     zeros(Ncy+1)
    ε̇zzt =     zeros(Ncy+1)
    τII  =     zeros(Ncy+1)
    ε̇xxp =     zeros(Ncy+1)
    ε̇yyp =     zeros(Ncy+1)
    ε̇zzp =     zeros(Ncy+1)
    ε̇xyp =     zeros(Ncy+1)
    f    =     zeros(Ncy+1)
    λ̇_pl =     zeros(Ncy+1)
    λ̇    =     zeros(Ncy+1)


    divV =     zeros(Ncy+1)
    divVp=     zeros(Ncy+1)
    τxx  = τxxi*ones(Ncy+1)
    τyy  = τyyi*ones(Ncy+1)
    τzz  = τzzi*ones(Ncy+1)
    τxy  = τxyi*ones(Ncy+1)
    P    =   Pi*ones(Ncy+1)
    fc   =     zeros(Ncy+1)
    Pc   =     zeros(Ncy+1)
    Vx   = collect(ε̇0.*yc)
    Vy   = zeros(Ncy+0)

    τxx0 = zeros(Ncy+1)
    τyy0 = zeros(Ncy+1)
    τzz0 = zeros(Ncy+1)
    τxy0 = zeros(Ncy+1)
    P0   = zeros(Ncy+1)

    N    = 2*(Ncy+0) + Ncy+1
    x    = zeros(N)
    iV   = (x=Ncy, y=2*(Ncy))  
    NumV = (x=1:Ncy, y=Ncy+1:2*Ncy)
    fric = zeros(Nt)
    
    for it=1:Nt

        @printf("########### Step %06d ###########\n", it)
        τxx0 .= τxx
        τyy0 .= τyy
        τzz0 .= τzz
        τxy0 .= τxy
        P0   .= P
        x[1:iV.x]      .= Vx
        x[iV.x+1:iV.y] .= Vy
        x[iV.y+1:end]  .= P

        ε̇yyp    .= 0.0
        divVp   .= 0.0
        σi      = σ_BC

        G = rheo.G
        K = rheo.K

        @. ε̇xxt   = 0.
        @. ε̇yyt   = (4.0 * G .* Δt .* ε̇yyp - K .* Δt .* ε̇xxt + 2.0 * K .* Δt .* divVp + 2.0 * P0 + 2.0 * σi.yy - 2.0 * τyy0) ./ (Δt .* (4.0 * G + 3.0 * K))
        @. ε̇zzt   = 1/2*(ε̇xxt + ε̇yyt)
        @. divV   = ε̇xxt + ε̇yyt + ε̇zzt

        @. ε̇xy = ε̇0/2
        @. ε̇xx = ε̇xxt - 1/3*divV
        @. ε̇yy = ε̇yyt - 1/3*divV
        @. ε̇zz = ε̇zzt - 1/3*divV
    
        @. τxx   = 2*G*Δt*ε̇xx + τxx0
        @. τyy   = 2*G*Δt*ε̇yy + τyy0
        @. τzz   = 2*G*Δt*ε̇zz + τzz0
        @. τxy   = 2*G*Δt*ε̇xy + τxy0

        fact= 3/2

        @. P = P0 - K*Δt *divV

        @. τII   = sqrt(1/2*(τxx^2 + τyy^2 + τzz^2) + τxy^2)
        @. Pc    = P
        @. f     = τII - rheo.c*cos(rheo.ϕ) - Pc*sin(rheo.ϕ)

        @. λ̇_pl  = f / (G*Δt + K*Δt*sin(rheo.ϕ)*sin(rheo.ψ)*fact)
        @. λ̇ = λ̇_pl

        @. λ̇     = IfElse.ifelse(f>0.0, λ̇_pl, 0.0)
        @. ε̇xxp  = λ̇*(τxx)/2/τII
        @. ε̇yyp  = λ̇*(τyy)/2/τII
        # @.  ε̇zzp  = λ̇*(τxx+τyy)/2/2/τII
        @. ε̇zzp  = λ̇*(τzz)/2/τII
        @. ε̇xyp  = λ̇*(τxy)/1/τII
        @. τxx   = 2*G*Δt*(ε̇xx - ε̇xxp  ) + τxx0
        @. τyy   = 2*G*Δt*(ε̇yy - ε̇yyp  ) + τyy0
        @. τzz   = 2*G*Δt*(ε̇zz - ε̇zzp  ) + τzz0
        @. τxy   = 2*G*Δt*(ε̇xy - ε̇xyp/2) + τxy0
        @. Pc    = P + λ̇*K*Δt*sin(rheo.ψ)*fact
        @. τII   = sqrt(1/2*(τxx^2 + τyy^2 + τzz^2) + τxy^2)
        @. fc    = τII - rheo.c*cos(rheo.ϕ) - Pc*sin(rheo.ϕ)
        
        P .= Pc
        
        @printf("σyy_BC = %2.4e, max(fc) = %2.4e\n", (τyy[end]-P[end])*sc.σ, maximum(fc)*sc.σ)

        fric[it] = -τxy[end]/(τyy[end]-P[end])

        nout = 100
        if mod(it, nout)==0
            p1 = plot()
            p1 = plot!(Vx, yc)
            p2 = scatter(V90.x, V90.y)
            p2 = plot!((1:it)*ε̇0*Δt*100, fric[1:it])
            p3 = scatter(V90.x, V90.y)
            p3 = plot!((1:it)*ε̇0*Δt*100, fric[1:it])
            p4 = scatter(V90.x, V90.y)
            p4 = plot!((1:it)*ε̇0*Δt*100, fric[1:it])
            display(plot(p1,p2,p3,p4))
        end
    end 
end

main()