using Plots, SparseArrays, Symbolics, SparseDiffTools, Printf, CSV, DataFrames, IfElse
import LinearAlgebra: norm

function PlasticDilationFactor(dt, B, K_d, alpha, eta_phi, phi)
    mt = K_d .* alpha .* dt .* eta_phi .* (B - 1) ./ (-2 * B .* K_d .* alpha .* dt + B .* K_d .* dt + B .* alpha .^ 2 .* eta_phi .* phi - B .* alpha .^ 2 .* eta_phi + K_d .* alpha .* dt - alpha .* eta_phi .* phi + alpha .* eta_phi)
    mf = B .* K_d .* dt .* eta_phi .* (1 - alpha) ./ (-2 * B .* K_d .* alpha .* dt + B .* K_d .* dt + B .* alpha .^ 2 .* eta_phi .* phi - B .* alpha .^ 2 .* eta_phi + K_d .* alpha .* dt - alpha .* eta_phi .* phi + alpha .* eta_phi)
    return mf, mt
end

function NeumannVerticalVelocity(σ_BC, VyS, G, K, τyy0, Pt0, Δy, Δt)
    return (2.0 * G .* VyS .* Δt + 3.0 * K .* VyS .* Δt + 2.0 * Pt0 .* Δy + 2.0 * Δy .* σ_BC.yy - 2.0 * Δy .* τyy0) ./ (Δt .* (2.0 * G + 3.0 * K))
end

function LocalStress(VxN, VxS, VyN, VyS, Pt, Pf, ϕ_poro, τxx0, τyy0, τzz0, τxy0, Pt0, Pf0, G, ϕ, ψ, c, ηvp, B, Kd, α, ηϕ, Δy, Δt)
    
    # Total strain rate
    ε̇xxt  = 0.0
    ε̇yyt  = (VyN - VyS)/Δy
    ε̇zzt  = 1/2*(ε̇xxt + ε̇yyt)
    ε̇xyt  = 1/2*( VxN -  VxS)/Δy
    divV  = ε̇xxt + ε̇yyt + ε̇zzt

    # Deviatoric strain rate
    ε̇xx   = ε̇xxt - 1/3*divV
    ε̇yy   = ε̇yyt - 1/3*divV
    ε̇zz   = ε̇zzt - 1/3*divV
    ε̇xy   = ε̇xyt

    # Deviatoric stress
    τxx   =      2*G*Δt*ε̇xx + τxx0
    τyy   =      2*G*Δt*ε̇yy + τyy0
    τzz   =      2*G*Δt*ε̇zz + τzz0
    τxy   =      2*G*Δt*ε̇xy + τxy0

    # # Yield function
    τII   = sqrt(1/2*(τxx^2 + τyy^2 + τzz^2 + τxy^2))
    Ptc   = Pt
    Pfc   = Pf
    f     = τII - c*cos(ϕ) - (Ptc - Pfc)*sin(ϕ)
    fc    = f
    pl    = 0.
    mt, mf = PlasticDilationFactor(Δt, B, Kd, α, ηϕ, ϕ_poro)

    # Return mapping
    fc    = f
    λ̇_pl  = f / (G*Δt + (mt-mf)*sin(ϕ)*sin(ψ) + ηvp)
    λ̇     = IfElse.ifelse(f>=0.0, λ̇_pl, 0.0)
    pl    = IfElse.ifelse(f>=0.0, 1.0, 0.0)
    ε̇xxp  = λ̇*(τxx)/2/τII
    ε̇yyp  = λ̇*(τyy)/2/τII
    ε̇zzp  = λ̇*(τzz)/2/τII
    ε̇xyp  = λ̇*(τxy)/1/τII
    ∇vp   = sin(ψ)*λ̇
    τxx   =      2*G*Δt*(ε̇xx - ε̇xxp  ) + τxx0
    τyy   =      2*G*Δt*(ε̇yy - ε̇yyp  ) + τyy0
    τzz   =      2*G*Δt*(ε̇zz - ε̇zzp  ) + τzz0
    τxy   =      2*G*Δt*(ε̇xy - ε̇xyp/2) + τxy0
    Ptc   = Pt + ∇vp * mt
    Pfc   = Pf + ∇vp * mf
    τII   = sqrt(1/2*(τxx^2 + τyy^2 + τzz^2 + τxy^2))
    fc    = τII - c*cos(ϕ) - (Ptc - Pfc)*sin(ϕ) - ηvp*λ̇
    # @show f, fc 

    return τxx, τyy, τzz, τxy, Ptc, Pfc, fc, pl, ε̇yyt
end

function ComputeStress!(τxx, τyy, τzz, τxy, qDy, Ptc, Pfc, ϕc, Fc, Pl, εyyt, x, τxx0, τyy0, τzz0, τxy0, Pt0, Pf0, V_BC, σ_BC, rheo, ind, Δy, Δt )

     # Loop over all centroids
     for i in eachindex(qDy)
        iPf = ind.pt+i
        iϕ  = ind.pf+i 
        if i==1 # Special case for bottom (South) vertex
            PfS =  x[iPf]
            PfN =  x[iPf]
        elseif i==length(qDy) # Special case for top (North) vertex
            PfS =  x[iPf]
            PfN =  x[iPf]
        else # General case
            PfS = x[iPf-1]
            PfN = x[iPf]
        end
        k_ηf   = rheo.k[i]/rheo.ηf[i]
        qDy[i] = -k_ηf*(PfN - PfS)/Δy
    end

    # Loop over all stress nodes
    for i in eachindex(Ptc)
        ix  = i 
        iy  = ind.x+i 
        iPt = ind.y+i
        iPf = ind.pt+i
        iϕ  = ind.pf+i
        VxS, VxN = 0., 0. 
        VyS, VyN = 0., 0. 
        pt   = x[iPt] 
        pf   = x[iPf]
        ϕ    = x[iϕ]
        if i==1 # Special case for bottom (South) vertex
            VxS = 2*V_BC.x.S - x[ix]
            VxN = x[ix]
            VyS = 2*V_BC.y.S - x[iy]
            VyN = x[iy]
        elseif i==length(Ptc) # Special case for top (North) vertex
            VxS = x[ix-1]
            VxN = 2*V_BC.x.N - x[ix-1]
            VyS = x[iy-1]
            VyN = 2*V_BC.y.N - x[iy-1] 
            # VyN = NeumannVerticalVelocity(σ_BC, x[iy-1], rheo.G[end], rheo.K[end], τyy0[end], Pt0[end], Δy, Δt)
        else # General case
            VxS = x[ix-1]
            VxN = x[ix]
            VyS = x[iy-1]
            VyN = x[iy]
        end
        # Compute stress and corrected pressure 
        τ11, τ22, τ33, τ21, ptc, pfc, fc, pl, ε̇yyt = LocalStress( VxN, VxS, VyN, VyS, pt, pf, ϕ, τxx0[i], τyy0[i], τzz0[i], τxy0[i], Pt0[i], Pf0[i], rheo.G[i], rheo.ϕ[i], rheo.ψ[i], rheo.c[i], rheo.ηvp[i], rheo.B[i], rheo.Kd[i], rheo.α[i], rheo.ηϕ[i], Δy, Δt)
        τxx[i]   = τ11
        τyy[i]   = τ22
        τzz[i]   = τ33
        τxy[i]   = τ21
        Ptc[i]   = ptc
        Pfc[i]   = pfc
        Fc[i]    = fc
        Pl[i]    = pl 
        εyyt[i] += ε̇yyt*Δt
    end
end

function Res!(F, x, ϕ0, Pt0, Pf0, τxx0, τyy0, τzz0, τxy0, ρ, Vx0, Vy0, V_BC, σ_BC, rheo, ind, NumV, Δy, Δt )
    # Loop over all velocity nodes
    for i=1:length(NumV.x)
        ix   = i
        iy   = ind.x+i
        iPt  = ind.y+i
        iPf  = ind.pt+i
        iϕ   = ind.pf+i
        VxC, VxS, VxN = x[ix],  0., 0.
        VyC, VyS, VyN = x[iy],  0., 0.
        PtS     = x[iPt]
        PtN     = x[iPt+1]
        PfS     = x[iPf]
        PfN     = x[iPf+1]
        ϕS      = x[iϕ+1]
        ϕN      = x[iϕ+1]
        if i==1
            VxS = 2*V_BC.x.S - VxC
            VyS = 2*V_BC.y.S - VyC
        else
            VxS = x[ix-1]
            VyS = x[iy-1]
        end
        if i==ind.x
            VxN = 2*V_BC.x.N - VxC
            VyN = 2*V_BC.y.N - VyC 
            # VyN = NeumannVerticalVelocity(σ_BC, VyC, rheo.G[end], rheo.K[end], τyy0[end], Pt0[end], Δy, Δt)
        else
            VxN = x[ix+1]
            VyN = x[iy+1]
        end
        τxxN, τyyN, τzzN, τxyN, PtN, PfN = LocalStress( VxN, VxC, VyN, VyC, PtN, PfN, ϕN, τxx0[i+1], τyy0[i+1], τzz0[i+1], τxy0[i+1], Pt0[i+1], Pf0[i+1], rheo.G[i+1], rheo.ϕ[i+1], rheo.ψ[i+1], rheo.c[i+1], rheo.ηvp[i+1], rheo.B[i+1], rheo.Kd[i+1], rheo.α[i+1], rheo.ηϕ[i+1], Δy, Δt)
        τxxS, τyyS, τzzS, τxyS, PtS, PfS = LocalStress( VxC, VxS, VyC, VyS, PtS, PfS, ϕS, τxx0[i],   τyy0[i],   τzz0[i],   τxy0[i],   Pt0[i],   Pf0[i],   rheo.G[i],   rheo.ϕ[i],   rheo.ψ[i],   rheo.c[i],   rheo.ηvp[i],   rheo.B[i],   rheo.Kd[i],   rheo.α[i],   rheo.ηϕ[i],   Δy, Δt) 
        F[ix]  = (τxyN - τxyS)/Δy - ρ[i]*(VxC-Vx0[i])/Δt
        F[iy]  = (τyyN - τyyS)/Δy - (PtN - PtS)/Δy - ρ[i]*(VyC-Vy0[i])/Δt
    end
    # Loop over all stress nodes
    for i=1:length(NumV.x)+1
        iy  = ind.x+i 
        iPt = ind.y+i
        iPf = ind.pt+i
        iϕ  = ind.pf+i 
        VyS, VyN = 0., 0.
        Pt = x[iPt]
        Pf, PfS, PfN = x[iPf], 0., 0.
        k_ηfS, k_ηfN = 0., 0.
        ϕ  = x[iϕ]  
        if i==1
            VyS = 2*V_BC.y.S - x[iy]
            VyN = x[iy]
            k_ηfN = rheo.k[i]/rheo.ηf[i]
            PfS   = x[iPf+1]
            PfN   = x[iPf+1]
        elseif i==length(NumV.x)+1
            VyS = x[iy-1]
            VyN = 2*V_BC.y.N - x[iy-1] 
            k_ηfS = rheo.k[i-1]/rheo.ηf[i-1]
            PfS   = x[iPf-1]
            PfN   = x[iPf-1]
            # VyN = NeumannVerticalVelocity(σ_BC, x[iy-1], rheo.G[end], rheo.K[end], τyy0[end], Pt0[end], Δy, Δt)
        else
            VyS = x[iy-1]
            VyN = x[iy]
            PfS = x[iPf-1]
            PfN = x[iPf+1]
            k_ηfS = rheo.k[i-1]/rheo.ηf[i-1]
            k_ηfN = rheo.k[i]/rheo.ηf[i]
        end
        qDS   = -k_ηfS * (Pf  - PfS)/Δy
        qDN   = -k_ηfN * (PfN - Pf )/Δy
        divV  =  (VyN - VyS)/Δy
        divqD =  (qDN - qDS)/Δy
        F[iPt] =  - ( divV  + (Pt - Pf)/rheo.ηϕ[i]/(1-ϕ) +       1.0/rheo.Kd[i]*((Pt -Pt0[i])/Δt -     rheo.α[i]*(Pf -Pf0[i])/Δt) )
        F[iPf] =  - ( divqD - (Pt - Pf)/rheo.ηϕ[i]/(1-ϕ) - rheo.α[i]/rheo.Kd[i]*((Pt -Pt0[i])/Δt - 1.0/rheo.B[i]*(Pf -Pf0[i])/Δt) )
        F[iϕ]  =  - ( (ϕ - ϕ0[i])/Δt - ( ((Pf -Pf0[i])-(Pt -Pt0[i]))/Δt/(rheo.G[i]/ϕ) + (Pf -Pt )/rheo.ηϕ[i]) )
    end
end

function LineSearch(Res_closed!, F, x, δx, LS)
    for i=1:length(LS.α)
        Res_closed!(F, x.-LS.α[i].*δx)
        LS.F[i] = norm(F)
    end
    v, i_opt = findmin(LS.F)
    return i_opt
end

function main(σ0)

    params = (
        #---------------#
        Ks  = 10e6,#6.6666666667e6, # K = 3/2*Gv in Vermeer (1990)
        Kf  = 10e7,#6.6666666667e6, # K = 3/2*Gv in Vermeer (1990)
        G   = 10e6,
        lc  = 0.01,  
        c   = 0.0e5,
        ϕ   = 40/180*π,
        ψ   = 0.00/180*π,
        θt  = 25/180*π,
        ρ   = 2000.,
        ϕi  = 1e-2,
        k   = 1e-12/1000,
        ηf  = 1e-1,
        η   = 1e70,
        ηvp = 2e7/5,
        γ̇xy = 0.00001,
        Δt  = 1/1,
        nt  = 250*4*8*1,
        law = :MC_Vermeer1990,
        oop = :Vermeer1990,
        pl  = true)

    sc   = (σ = params.G, L = 1.0, t = 1.0/params.γ̇xy)

    ϕi   = params.ϕi
    ε̇0   = params.γ̇xy/(1/sc.t)
    σxxi = σ0.xx/sc.σ
    σyyi = σ0.yy/sc.σ
    σzzi = 0.5*(σxxi + σyyi)
    Pi   = -(σxxi + σyyi)/2.0
    τxxi = Pi + σxxi
    τyyi = Pi + σyyi
    τzzi = Pi + σzzi
    τxyi = 0.0
    σ_BC = (yy = σyyi,)

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
  
    Ncy = 50
    Nt  = params.nt
    Δy  = (y.max-y.min)/Ncy
    Δt  = params.Δt/sc.t
    yc  = LinRange(y.min+Δy/2, y.max-Δy/2, Ncy  )
    yv  = LinRange(y.min,      y.max,      Ncy+1)

    rheo = (
        c    = params.c/sc.σ*ones(Ncy+1).+250/sc.σ,
        # c    = params.c/sc.σ*ones(Ncy+1).+0.001/sc.σ,
        # c    = params.c/sc.σ*ones(Ncy+1),
        ψ    =  params.ψ      *ones(Ncy+1),
        ϕ    =  params.ϕ      *ones(Ncy+1),
        G    =  params.G/sc.σ *ones(Ncy+1),
        Ks   = params.Ks/sc.σ *ones(Ncy+1),
        Kf   = params.Kf/sc.σ *ones(Ncy+1),
        Kd   = params.Ks/sc.σ *ones(Ncy+1),
        B    =                zeros(Ncy+1),
        α    =                zeros(Ncy+1),
        k    = params.k/sc.L^2       *ones(Ncy+0),
        ηf   = params.ηf /(sc.σ*sc.t)*ones(Ncy+0),
        η    = params.η  /(sc.σ*sc.t)*ones(Ncy+1),
        ηϕ   = params.η  /(sc.σ*sc.t)*ones(Ncy+1),
        ηvp  = params.ηvp/(sc.σ*sc.t)*ones(Ncy+1),
        lc   = params.lc/sc.L*ones(Ncy+1),
    )

    # rheo.c[Int64(ceil(Ncy/2)+1)] = params.c/sc.σ
    # @. rheo.K = 2/3 .* rheo.G.*(1 .+ rheo.ν) ./ (1 .- 2 .* rheo.ν)
    # rheo.ϕ[Int64(floor(Ncy/2))] *= 0.99999999
    qDy  = τxxi*ones(Ncy+0)
    τxx  = τxxi*ones(Ncy+1)
    τyy  = τyyi*ones(Ncy+1)
    τzz  = τzzi*ones(Ncy+1)
    τxy  = τxyi*ones(Ncy+1)
    Pt   = Pi  *ones(Ncy+1)
    Pf   = Pi/2*ones(Ncy+1)

    Pf  .= Pi/4 .+ Pi/2 .* exp.(-yv.^2 ./ ((y.max-y.min)/10)^2)

    ϕ    =  ϕi*ones(Ncy+1)
    fc   = zeros(Ncy+1)
    pl   = zeros(Ncy+1)
    Ptc  = zeros(Ncy+1)
    Pfc  = zeros(Ncy+1)
    ϕc   = zeros(Ncy+1)
    
    # rheo.G[1:Int64(floor(Ncy/2))] .*= 2.1
    # Pt[Int64(floor(Ncy/2))] = Pt[Int64(floor(Ncy/2))]*1.1
    # rheo.G[Int64(floor(Ncy/2))] /= 10
    # rheo.c[end] = 1000

    Vx   = collect(ε̇0.*yc)
    Vy   = zeros(Ncy+0)
    Vx0  = collect(ε̇0.*yc)
    Vy0  = zeros(Ncy+0)
    ρ    = params.ρ/(sc.L*sc.σ*sc.t^2) * ones(Ncy+0)

    τxx0 = zeros(Ncy+1)
    τyy0 = zeros(Ncy+1)
    τzz0 = zeros(Ncy+1)
    τxy0 = zeros(Ncy+1)
    εyyt = zeros(Ncy+1)
    Pt0  = zeros(Ncy+1)
    Pf0  = zeros(Ncy+1)
    ϕ0   = zeros(Ncy+1)

    N    = 2*(Ncy+0) + 3*(Ncy+1)
    F    = zeros(N)
    x    = zeros(N)
    ind   = (x=Ncy, y=2*(Ncy), pt=2*(Ncy)+(Ncy+1), pf=2*(Ncy)+2*(Ncy+1))  
    NumV = (x=1:Ncy, y=Ncy+1:2*Ncy)
    probes = (
        fric = zeros(Nt),
        σxx  = zeros(Nt),
        θσ3  = zeros(Nt),
        εyy  = zeros(Nt),
    )

    maps = ( 
       Vx = zeros(Nt,Ncy+0),
       Vy = zeros(Nt,Ncy+0),
       Pt = zeros(Nt,Ncy+1),
       Pf = zeros(Nt,Ncy+1),
       ϕ  = zeros(Nt,Ncy+1),
    )

    # Sparsity pattern
    input       = rand(N)
    output      = similar(input)
    Res_closed! = (F,x) -> Res!(F, x, ϕ0, Pt0, Pf0, τxx0, τyy0, τzz0, τxy0, ρ, Vx0, Vy0, V_BC, σ_BC, rheo, ind, NumV, Δy, Δt )
    sparsity    = Symbolics.jacobian_sparsity(Res_closed!, output, input)
    J           = Float64.(sparse(sparsity))

    # Makes coloring
    colors   = matrix_colors(J) 

    # Globalisation
    LS = (
        α = [0.01 0.025 0.05 0.1 0.25 0.5 0.75 1.0], 
        F = zeros(8),
    )

    
    for it=1:Nt

        @printf("########### Step %06d ###########\n", it)

        # From previous time step
        τxx0 .= τxx
        τyy0 .= τyy
        τzz0 .= τzz
        τxy0 .= τxy
        Pt0  .= Pt
        Pf0  .= Pf
        ϕ0   .= ϕ
        Vx0  .= Vx 
        Vy0  .= Vy

        # Vx  .= collect(ε̇0.*yc) 
        # Vy  .= 0*Vy

        # Populate global solution array
        x[1:ind.x]         .= Vx
        x[ind.x+1:ind.y]   .= Vy
        x[ind.y+1:ind.pt]  .= Pt
        x[ind.pt+1:ind.pf] .= Pf
        x[ind.pf+1:end]    .= ϕ

        @. rheo.ηϕ = rheo.η/ϕ 
        @. rheo.Kd = (1 - ϕ)*(1/rheo.Ks + 1/(rheo.G/ϕ))^(-1)
        @. rheo.α  = 1 - rheo.Kd/rheo.Ks
        @. rheo.B  = (1/rheo.Kd - 1/rheo.Ks) / ((1/rheo.Kd - 1/rheo.Ks) +  ϕ*(1/rheo.Kf - 1/rheo.Ks))

        ϵglob = 1e-13

        # Newton iterations
        for iter=1:100

            # Residual
            Res!(F, x, ϕ0, Pt0, Pf0, τxx0, τyy0, τzz0, τxy0, ρ, Vx0, Vy0, V_BC, σ_BC, rheo, ind, NumV, Δy, Δt )
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
        Pt .= x[ind.y+1:ind.pt]
        Pf .= x[ind.pt+1:ind.pf]
        ϕ  .= x[ind.pf+1:end  ]

        # Compute stress for postprocessing
        ComputeStress!(τxx, τyy, τzz, τxy, qDy, Ptc, Pfc, ϕc, fc, pl, εyyt, x, τxx0, τyy0, τzz0, τxy0, Pt0, Pf0, V_BC, σ_BC, rheo, ind, Δy, Δt )        # Pt .= Ptc #!!!!!!!!!
        Pf .= Pfc
        Pt .= Ptc

        # Postprocessing
        @printf("σyy_BC = %2.4e, max(f) = %2.4e\n", (τyy[end]-Pt[end])*sc.σ, maximum(fc)*sc.σ)
        probes.fric[it] = -τxy[end]/(τyy[end]-Pt[end])
        probes.εyy[it]  = εyyt[end]
        maps.Vx[it,:]  .= Vx
        maps.Vy[it,:]  .= Vy
        maps.Pt[it,:]  .= Pt
        maps.Pf[it,:]  .= Pf
        maps.ϕ[it,:]  .= ϕ

        nout = 1000
        if mod(it, nout)==0 || it==1
            @show "start plot"
            # p1 = plot()
            # p1 = plot!(Vx, yc, label="Vx")
            # p1 = plot!(Vy, yc, label="Vy")
            # p1 = plot!(Pt,  yv, label="Pt")
            p1 = heatmap((1:Nt)*ε̇0*Δt*100, yc, maps.Vx[1:Nt,:]', title="Vx", xlabel="strain", ylabel="y") #, clim=(0,1)
            # p2 = heatmap((1:Nt)*ε̇0*Δt*100, yc, maps.Vy[1:Nt,:]', title="Vy", xlabel="strain", ylabel="y") #, clim=(0,1/2)
            # p3 = heatmap((1:Nt)*ε̇0*Δt*100, yv, maps.Pt[1:Nt,:]',  title="Pt",  xlabel="strain", ylabel="y")
            p2 = plot(Pf, yv, label="Pf")
            # p2 = plot!(Pt, yv, label="Pt")
            p3 = scatter(V90.x, V90.y)
            p3 = plot!((1:it)*ε̇0*Δt*100, probes.fric[1:it])
            # p4 = plot((1:it)*ε̇0*Δt*100, probes.εyy[1:it]*100)
            p4 = plot(εyyt[1:end-1], yv[1:end-1])
            p4 = scatter!(εyyt[pl.==1], yv[pl.==1])
            p5 = plot(qDy, yc, label="qDy")
            p6 = plot(  ϕ, yv, label="ϕ")
            display(plot(p1,p2,p3,p4,p5,p6))
            @show "end plot"
        end
    end 
   
end

# σ0 = (xx= -25e3, yy=-100e3) # Case A
σ0 = (xx=-400e3, yy=-100e3) # Case B 
main(σ0)