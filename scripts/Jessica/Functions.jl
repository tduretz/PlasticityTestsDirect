# Ghost node velocity for a Neumann boundary condition: constant total stress at the top of the model 
function NeumannVerticalVelocity(σ_BC, VyS, G, K, τyy0, P0, Δy, Δt)
    return (2.0 * G .* VyS .* Δt + 3.0 * K .* VyS .* Δt + 2.0 * P0 .* Δy + 2.0 * Δy .* σ_BC.yy - 2.0 * Δy .* τyy0) ./ (Δt .* (2.0 * G + 3.0 * K))
end

# Computation of the stress at ONE POINT given a velocity/pressure values and material parameters
function LocalStress(VxN, VxS, VyN, VyS, ω̇zN, ω̇zS, P, P0, τxx0, τyy0, τzz0, τxy0, Rz0, myz0, G, K, lc, ϕ, ψ, c, ηvp, Δy, Δt)
    
    # Total strain rate
    ε̇xxt  = 0.0
    ε̇yyt  = (VyN - VyS)/Δy
    ε̇zzt  = 1/2*(ε̇xxt + ε̇yyt)
    ε̇xyt  = 1/2*( VxN -  VxS)/Δy
    divV  = ε̇xxt + ε̇yyt + ε̇zzt
    ω̇z    = 0.5*(ω̇zN + ω̇zS)
    Ẇz    = 1/2*( VxN -  VxS)/Δy + ω̇z
    κ̇yz   = (ω̇zN - ω̇zS)/Δy

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
    Rz    =     -2*G*Δt*Ẇz  + Rz0 
    myz   = lc^2*2*G*Δt*κ̇yz + myz0

    # Yield function
    τII   = sqrt(1/2*(τxx^2 + τyy^2 + τzz^2 + (myz/lc)^2) + τxy^2 + Rz^2)
    Pc    = P
    f     = τII - c*cos(ϕ) - Pc*sin(ϕ)

    # Return mapping
    fc    = f
    λ̇_pl  = f / (G*Δt + K*Δt*sin(ϕ)*sin(ψ) + ηvp)
    λ̇     = IfElse.ifelse(f>=0.0, λ̇_pl, 0.0)
    pl    = IfElse.ifelse(f>=0.0, 1.0, 0.0)
    ε̇xxp  = λ̇*(τxx)/2/τII
    ε̇yyp  = λ̇*(τyy)/2/τII
    ε̇zzp  = λ̇*(τzz)/2/τII
    ε̇xyp  = λ̇*(τxy)/1/τII
    ẇzp   = λ̇*( Rz/2/τII) 
    κ̇yzp  = λ̇*(myz/2/τII)/lc^2
    τxx   =      2*G*Δt*(ε̇xx - ε̇xxp  ) + τxx0
    τyy   =      2*G*Δt*(ε̇yy - ε̇yyp  ) + τyy0
    τzz   =      2*G*Δt*(ε̇zz - ε̇zzp  ) + τzz0
    τxy   =      2*G*Δt*(ε̇xy - ε̇xyp/2) + τxy0
    Rz    =     -2*G*Δt*(Ẇz + ẇzp)   + Rz0 
    myz   = lc^2*2*G*Δt*(κ̇yz - κ̇yzp) + myz0
    Pc    = P + λ̇*K*Δt*sin(ψ)
    τII   = sqrt(1/2*(τxx^2 + τyy^2 + τzz^2 + (myz/lc)^2) + τxy^2 + Rz^2)
    fc    = τII - c*cos(ϕ) - Pc*sin(ϕ) - ηvp*λ̇

    return τxx, τyy, τzz, τxy, Pc, Rz, myz, fc, pl, ε̇yyt
end

# Computation of stress at ALL NODES given a velocity/pressure values and material parameters
function ComputeStress!(τxx, τyy, τzz, τxy, Pc, Rz, Myz, Fc, Pl, εyyt,  x, P, P0, τxx0, τyy0, τzz0, τxy0, Rz0, myz0, V_BC, σ_BC, rheo, ind, NumV, Δy, Δt )

    # Loop over all stress nodes
    for i=1:length(NumV.x)+1
        ix  = i 
        iy  = ind.x+i 
        iP  = ind.y+i
        iω̇z = ind.p+i
        VxS, VxN = 0., 0. 
        VyS, VyN = 0., 0. 
        p   = x[iP] 
        if i==1 # Special case for bottom (South) vertex
            VxS = 2*V_BC.x.S - x[ix]
            VxN = x[ix]
            VyS = 2*V_BC.y.S - x[iy]
            VyN = x[iy]
            ω̇zS = x[iω̇z]
            ω̇zN = x[iω̇z]
        elseif i==length(NumV.x)+1 # Special case for top (North) vertex
            VxS = x[ix-1]
            VxN = 2*V_BC.x.N - x[ix-1]
            VyS = x[iy-1]
            # VyN = 2*V_BC.y.N - x[iy-1] 
            VyN = NeumannVerticalVelocity(σ_BC, x[iy-1], rheo.G[end], rheo.K[end], τyy0[end], P0[end], Δy, Δt)
            ω̇zS = x[iω̇z-1]
            ω̇zN = x[iω̇z-1]
        else # General case
            VxS = x[ix-1]
            VxN = x[ix]
            VyS = x[iy-1]
            VyN = x[iy]
            ω̇zS = x[iω̇z-1]
            ω̇zN = x[iω̇z]
        end
        # Compute stress and corrected pressure 
        τ11, τ22, τ33, τ21, pc, rz, myz, fc, pl, ε̇yyt = LocalStress( VxN, VxS, VyN, VyS, ω̇zN, ω̇zS, p, P0[i], τxx0[i], τyy0[i], τzz0[i], τxy0[i], Rz0[i], myz0[i], rheo.G[i], rheo.K[i], rheo.lc[i], rheo.ϕ[i], rheo.ψ[i], rheo.c[i], rheo.ηvp[i], Δy, Δt )
        τxx[i]   = τ11
        τyy[i]   = τ22
        τzz[i]   = τ33
        τxy[i]   = τ21
        Pc[i]    = pc
        Rz[i]    = rz
        Myz[i]   = myz
        Fc[i]    = fc
        Pl[i]    = pl 
        εyyt[i] += ε̇yyt*Δt
    end
end

# Computation of the resiudal of the mechanical equations
function Res!(F, x, P, P0, τxx0, τyy0, τzz0, τxy0, Rz0, myz0, ρ, Vx0, Vy0, V_BC, σ_BC, rheo, ind, NumV, Δy, Δt )

    # Loop over all velocity nodes
    for i=1:length(NumV.x)
        ix   = i
        iy   = ind.x+i
        iP   = ind.y+i
        iω̇z  = ind.p+i
        VxC, VxS, VxN = x[ix],  0., 0.
        VyC, VyS, VyN = x[iy],  0., 0.
        ω̇zC, ω̇zS, ω̇zN = x[iω̇z], 0., 0.
        PS     = x[iP]
        PN     = x[iP+1]
        if i==1
            VxS = 2*V_BC.x.S - VxC
            VyS = 2*V_BC.y.S - VyC
            ω̇zS = ω̇zC
        else
            VxS = x[ix-1]
            VyS = x[iy-1]
            ω̇zS = x[iω̇z-1]
        end
        if i==ind.x
            VxN = 2*V_BC.x.N - VxC
            # VyN = 2*V_BC.y.N - VyC 
            VyN = NeumannVerticalVelocity(σ_BC, VyC, rheo.G[end], rheo.K[end], τyy0[end], P0[end], Δy, Δt)
            ω̇zN = ω̇zC
        else
            VxN = x[ix+1]
            VyN = x[iy+1]
            ω̇zN = x[iω̇z+1]
        end
        τxxN, τyyN, τzzN, τxyN, PN, RzN, myzN = LocalStress( VxN, VxC, VyN, VyC, ω̇zN, ω̇zC, PN, P0[i+1], τxx0[i+1], τyy0[i+1], τzz0[i+1], τxy0[i+1], Rz0[i+1], myz0[i+1], rheo.G[i+1], rheo.K[i+1],rheo.lc[i+1], rheo.ϕ[i+1], rheo.ψ[i+1], rheo.c[i+1], rheo.ηvp[i+1], Δy, Δt )
        τxxS, τyyS, τzzS, τxyS, PS, RzS, myzS = LocalStress( VxC, VxS, VyC, VyS, ω̇zC, ω̇zS, PS, P0[i],   τxx0[i],   τyy0[i],   τzz0[i],   τxy0[i],   Rz0[i],   myz0[i],   rheo.G[i],   rheo.K[i],  rheo.lc[i],   rheo.ϕ[i],   rheo.ψ[i],   rheo.c[i],   rheo.ηvp[i],   Δy, Δt )
        F[ix]  = (τxyN - τxyS)/Δy - (RzN - RzS)/Δy - ρ[i]*(VxC-Vx0[i])/Δt
        F[iy]  = (τyyN - τyyS)/Δy - (PN  - PS )/Δy - ρ[i]*(VyC-Vy0[i])/Δt
        F[iω̇z] = (myzN - myzS)/Δy + 2*(RzN + RzS)/2
    end
    # Loop over all stress nodes
    for i=1:length(NumV.x)+1
        iP  = ind.y+i
        iy  = ind.x+i 
        VyS, VyN = 0., 0. 
        if i==1
            VyS = 2*V_BC.y.S - x[iy]
            VyN = x[iy]
        elseif i==length(NumV.x)+1
            VyS = x[iy-1]
            # VyN = 2*V_BC.y.N - x[iy-1] 
            VyN = NeumannVerticalVelocity(σ_BC, x[iy-1], rheo.G[end], rheo.K[end], τyy0[end], P0[end], Δy, Δt)
        else
            VyS = x[iy-1]
            VyN = x[iy]
        end
        divV  =  (VyN-VyS)/Δy
        F[iP] = -(x[iP] - P0[i]) - divV*rheo.K[i]*Δt
    end
end

# Line search function: seach for the optimal value of α
function LineSearch(Res_closed!, F, x, δx, LS)
    for i=1:length(LS.α)
        Res_closed!(F, x.-LS.α[i].*δx)
        LS.F[i] = norm(F)
    end
    v, i_opt = findmin(LS.F)
    return i_opt
end