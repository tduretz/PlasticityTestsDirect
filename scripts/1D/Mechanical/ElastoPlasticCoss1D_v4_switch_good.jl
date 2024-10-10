using Plots, SparseArrays, Symbolics, SparseDiffTools, Printf, CSV, DataFrames, IfElse, StaticArrays,LinearAlgebra 


# Neumann off

# with inertia

BC_Vy_N = :Neumann
#BC_Vy_N = :Dirichlet

function PrincipalStress!(σ1, σ3, τxx, τyy, τzz, τxy, Rz, myz, P, lc)
    for i in eachindex(τxy)
        #σ  = @SMatrix[-P[i]+τxx[i] τxy[i]-Rz[i] 0. 0.; τxy[i]+Rz[i] -P[i]+τyy[i] 0. 0.; 0. 0. -P[i]+τzz[i] 0.; 0. 0. 0. myz[i]/lc] 
        σ  = @SMatrix[-P[i]+τxx[i] τxy[i] 0. 0.; τxy[i] -P[i]+τyy[i] 0. 0.; 0. 0. -P[i]+τzz[i] 0.; 0. 0. 0. myz[i]/lc] 
       
        #σ  = @SMatrix[-P[i]+τxx[i] τxy[i] 0.; τxy[i] -P[i]+τyy[i] 0.; 0. 0. -P[i]+τzz[i]]
        v  = eigvecs(σ)
        σp = eigvals(Array(σ))
        σ1.x[i] = v[1,1]
        σ1.z[i] = v[2,1]
        σ3.x[i] = v[1,3]
        σ3.z[i] = v[2,3]
        σ1.v[i] = σp[1]
        σ3.v[i] = σp[3]
    end
end

function NeumannVerticalVelocity(σ_BC, VyS, G, K, τyy0, P0, Δy, Δt)
    return (2.0 * G .* VyS .* Δt + 3.0 * K .* VyS .* Δt + 2.0 * P0 .* Δy + 2.0 * Δy .* σ_BC.yy - 2.0 * Δy .* τyy0) ./ (Δt .* (2.0 * G + 3.0 * K))
end

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

    return τxx, τyy, τzz, τxy, Pc, Rz, myz, fc, pl, ε̇yyt, λ̇, τII 
end

function ComputeStress!(τxx, τyy, τzz, τxy, Pc, Rz, Myz, Fc, Pl, εyyt,  x, P, P0, τxx0, τyy0, τzz0, τxy0, Rz0, myz0, V_BC, σ_BC, rheo, ind, NumV, Δy, Δt,lambda_dot, sec_inv)

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
           

            if BC_Vy_N == :Neumann
                VyN = NeumannVerticalVelocity(σ_BC, x[iy-1], rheo.G[end], rheo.K[end], τyy0[end], P0[end], Δy, Δt)
            else
                VyN = 2*V_BC.y.N - x[iy-1]
            end


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
        τ11, τ22, τ33, τ21, pc, rz, myz, fc, pl, ε̇yyt, λ̇, τII = LocalStress( VxN, VxS, VyN, VyS, ω̇zN, ω̇zS, p, P0[i], τxx0[i], τyy0[i], τzz0[i], τxy0[i], Rz0[i], myz0[i], rheo.G[i], rheo.K[i], rheo.lc[i], rheo.ϕ[i], rheo.ψ[i], rheo.c[i], rheo.ηvp[i], Δy, Δt )
        τxx[i]         = τ11
        τyy[i]         = τ22
        τzz[i]         = τ33
        τxy[i]         = τ21
        Pc[i]          = pc
        Rz[i]          = rz
        Myz[i]         = myz
        Fc[i]          = fc
        Pl[i]          = pl 
        εyyt[i]       += ε̇yyt*Δt
        lambda_dot[i]  = λ̇
        sec_inv[i]     = τII
    end
end

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
            if BC_Vy_N == :Neumann
                VyN = NeumannVerticalVelocity(σ_BC, VyC, rheo.G[end], rheo.K[end], τyy0[end], P0[end], Δy, Δt)
            else
                VyN = 2*V_BC.y.N - VyC
            end
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
            if BC_Vy_N == :Neumann
                VyN = NeumannVerticalVelocity(σ_BC, x[iy-1], rheo.G[end], rheo.K[end], τyy0[end], P0[end], Δy, Δt)
            else
                VyN = 2*V_BC.y.N - x[iy-1] 
            end
        else
            VyS = x[iy-1]
            VyN = x[iy]
        end
        divV  =  (VyN-VyS)/Δy
        F[iP] = -(x[iP] - P0[i]) - divV*rheo.K[i]*Δt
    end
end

function LineSearch(Res_closed!, F, x, δx, LS)
    normF_k = norm(F)
    for i=1:length(LS.α)
        Res_closed!(F, x.-LS.α[i].*δx)
        LS.F[i] = norm(F)
    end
    v, i_opt = findmin(LS.F)
    if LS.F[i_opt] > normF_k
        print("Diverged Linesearch! \n") 
        i_opt = -1 
    end
    return i_opt
end

function main(σ0)

    params = (
        #---------------#
        K   = 10e6,#6.6666666667e6, # K = 3/2*Gv in Vermeer (1990)
        G   = 10e6,
        lc  = 0.01,  # 1/3 dx or it does not converge even with my slopy tol 
        ν   = 0., 
        c   = 0.0e4,
        ϕ   = 40/180*π,
        ψ   = 0/180*π,
        θt  = 25/180*π,
        ρ   = 2000.,
        ηvp = 0.0e7,
        γ̇xy = 0.00001,
        Δt  = 0.1,
        Δtmax  = 5,
        nt  = 1600*5*10,
        law = :MC_Vermeer1990,
        oop = :Vermeer1990,
        time_max = 8000,
        pl  = true )
        
    sc   = (σ = params.G, L = 1.0, t = 1.0/params.γ̇xy)

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
  
    Ncy = 100
    Nt  = params.nt
    Δy  = (y.max-y.min)/Ncy
    Δt  = params.Δtmax/sc.t
    time_max = params.time_max /sc.t
    yc  = LinRange(y.min+Δy/2, y.max-Δy/2, Ncy  )
    yv  = LinRange(y.min,      y.max,      Ncy+1)

    rheo = (
        c    = params.c/sc.σ*ones(Ncy+1).+250/sc.σ,
        # c    = params.c/sc.σ*ones(Ncy+1).+0.001/sc.σ,
        # c    = params.c/sc.σ*ones(Ncy+1),
        ψ    = params.ψ      *ones(Ncy+1),
        ϕ    = params.ϕ      *ones(Ncy+1),
        ν    = params.ν      *ones(Ncy+1),
        G    = params.G/sc.σ *ones(Ncy+1),
        K    = params.K/sc.σ *ones(Ncy+1),
        ηvp  = params.ηvp    *ones(Ncy+1)/(sc.σ*sc.t),
        lc   = params.lc/sc.L*ones(Ncy+1),
    )

    #rheo.c[Int64(ceil(Ncy/2)+1)] = params.c/sc.σ
    rheo.c[Int64(floor(Ncy/2-Ncy/4)):Int64(floor(Ncy/2+Ncy/4))] .= params.c/sc.σ
    #rheo.c[Int64(floor(Ncy/2)+1):end] .= params.c/sc.σ
    # @. rheo.K = 2/3 .* rheo.G.*(1 .+ rheo.ν) ./ (1 .- 2 .* rheo.ν)
    # rheo.ϕ[Int64(floor(Ncy/2))] *= 0.99999999
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
    lambda_dot = zeros(Ncy+1)
    sec_inv    = zeros(Ncy+1)
    
    # rheo.G[1:Int64(floor(Ncy/2))] .*= 2.1
    # P[Int64(floor(Ncy/2))] = P[Int64(floor(Ncy/2))]*1.1
    # rheo.G[Int64(floor(Ncy/2))] /= 10
    # rheo.c[end] = 1000

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

    N    = 3*(Ncy+0) + Ncy+1
    F    = zeros(N)
    x    = zeros(N)
    ind   = (x=Ncy, y=2*(Ncy), p=2*(Ncy)+(Ncy+1))  
    NumV = (x=1:Ncy, y=Ncy+1:2*Ncy)
    probes = (
        fric = zeros(Nt),
        σxx  = zeros(Nt),
        θσ3  = zeros(Nt),
        εyy  = zeros(Nt),
        θs3_out = zeros(Nt),
        θs3_in  = zeros(Nt),
        time = zeros(Nt),
    )
    

    maps = ( 
       Vx = zeros(Nt,Ncy+0),
       Vy = zeros(Nt,Ncy+0),
       P  = zeros(Nt,Ncy+1),
       ε̇xy  = zeros(Nt,Ncy+1),
    )
    σ1         = (x=zeros(size(τxx)), z=zeros(size(τxx)), v=zeros(size(τxx)) )
    σ3         = (x=zeros(size(τxx)), z=zeros(size(τxx)), v=zeros(size(τxx)) )
    # Sparsity pattern
    input       = rand(N)
    output      = similar(input)
    Res_closed! = (F,x) -> Res!(F, x, P, P0, τxx0, τyy0, τzz0, τxy0, Rz0, myz0, ρ, Vx0, Vy0, V_BC, σ_BC, rheo, ind, NumV, Δy, Δt )
    sparsity    = Symbolics.jacobian_sparsity(Res_closed!, output, input)
    J           = Float64.(sparse(sparsity))

    # Makes coloring
    colors   = matrix_colors(J) 

    # Globalisation
    coeff = [1e-4 1e-3 0.01 0.1 0.25 0.5 0.75 0.99 1.01]
    LS = (
        α = coeff, 
        #α = [0.1 0.2 0.3 0.4 0.5 0.75 1.0],
        F = zeros(length(coeff)),
    )
    probes.time[1] = 0. 
    for it=1:Nt
        
        @printf("########### Step %06d time = %f time_max %f ###########\n", it, probes.time[it], time_max)
        

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
        i_opt = -1
        # Newton iterations
        
            for iter=1:100000
                
                # Residual
                Res!(F, x, P, P0, τxx0, τyy0, τzz0, τxy0, Rz0, myz0, ρ, Vx0, Vy0, V_BC, σ_BC, rheo, ind, NumV, Δy, Δt )
                
                if norm(F)/length(F)<ϵglob break end

                # Jacobian assembly
                forwarddiff_color_jacobian!(J, Res_closed!, x, colorvec = colors)

                # Solve
                δx    = J\F
                Δt_change = 0
                # Line search and update of global solution array
                i_opt = LineSearch(Res_closed!, F, x, δx, LS) 
                if i_opt == -1 
                    Δt = Δt/2
                    print("decrease delta t to %f\n", Δt)
                    Δt_change = 1
                end
                if i_opt == (length(coeff)) && Δt < params.Δtmax/sc.t
                    Δt = Δt*1.2
                    Δt_change = 1
                    print("increase delta t to %f\n", Δt)
                end
                if Δt_change == 0
                x   .-= LS.α[i_opt]*δx
                end
                @show (iter,i_opt, norm(F),Δt)

            end
            
        
            if (norm(F)/length(F)>ϵglob) error("Diverged!") end

            # Extract fields from global solution array
            Vx .= x[1:ind.x]
            Vy .= x[ind.x+1:ind.y]
            P  .= x[ind.y+1:ind.p]
            ω̇z .= x[ind.p+1:end  ]

            # Compute stress for postprocessing
            ComputeStress!(τxx, τyy, τzz, τxy, Pc, Rz, myz, fc, pl, εyyt, x, P, P0, τxx0, τyy0, τzz0, τxy0, Rz0, myz0, V_BC, σ_BC, rheo, ind, NumV, Δy, Δt, lambda_dot, sec_inv)
            P .= Pc #!!!!!!!!!

            # Postprocessing
            # Probe model state
            _, iA = findmax(P)
            #iA    = 100
            _, iB = findmax(lambda_dot)
            @show iA,iB
            @printf("σyy_BC = %2.4e, max(f) = %2.4e\n", (τyy[end]-P[end])*sc.σ, maximum(fc)*sc.σ)
        # probes.fric[it] = -τxy[end]/(τyy[end]-P[end])
        #probes.fric[it] = -τxy[1]/(τyy[1]-P[1])
            probes.fric[it] = -τxy[iB]/(τyy[iB]-P[iB])

            
            probes.εyy[it]  = εyyt[end]
            maps.Vx[it,:]  .= Vx
            maps.Vy[it,:]  .= Vy
            maps.P[it,:]   .= P
            maps.ε̇xy[it,2:end-1] .= 0.5*diff(Vx,dims=1)/Δy

            PrincipalStress!(σ1, σ3, τxx, τyy, τzz, τxy, Rz, myz, P, params.lc/sc.L)
            

            # # Probe model state
            # _, iA = findmin(σ3.v)
            # _, iB = findmax(σ1.v)
            # probes.θs3_out[it]  = atand(σ3.z[iA] ./ σ3.x[iA])
            # probes.θs3_in[it]   = atand(σ3.z[iB] ./ σ3.x[iB])
            # Probe model state
            probes.θs3_out[it]  = atand(σ3.z[iA] ./ σ3.x[iA])
            probes.θs3_in[it]   = atand(σ3.z[iB] ./ σ3.x[iB])

            probes.time[it+1] = probes.time[it]+Δt
            

            nout = 1000
            if mod(it, nout)==0 || it==1 || probes.time[it+1] > time_max
                θ   = LinRange(-π, 0, 100)
                σMC = LinRange(-500, 0, 100 ) .*1e3
                τMC = -σMC.*tan(params.ϕ) 

                PA    = ((σ1.v[iA] + σ3.v[iA])/2)*sc.σ
                τA    = ((σ1.v[iA] - σ3.v[iA])/2)*sc.σ
                τB    = ((σ1.v[iB] - σ3.v[iB])/2)*sc.σ
                PB    = ((σ1.v[iB] + σ3.v[iB])/2)*sc.σ
                
                yield = (x = σMC./1e3, y = τMC./1e3)
                MC_A  = (x = (τA.*cos.(θ) .+ PA)./1e3, y = (τA.*sin.(θ))./1e3) 
                MC_B  = (x = (τB.*cos.(θ) .+ PB)./1e3, y = (τB.*sin.(θ))./1e3)

                p1 = plot(title = "Stress orientation", ylabel = "θ σ3 [ᵒ]", xlabel = "γxy BC [%]", xlims=(0,8), foreground_color_legend = nothing, background_color_legend = nothing  )
                p1 = plot!((1:it)*ε̇0*Δt*100, probes.θs3_out[1:it], label="out", color=:blue  )
                p1 = plot!((1:it)*ε̇0*Δt*100, probes.θs3_in[1:it],  label="in" , color=:green )

                p2 = plot(title="Mohr circles", ylabel="τ [kPa]", xlabel="σₙ [kPa]", size=(300,300), aspect_ratio=1, xlim=(-500,0), ylim=(0,400))
                p2 = plot!( MC_A... , color=:blue, label="out" )
                p2 = plot!( MC_B...,  color=:green, label="in"  )
                p2 = plot!( yield..., color=:red, label="Yield"  )
                p2 = plot!(ylabel="τ [kPa]", xlabel="σₙ [kPa]", foreground_color_legend = nothing, background_color_legend = nothing, legend=:topright)

                p3 = plot()
                # p3 = scatter(V90.x, V90.y)
                p3 = plot!(probes.time[1:it], probes.fric[1:it], label=:none)
                p3 = plot!(probes.time[1:it], tan(params.ϕ).*ones(it), linestyle=:dashdot, label="tan(ϕ)") # 22b Vermeer1990
                p3 = plot!(probes.time[1:it], cos(params.ψ)*sin(params.ϕ)/(1-sin(params.ψ)*sin(params.ϕ)).*ones(it), linestyle=:dashdot, label="tan(α)") # 22c Vermeer1990
                # p4 = plot((1:it)*ε̇0*Δt*100, probes.εyy[1:it]*100)
                # p4 = plot(εyyt[1:end-1], yv[1:end-1])
                # p4 = scatter!(εyyt[pl.==1], yv[pl.==1])
                p4 = heatmap(probes.time[1:it], yv, maps.ε̇xy[1:it,:]', title="ε̇xy", xlabel="strain", ylabel="y", clim=(-0.1,0.1)) # , clim=(0,20)
                display(plot(p1,p2,p3,p4))

                @show Pi, τxxi, τyyi, τzzi, τxyi
                if probes.time[it+1] > time_max break end
           
        end
        
    end 
    
   
end

# σ0 = (xx= -25e3, yy=-100e3) # Case A
σ0 = (xx=-400e3, yy=-100e3) # Case B 
main(σ0)