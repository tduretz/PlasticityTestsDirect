using Plots, SparseArrays, Symbolics, SparseDiffTools, Printf, CSV, DataFrames, IfElse, StaticArrays
import LinearAlgebra: norm, eigvecs, eigvals

# with Laeti with nice visualisation
# BC switch
# add sxy0

#BC_Vy_N = :Neumann
BC_Vy_N = :Dirichlet
function PrincipalStress!(σ1, σ3, τxx, τyy, τzz, τxy, P)
    for i in eachindex(τxy)
        σ  = @SMatrix[-P[i]+τxx[i] τxy[i] 0.; τxy[i] -P[i]+τyy[i] 0.; 0. 0. -P[i]+τzz[i]]
        v  = eigvecs(σ)
        σp = eigvals(σ)
        σ1.x[i] = v[1,1]
        σ1.z[i] = v[2,1]
        σ3.x[i] = v[1,3]
        σ3.z[i] = v[2,3]
        σ1.v[i] = σp[1]
        σ3.v[i] = σp[3]
    end
end

# with inertia

function NeumannVerticalVelocity(σ_BC, VyS, G, K, τyy0, P0, Δy, Δt)
    return (2.0 * G .* VyS .* Δt + 3.0 * K .* VyS .* Δt + 2.0 * P0 .* Δy + 2.0 * Δy .* σ_BC.yy - 2.0 * Δy .* τyy0) ./ (Δt .* (2.0 * G + 3.0 * K))
end

function LocalStress(VxN, VxS, VyN, VyS, P, P0, τxx0, τyy0, τzz0, τxy0, G, K, ϕ, ψ, c, ηvp, Δy, Δt)
    
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
    τxx   = 2*G*Δt*ε̇xx + τxx0
    τyy   = 2*G*Δt*ε̇yy + τyy0
    τzz   = 2*G*Δt*ε̇zz + τzz0
    τxy   = 2*G*Δt*ε̇xy + τxy0
    
    # Yield function
    τII   = sqrt(1/2*(τxx^2 + τyy^2 + τzz^2) + τxy^2)
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
    τxx   = 2*G*Δt*(ε̇xx - ε̇xxp  ) + τxx0
    τyy   = 2*G*Δt*(ε̇yy - ε̇yyp  ) + τyy0
    τzz   = 2*G*Δt*(ε̇zz - ε̇zzp  ) + τzz0
    τxy   = 2*G*Δt*(ε̇xy - ε̇xyp/2) + τxy0
    Pc    = P + λ̇*K*Δt*sin(ψ)
    τII   = sqrt(1/2*(τxx^2 + τyy^2 + τzz^2) + τxy^2)
    fc    = τII - c*cos(ϕ) - Pc*sin(ϕ) - ηvp*λ̇

    return τxx, τyy, τzz, τxy, Pc, fc, pl, ε̇yyt
end

function ComputeStress!(τxx, τyy, τzz, τxy, Pc, Fc, Pl, εyyt,  x, P, P0, τxx0, τyy0, τzz0, τxy0, V_BC, σ_BC, rheo, iV, NumV, Δy, Δt )

    # Loop over all stress nodes
    for i=1:length(NumV.x)+1
        iVx = i 
        iVy = iV.x+i 
        iP  = iV.y+i
        VxS, VxN = 0., 0. 
        VyS, VyN = 0., 0. 
        p   = x[iP] 
        if i==1 # Special case for bottom (South) vertex
            VxS = 2*V_BC.x.S - x[iVx]
            VxN = x[iVx]
            VyS = 2*V_BC.y.S - x[iVy]
            VyN = x[iVy]
        elseif i==length(NumV.x)+1 # Special case for top (North) vertex
            VxS = x[iVx-1]
            VxN = 2*V_BC.x.N - x[iVx-1]
            VyS = x[iVy-1]
            if BC_Vy_N == :Neumann
                VyN = NeumannVerticalVelocity(σ_BC, x[iVy-1], rheo.G[end], rheo.K[end], τyy0[end], P0[end], Δy, Δt)
            else
                VyN = 2*V_BC.y.N - x[iVy-1]
            end
        else # General case
            VxS = x[iVx-1]
            VxN = x[iVx]
            VyS = x[iVy-1]
            VyN = x[iVy]
        end
        # Compute stress and corrcted pressure 
        τ11, τ22, τ33, τ21, pc, fc, pl, ε̇yyt = LocalStress( VxN, VxS, VyN, VyS, p, P0[i], τxx0[i], τyy0[i], τzz0[i], τxy0[i], rheo.G[i], rheo.K[i], rheo.ϕ[i], rheo.ψ[i], rheo.c[i], rheo.ηvp[i], Δy, Δt )
        τxx[i]   = τ11
        τyy[i]   = τ22
        τzz[i]   = τ33
        τxy[i]   = τ21
        Pc[i]    = pc
        Fc[i]    = fc
        Pl[i]    = pl 
        εyyt[i] += ε̇yyt*Δt
    end
end

function Res!(F, x, P, P0, τxx0, τyy0, τzz0, τxy0, ρ, Vx0, Vy0, V_BC, σ_BC, rheo, iV, NumV, Δy, Δt )

    # Loop over all velocity nodes
    for i=1:length(NumV.x)
        iVx = i
        iVy = iV.x+i
        iP  = iV.y+i
        VxC, VxS, VxN = x[iVx], 0., 0.
        VyC, VyS, VyN = x[iVy], 0., 0.
        PS     = x[iP]
        PN     = x[iP+1]
        if i==1
            VxS = 2*V_BC.x.S - VxC
            VyS = 2*V_BC.y.S - VyC
        else
            VxS = x[iVx-1]
            VyS = x[iVy-1]
        end
        if i==iV.x
            VxN = 2*V_BC.x.N - VxC
            if BC_Vy_N == :Neumann
                VyN = NeumannVerticalVelocity(σ_BC, VyC, rheo.G[end], rheo.K[end], τyy0[end], P0[end], Δy, Δt)
            else
                VyN = 2*V_BC.y.N - VyC
            end
        else
            VxN = x[iVx+1]
            VyN = x[iVy+1]
        end
        τxxN, τyyN, τzzN, τxyN, PN = LocalStress( VxN, VxC, VyN, VyC, PN, P0[i+1], τxx0[i+1], τyy0[i+1], τzz0[i+1], τxy0[i+1], rheo.G[i+1], rheo.K[i+1], rheo.ϕ[i+1], rheo.ψ[i+1], rheo.c[i+1], rheo.ηvp[i+1], Δy, Δt )
        τxxS, τyyS, τzzS, τxyS, PS = LocalStress( VxC, VxS, VyC, VyS, PS, P0[i],   τxx0[i],   τyy0[i],   τzz0[i],   τxy0[i],   rheo.G[i],   rheo.K[i],   rheo.ϕ[i],   rheo.ψ[i],   rheo.c[i],   rheo.ηvp[i],   Δy, Δt )
        F[iVx] = (τxyN - τxyS)/Δy                - ρ[i]*(VxC-Vx0[i])/Δt
        F[iVy] = (τyyN - τyyS)/Δy - (PN - PS)/Δy - ρ[i]*(VyC-Vy0[i])/Δt
    end
    # Loop over all stress nodes
    for i=1:length(NumV.x)+1
        iP  = iV.y+i
        iVy = iV.x+i 
        VyS, VyN = 0., 0. 
        if i==1
            VyS = 2*V_BC.y.S - x[iVy]
            VyN = x[iVy]
        elseif i==length(NumV.x)+1
            VyS = x[iVy-1]
            if BC_Vy_N == :Neumann
                VyN = NeumannVerticalVelocity(σ_BC, x[iVy-1], rheo.G[end], rheo.K[end], τyy0[end], P0[end], Δy, Δt)
            else
                VyN = 2*V_BC.y.N - x[iVy-1]
            end
        else
            VyS = x[iVy-1]
            VyN = x[iVy]
        end
        divV  =  (VyN-VyS)/Δy
        F[iP] = -(x[iP] - P0[i]) - divV*rheo.K[i]*Δt
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
        K   = 10e6,#6.6666666667e6, # K = 3/2*Gv in Vermeer (1990)
        G   = 10e6,
        ν   = 0., 
        c   = 0.0e4,
        ϕ   = 40/180*π,
        ψ   = 00/180*π,
        θt  = 25/180*π,
        ρ   = 2000.,
        ηvp = 0.001e7,
        γ̇xy = 0.00001,
        Δt  = 1,
        nt  = 1600*5,
        law = :MC_Vermeer1990,
        oop = :Vermeer1990,
        pl  = true)

    sc   = (σ = params.G, L = 1.0, t = 1.0/params.γ̇xy)

    ε̇0   = params.γ̇xy/(1/sc.t)
    σxxi = σ0.xx/sc.σ
    σyyi = σ0.yy/sc.σ
    σzzi = 0.5*(σxxi + σyyi)
    Pi   = -(σxxi + σyyi)/2.0
    τxxi = Pi + σxxi
    τyyi = Pi + σyyi
    τzzi = Pi + σzzi
    τxyi = σ0.xy/sc.σ
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
  
    Ncy = 100
    Nt  = params.nt * Int16(Ncy/10)
    Δy  = (y.max-y.min)/Ncy
    Δt  = params.Δt/sc.t * 10/Ncy
    yc  = LinRange(-y.min+Δy/2, y.max-Δy/2, Ncy  )
    yv  = LinRange(-y.min,      y.max,      Ncy+1)

    rheo = (
        c    = params.c/sc.σ*ones(Ncy+1).+250/sc.σ,
        # c    = params.c/sc.σ*ones(Ncy+1).+0.0000001/sc.σ,
        # c    = params.c/sc.σ*ones(Ncy+1),
        ψ    = params.ψ     *ones(Ncy+1),
        ϕ    = params.ϕ     *ones(Ncy+1),
        ν    = params.ν     *ones(Ncy+1),
        G    = params.G/sc.σ*ones(Ncy+1),
        K    = params.K/sc.σ*ones(Ncy+1),
        ηvp  = params.ηvp   *ones(Ncy+1)/(sc.σ*sc.t),
    )

    rheo.c[Int64(floor(Ncy/2)+1):end] .= params.c/sc.σ
    #rheo.c[Int64(floor(Ncy/2)+1)] = params.c/sc.σ
    #rheo.c[Int64(floor(Ncy/2)-2):Int64(floor(Ncy/2)+2)] .= params.c/sc.σ
    # @. rheo.K = 2/3 .* rheo.G.*(1 .+ rheo.ν) ./ (1 .- 2 .* rheo.ν)
    τxx  = τxxi*ones(Ncy+1)
    τyy  = τyyi*ones(Ncy+1)
    τzz  = τzzi*ones(Ncy+1)
    τxy  = τxyi*ones(Ncy+1)
    P    =   Pi*ones(Ncy+1)
    fc   = zeros(Ncy+1)
    pl   = zeros(Ncy+1)
    Pc   = zeros(Ncy+1)
    # rheo.G[Int64(floor(Ncy/2))] = rheo.G[Int64(floor(Ncy/2))]/3
    # P[Int64(floor(Ncy/2))] = P[Int64(floor(Ncy/2))]*1.00001
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
    P0   = zeros(Ncy+1)

    N    = 2*(Ncy+0) + Ncy+1
    F    = zeros(N)
    x    = zeros(N)
    iV   = (x=Ncy, y=2*(Ncy))  
    NumV = (x=1:Ncy, y=Ncy+1:2*Ncy)
    probes = (
        fric = zeros(Nt),
        σxx  = zeros(Nt),
        θσ3  = zeros(Nt),
        εyy  = zeros(Nt),
        θs3_out = zeros(Nt),
        θs3_in  = zeros(Nt),
    )

    maps = ( 
       Vx  = zeros(Nt,Ncy+0),
       Vy  = zeros(Nt,Ncy+0),
       P   = zeros(Nt,Ncy+1),
       ε̇xy = zeros(Nt,Ncy+1),
    )

    σ1         = (x=zeros(size(τxx)), z=zeros(size(τxx)), v=zeros(size(τxx)) )
    σ3         = (x=zeros(size(τxx)), z=zeros(size(τxx)), v=zeros(size(τxx)) )
 

    # Sparsity pattern
    input       = rand(N)
    output      = similar(input)
    Res_closed! = (F,x) -> Res!(F, x, P, P0, τxx0, τyy0, τzz0, τxy0, ρ, Vx0, Vy0, V_BC, σ_BC, rheo, iV, NumV, Δy, Δt )
    sparsity    = Symbolics.jacobian_sparsity(Res_closed!, output, input)
    J           = Float64.(sparse(sparsity))

    # Makes coloring
    colors   = matrix_colors(J) 

    # Globalisation
    LS = (
        α = [0.01 0.05 0.1 0.25 0.5 0.75 1.0], 
        F = zeros(7),
    )
    
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

        # Vx  .= collect(ε̇0.*yc) 
        # Vy  .= 0*Vy

        # Populate global solution array
        x[1:iV.x]      .= Vx
        x[iV.x+1:iV.y] .= Vy
        x[iV.y+1:end]  .= P

        ϵglob = 1e-13

        # Newton iterations
        for iter=1:100

            # Residual
            Res!(F, x, P, P0, τxx0, τyy0, τzz0, τxy0, ρ, Vx0, Vy0, V_BC, σ_BC, rheo, iV, NumV, Δy, Δt )
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
        Vx .= x[1:iV.x]
        Vy .= x[iV.x+1:iV.y]
        P  .= x[iV.y+1:end]

        # Compute stress for postprocessing
        ComputeStress!(τxx, τyy, τzz, τxy, Pc, fc, pl, εyyt, x, P, P0, τxx0, τyy0, τzz0, τxy0, V_BC, σ_BC, rheo, iV, NumV, Δy, Δt )
        P .= Pc #!!!!!!!!!

        # Postprocessing
        @printf("σyy_BC = %2.4e, max(f) = %2.4e\n", (τyy[end]-P[end])*sc.σ, maximum(fc)*sc.σ)
        probes.fric[it] = -τxy[end]/(τyy[end]-P[end])
        probes.εyy[it]  = εyyt[end]
        maps.Vx[it,:]  .= Vx
        maps.Vy[it,:]  .= Vy
        maps.P[it,:]   .= P
        maps.ε̇xy[it,2:end-1] .= 0.5*diff(Vx,dims=1)/Δy

        PrincipalStress!(σ1, σ3, τxx, τyy, τzz, τxy, P)

        # Probe model state
        _, iA = findmin(σ3.v)
        _, iB = findmax(σ1.v)
        probes.θs3_out[it]  = atand(σ3.z[iA] ./ σ3.x[iA])
        probes.θs3_in[it]   = atand(σ3.z[iB] ./ σ3.x[iB])

        nout = 1000
        if mod(it, nout)==0

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


            # p1 = plot()
            # p1 = plot!(Vx, yc, label="Vx")
            # p1 = plot!(Vy, yc, label="Vy")
            # p1 = plot!(P,  yv, label="P")
            p1 = plot(title = "Stress orientation", ylabel = "θ σ3 [ᵒ]", xlabel = "γxy BC [%]", xlims=(0,8), foreground_color_legend = nothing, background_color_legend = nothing  )
            p1 = plot!((1:it)*ε̇0*Δt*100, probes.θs3_out[1:it], label="out", color=:blue  )
            p1 = plot!((1:it)*ε̇0*Δt*100, probes.θs3_in[1:it],  label="in" , color=:green )

            p2 = plot(title="Mohr circles", ylabel="τ [kPa]", xlabel="σₙ [kPa]", size=(300,300), aspect_ratio=1, xlim=(-500,0), ylim=(0,400))
            p2 = plot!( MC_A... , color=:blue, label="out" )
            p2 = plot!( MC_B...,  color=:green, label="in"  )
            p2 = plot!( yield..., color=:red, label="Yield"  )
            p2 = plot!(ylabel="τ [kPa]", xlabel="σₙ [kPa]", foreground_color_legend = nothing, background_color_legend = nothing, legend=:topright)

            # p1 = heatmap((1:Nt)*ε̇0*Δt*100, yc, maps.Vx[1:Nt,:]', title="Vx", xlabel="strain", ylabel="y") #, clim=(0,1)
            # p2 = heatmap((1:Nt)*ε̇0*Δt*100, yc, maps.Vy[1:Nt,:]', title="Vy", xlabel="strain", ylabel="y") #, clim=(0,1/2)
            # p3 = heatmap((1:Nt)*ε̇0*Δt*100, yv, maps.P[1:Nt,:]',  title="P",  xlabel="strain", ylabel="y")
            p3 = plot()
            # p3 = scatter(V90.x, V90.y)
            p3 = plot!((1:it)*ε̇0*Δt*100, probes.fric[1:it], label=:none)
            p3 = plot!((1:it)*ε̇0*Δt*100, tan(params.ϕ).*ones(it), linestyle=:dashdot, label="tan(ϕ)") # 22b Vermeer1990
            p3 = plot!((1:it)*ε̇0*Δt*100, cos(params.ψ)*sin(params.ϕ)/(1-sin(params.ψ)*sin(params.ϕ)).*ones(it), linestyle=:dashdot, label="tan(α)") # 22c Vermeer1990
            # p4 = plot((1:it)*ε̇0*Δt*100, probes.εyy[1:it]*100)
            # p4 = plot(εyyt[1:end-1], yv[1:end-1])
            # p4 = scatter!(εyyt[pl.==1], yv[pl.==1])
            p4 = heatmap((1:Nt)*ε̇0*Δt*100, yv, maps.ε̇xy[1:Nt,:]', title="ε̇xy", xlabel="strain", ylabel="y", clim=(0,20)) # 
            display(plot(p1,p2,p3,p4))

            @show Pi, τxxi, τyyi, τzzi, τxyi

        end
    end 
   
end

# σ0 = (xx= -25e3, yy=-100e3, xy = 0.) # Case A
σ0 = (xx=-400e3, yy=-100e3, xy = 0.) # Case B 

# syy  = -100e3
# c    = 0.
# phi  = 40*pi/180
# mu_eff = tan(phi)
# sxx = (c .* sin(2 * phi) + syy .* sin(phi) .^ 2 + syy + 2 * sqrt(c .^ 2 .* cos(phi) .^ 2 + c .* syy .* sin(2 * phi) + mu_eff .^ 2 .* syy .^ 2 .* sin(phi) .^ 2 - mu_eff .^ 2 .* syy .^ 2 + syy .^ 2 .* sin(phi) .^ 2)) ./ cos(phi) .^ 2
# σ0 = (xx=sxx, yy=syy, xy = -mu_eff*syy) # Case B 
main(σ0)