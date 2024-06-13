using Plots

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

        dt = [1 1e5 1e10]

        G = params.G
        K = params.K
        ψ = params.ψ
        ϕ = params.ϕ

        p  = plot(xlabel="P [MPa]", ylabel="τ [MPa]")
        Pv = LinRange(0, 1e8, 100)
        τ_DP = Pv.*sin(params.ϕ) .+ params.c*cos(params.ϕ)
        p    = plot!(Pv./1e6, τ_DP./1e6, label=:none)
        τ_trial = [20. 50.].*1e6 
        P_trial = [10. 50.].*1e6
        
        for idt in eachindex(dt)

            Δt = dt[idt]
            η_ve = G*Δt
            # η_ve = 1e10
    
            f       = τ_trial .- P_trial.*sin(params.ϕ) .+ params.c*cos(params.ϕ)
            λ̇       = f / (η_ve + K*Δt*sin(ϕ)*sin(ψ))
            P       = P_trial .+ K*Δt*sin(ψ).*λ̇
            τ       = τ_trial .- η_ve*λ̇

            for i in eachindex(τ_trial)
                p=scatter!([P_trial[i]  P[i]]./1e6, [τ_trial[i]  τ[i]]./1e6, label=:none)
                p=   plot!([P_trial[i]; P[i]]./1e6, [τ_trial[i]; τ[i]]./1e6, label="dt = $(Δt)")
            end
    end

    display(p)
 
end

main()