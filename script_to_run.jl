using Plots
using LaTeXStrings

include("find_equilibria.jl")

n = 8
ω = 1.0
r0 = 0.0:0.01:0.5
#r0 = 0.2
r1 = 0.35:0.01:0.55
T = 2π / ω
times = T/4:0.5:T/4

#=
result, sym = find_equilibria_series(n, times, ω, r0, r1)
aligned = align_solutions(result)
real_result = mark_real(aligned)
stablereal_result = mark_real_stable(aligned, ω, times, sym, real_result)

real_sol = real(result[1][real_result[1]])
stable_sol = real(result[1][stablereal_result[1]])
unstable_sol = real(result[1][real_result[1] .&& .!(stablereal_result[1])])

=#
#=
real_results, real_stable_results = sweep_one_parameter(n, times, ω, r0, r1, "r0")



number_real = Vector{Int}(undef, length(r0))
number_stable_real = Vector{Int}(undef, length(r0))

for i in eachindex(r0)
    number_real[i] = count(real_results[i][1])
    number_stable_real[i] = count(real_stable_results[i][1])
end

plt = plot(r0, number_stable_real, xlabel=L"r_0", ylabel="# solutions", label="Stable Real")
#plot!(r1, number_stable, label="Real")
display(plt)

=#

real_results, real_stable_results = sweep_two_parameters(n, times, ω, r0, r1)

number_real = Array{Int}(undef, length(r1), length(r0))
number_stable_real = similar(number_real)

for i in eachindex(r1)
    for j in eachindex(r0)
        number_real[i, j] = count(real_results[i][j][1])
        number_stable_real[i, j] = count(real_stable_results[i][j][1])
    end
end

heatmap(r0, r1, number_stable_real, xlabel=L"r_0", ylabel=L"r_1", color = cgrad(:turbo, 9, categorical = true), colorbar_title="# Stable Solutions", colorbar_ticks = 2:2:18, clim=(2,18))

