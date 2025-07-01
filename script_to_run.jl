using Plots
using LaTeXStrings

#using Pkg
#Pkg.develop(path="/Users/gmariani/Documents/ETHz - PhD/Scripts/FindEquilibria")
#Pkg.instantiate()
#using FindEquilibria
include("/Users/gmariani/Documents/ETHz - PhD/Scripts/FindEquilibria/src/FindEquilibria.jl")

#=
n = 4
ω = 1.0
#r0 = 0.0:0.01:0.5
r0 = 0.0
#r1 = 0.35:0.01:0.55
r1 = 0.5
T = 2π / ω
times = T/4:0.01:T/4


result, sym = FindEquilibria.find_equilibria_series(n, times, ω, r0, r1)
aligned = FindEquilibria.align_solutions(result)
real_result = FindEquilibria.mark_real(aligned)
stablereal_result = FindEquilibria.mark_real_stable(aligned, ω, times, sym, real_result)

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
#=
real_results, real_stable_results = FindEquilibria.sweep_two_parameters(n, times, ω, r0, r1)

number_real = Array{Int}(undef, length(r1), length(r0))
number_stable_real = similar(number_real)

for i in eachindex(r1)
    for j in eachindex(r0)
        number_real[i, j] = count(real_results[i][j][1])
        number_stable_real[i, j] = count(real_stable_results[i][j][1])
    end
end

heatmap(r0, r1, number_stable_real, xlabel=L"r_0", ylabel=L"r_1", color = cgrad(:turbo, 9, categorical = true), colorbar_title="# Stable Solutions", colorbar_ticks = 2:2:64, clim=(2,64))
=#

#N=10
#FindEquilibria.find_real_equilibria_fast(n, times, ω, r0, r1; N)


#=
using CriticalTransitions, DynamicalSystems, Symbolics

@variables p_sym, u_sym[1:1]

grad = p_sym * u_sym - u_sym.^3

f_expr = build_function(grad, u_sym, [p_sym]; expression=Val{false})
fa = eval(f_expr)

function f!(u, p, t)
    return fa(u, p)
end

u0 = 1.0
p0 = 1.0

sde = CoupledSDEs(f, u0, p0; noise_strength = 0.05)
pathStruct = geometric_min_action_method(sde, [-1.0], [1.0])
println(pathStruct.path)
println(pathStruct.action)
=#



n = 4
ω = 1.0
#r0 = 0.0:0.01:0.5
r0 = 0.3
#r1 = 0.35:0.01:0.55
r1 = 0.5
T = 2π / ω
times = 0.0:0.1:T

FindEquilibria.get_solutions_flags(n, times, ω, r0, r1)