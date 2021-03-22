using Random
using LinearAlgebra
using Plots

include("utilities.jl")

##################################
### Funkce z minule prednasky
##################################

function grad_descent(grad, x; α=1e-1, max_iter=100, ϵ_tol=1e-6)
    res = zeros(max_iter)
    x_all = zeros(length(x), max_iter)
    for i in 1:max_iter
        x_all[:,i] .= x
        x -= α*grad(x)
        res[i] = norm(grad(x))
        if norm(grad(x)) <= ϵ_tol
            res = res[1:i]
            x_all = x_all[:,1:i]
            break
        end
    end
    return x, x_all, res
end

function newton(grad, hess, x; max_iter=100, ϵ_tol=1e-12)
    res = zeros(max_iter)
    x_all = zeros(length(x), max_iter)
    for i in 1:max_iter
        x_all[:,i] .= x
        x -= hess(x) \ grad(x)
        res[i] = norm(grad(x))
        if norm(grad(x)) <= ϵ_tol
            res = res[1:i]
            x_all = x_all[:,1:i]
            break
        end
    end
    return x, x_all, res
end

##################################
### Linearni nejmensi ctverce
##################################

## Create and show data

n = 1000
xs = range(-2, 2; length=n)
ys = sin.(xs) .- 0.1*xs .+ 1

A = hcat(xs, ones(n))
scatter(A[:,1], ys, label="Data", legend=:topleft)

## Prepare functions

g(x) = A*x .- ys
g_grad(x) = A

f(x) = g(x)'*g(x) / (2*length(g(x)))
f_grad(x) = g_grad(x)'*g(x) / length(g(x))

xlim = range(0, 1; length=31)
ylim = range(0, 2; length=31)

contourf(xlim, ylim, (x,y) -> f([x,y]); color=:jet)

## Closed-form solution

x_opt1 = (A'*A) \ (A'*ys)

contourf(xlim, ylim, (x,y) -> f([x,y]); color=:jet)
scatter!([x_opt1[1]], [x_opt1[2]]; label="Optimum")

## Gradient descent

x0 = [0;0]

x_opt2, x_all2, res2 = grad_descent(f_grad, x0)

scatter(A[:,1], ys, label="Data", legend=:topleft)
plot!(xs, x -> x_opt2[1]*x + x_opt2[2], label="Fit")

create_anim("Anim_NC1.gif", (x,y) -> f([x,y]), x_all2, xlim, ylim)

## Newton-type methods

f_hess_approx(x,λ) = g_grad(x)'*g_grad(x) / length(g(x)) + λ*Diagonal(ones(length(x)))

ls_gauss_newton(f_grad, x0) = newton(f_grad, x -> f_hess_approx(x, 0), x0)
ls_levenberg_marquardt(f_grad, x0, μ) = newton(f_grad, x -> f_hess_approx(x, μ), x0)

x_opt3, x_all3, res3 = ls_gauss_newton(f_grad, x0)
x_opt4, x_all4, res4 = ls_levenberg_marquardt(f_grad, x0, 0.5)

plot(res4; yscale=:log10, label="Residual")
create_anim("Anim_NC2.gif", (x,y) -> f([x,y]), x_all4, xlim, ylim)

## Stochastic gradient descent

g(x, I) = A[I,:]*x - ys[I] 
g_grad(x, I) = A[I,:]

f(x, I) = g(x, I)'*g(x, I) / length(g(x, I))
f_grad(x, I) = g_grad(x, I)'*g(x, I) / length(g(x, I))

function stoch_grad_descent(grad, x, n; n_minibatch=8, α=1e-1, max_iter=100)
    res1 = zeros(max_iter)
    res2 = zeros(max_iter)
    x_all = zeros(length(x), max_iter)
    for i in 1:max_iter
        x_all[:,i] = x        
        I = randperm(n)[1:n_minibatch]
        x -= α*grad(x, I)
        res1[i] = norm(grad(x, I))
        res2[i] = norm(grad(x, 1:n))
    end
    return x, x_all, res1, res2
end

x_opt5, x_all5, res5_1, res5_2 = stoch_grad_descent(f_grad, x0, n)

plot(res5_1; yscale=:log10, label="Residual minibatch")
plot!(res5_2; yscale=:log10, label="Residual true")
create_anim("Anim_NC3.gif", (x,y) -> f([x,y]), x_all5, xlim, ylim)

##################################
### NELinearni nejmensi ctverce
##################################

## Non-linear regression (aka neural networks) - gradient descent

using Flux
using Flux: mse
using Base.Iterators: partition

xs_row = Float32.(reshape(xs,1,:))
ys_row = Float32.(reshape(ys,1,:))

n_hidden = 10
m = Chain(
    Dense(1, n_hidden, relu),
    Dense(n_hidden, n_hidden, relu),
    Dense(n_hidden, 1),
)

plot(xs, m(xs_row)[:], label="Initial neural network")

loss(x, y) = mse(m(x), y)

ps = params(m)
opt = Descent(1e-1)
@time for i in 1:100
    gs = gradient(ps) do
        loss(xs_row, ys_row)
    end

    Flux.update!(opt, ps, gs)
    println(loss(xs_row, ys_row))
end

plot(xs, ys, label="Data", legend=:topleft)
plot!(xs, m(xs_row)[:], label="Fit")

## Non-linear regression (aka neural networks) - stochastic gradient descent

m = Chain(
    Dense(1, n_hidden, relu),
    Dense(n_hidden, n_hidden, relu),
    Dense(n_hidden, 1),
)

batch_size = 10
batches_train = map(partition(randperm(size(ys_row, 2)), batch_size)) do inds
    return (xs_row[:, inds], ys_row[:, inds])
end

@time for _ in 1:100
    Flux.train!(loss, params(m), batches_train, opt)
    println(loss(xs_row, ys_row))
end

plot(xs, ys, label="Data", legend=:topleft)
plot!(xs, m(xs_row)[:], label="Fit")
