using LinearAlgebra
using Plots

include("utilities.jl")

##################################
### Visualizace gradientu
##################################

f(x) = sin(x[1] + x[2]) + cos(x[1])^2
g(x) = [cos(x[1] + x[2]) - 2*cos(x[1])*sin(x[1]); cos(x[1] + x[2])]
f(x1,x2) = f([x1;x2])
g(x1,x2) = g([x1;x2])

xlim = (-3, 2)
ylim = (-2, 2)
xs = range(xlim[1], xlim[2], length = 100)
ys = range(ylim[1], ylim[2], length = 100)

plt = contourf(xs, ys, f; color = :jet)

xs = range(xlim[1], xlim[2], length = 20)
ys = range(ylim[1], ylim[2], length = 20)

α = 0.25
for x1 in xs, x2 in ys
    x = [x1; x2]
    x_grad = [x x.+α.*g(x)]

    plot!(x_grad[1, :], x_grad[2, :];
        line = (:arrow, 1, :green),
        label = "",
    )
end
display(plt)

##################################
### Gradient descent
##################################

# Zakladni varianta
function grad_descent(grad, x; α=1e-1, max_iter=100, ϵ_tol=1e-6)
    for i in 1:max_iter
        x .-= α*grad(x)
        if norm(grad(x)) <= ϵ_tol
            break
        end
    end
    return x
end

# Varianta vracejici vsechny iterace a residual
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

x0 = [0.; -1]
x, x_all, res = grad_descent(g, x0)
plot(res; yscale=:log10, label="Residual")
create_anim("Anim_GD1.gif", f, x_all, xlim, ylim)

x1, x_all1, res1 = grad_descent(g, x0; max_iter=1000)
plot(res1; yscale=:log10, label="Residual")

x2, x_all2, res2 = grad_descent(g, x0; α=1e-2)
create_anim("Anim_GD2.gif", f, x_all2, xlim, ylim)

x3, x_all3, res3 = grad_descent(g, x0; α=1e0)
create_anim("Anim_GD3.gif", f, x_all3, xlim, ylim)

x4, x_all4, res4 = grad_descent(g, x0; α=1e1)
create_anim("Anim_GD4.gif", f, x_all4, xlim, ylim)

# Zakladni Armijo varianta
function armijo(grad, x; α=1e-1, max_iter=100, ϵ_tol=1e-6, α0=1, c=1e-4)
    for i in 1:max_iter
        α = α0  
        while f(x - α*grad(x)) > f(x) - c*α*norm(grad(x))^2
            α /= 2
            if α <= 1e-8
                error("Armijo search failed")
            end
        end
        x -= α*grad(x)
        if norm(grad(x)) <= ϵ_tol
            break
        end
    end
    return x
end

# Armijo varianta vracejici vsechny iterace a residual
function armijo(grad, x; α=1e-1, max_iter=100, ϵ_tol=1e-6, α0=1, c=1e-4)
    res = zeros(max_iter)
    x_all = zeros(length(x), max_iter)
    for i in 1:max_iter
        α = α0  
        while f(x - α*grad(x)) > f(x) - c*α*norm(grad(x))^2
            α /= 2
            if α <= 1e-8
                error("Armijo search failed")
            end
        end
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

x5, x_all5, res5 = armijo(g, x0)
plot(res5; yscale=:log10, label="Residual")
create_anim("Anim_GD5.gif", f, x_all5, xlim, ylim)

##################################
### Babylonska metoda
##################################

a = 5
max_iter = 10
babs = zeros(max_iter)
babs[1] = a
for i in 1:max_iter-1
    babs[i+1] = 0.5(babs[i] + a/babs[i])
end
abs.(babs .- sqrt(a))

##################################
### Newtonova metoda
##################################

# Zakladni varianta
function newton(grad, hess, x; max_iter=100, ϵ_tol=1e-12)
    for i in 1:max_iter
        x -= hess(x) \ grad(x)
        if norm(grad(x)) <= ϵ_tol
            break
        end
    end
    return x
end

# Varianta vracejici vsechny iterace a residual
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

# Stejna funkce, ale automaticky vypocet derivaci a Hessianu
using Zygote
using Zygote: hessian

f(x) = sin(x[1] + x[2]) + cos(x[1])^2
g(x) = gradient(f, x)[1]
h(x) = hessian(f, x)

x, x_all, res = newton(g, h, x0)
plot(res; yscale=:log10, label="Residual")
create_anim("Anim_Newton1.gif", f, x_all, xlim, ylim)
h(x)
eigvals(h(x))

x, x_all, res = newton(g, h, [-0.5; 0.5])
h([-0.5; 0.5])
rank(h([-0.5; 0.5]))

x, x_all, res = newton(g, h, [-1; 0])
plot(res; yscale=:log10, label="Residual")
create_anim("Anim_Newton2.gif", f, x_all, xlim, ylim)
h(x)
eigvals(h(x))

##################################
### Porovnani metod
##################################

f(x) = exp(-x^2) - 0.5*exp(-(x-1)^2) - 0.5*exp(-(x+1)^2)
g(x) = gradient(f, x)[1]
h(x) = hessian(x -> f(x[1]), [x])[1]

xs = -4:0.01:4
plot(xs, [f; g; h], label=["f" "g" "h"])

x0 = 1
x1, x_all1 = grad_descent(g, x0)
x2, x_all2 = newton(g, h, x0)

f_neg(x) = -f(x)
g_neg(x) = -g(x)
h_neg(x) = -h(x)

grad_descent(g_neg, x)
newton(g_neg, h_neg, x)

x3, x_all3 = grad_descent(g_neg, x0)
x4, x_all4 = newton(g_neg, h_neg, x0)

plot(xs, f; label="f")
scatter!(x_all1[:], f; label="GD for f")
scatter!(x_all3[:], f; label="GD for -f")

plot(xs, f; label="f")
scatter!(x_all2[:], f; label="Newton for f")
scatter!(x_all4[:], f; label="Newton for -f")
