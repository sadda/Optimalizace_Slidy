using Plots
using LinearAlgebra
using Zygote
using Zygote: hessian

include("utilities.jl")

# # PGD


f(x) = sin(x[1] + x[2]) + cos(x[1])^2
g(x) = gradient(f, x)[1]
h(x) = hessian(f, x)

f(x1,x2) = f([x1;x2])








function optim(f, g, P, x, α; max_iter=100)
    xs = zeros(length(x), max_iter+1)
    ys = zeros(length(x), max_iter)
    xs[:,1] = x
    for i in 1:max_iter
        ys[:,i] = xs[:,i] - α*g(xs[:,i])
        xs[:,i+1] = P(ys[:,i])
    end
    return xs, ys
end



function sqp(f, f_grad, f_hess, g, g_grad, g_hess, x, λ::Real; max_iter=100, ϵ_tol=1e-8)
    n = length(x)
    m = length(λ)
    xs = zeros(n, max_iter+1)
    λs = zeros(m, max_iter+1)
    xs[:,1] = x
    λs[:,1] = [λ]
    for i in 1:max_iter
        x = xs[:,i]
        λ = λs[:,i]
        println(x)
        println(λ)
        println("########")
        if norm(g(x)) <= ϵ_tol && norm(f_grad(x) + λ.*g_grad(x)) <= ϵ_tol
            xs = xs[:,1:i]
            λs = λs[:,1:i]
            break
        end
        A = [f_hess(x) + λ.*g_hess(x) g_grad(x); g_grad(x)' 0]
        b = [f_grad(x) + λ.*g_grad(x); g(x)]
        println(A \ b)
        println("########")
        step = A \ b
        xs[:,i+1] = xs[:,i] - step[1:n]
        λs[:,i+1] = λs[:,i] - step[n+1:n+m]
    end
    return xs, λs
end



merge_iterations(xs, ys) = hcat(reshape([xs[:,1:end-1]; ys][:], 2, :), xs[:,end])

# Priklad 1

P(x, x_min, x_max) = min.(max.(x, x_min), x_max)

x_min = [-1; -1]
x_max = [0; 0]

xs, ys = optim(f, g, x -> P(x,x_min,x_max), [0;-1], 0.1)

xlims = (-3, 1)
ylims = (-2, 1)

create_anim(f, xs, xlims, ylims, "Anim_PGD1.gif";
    xbounds=(x_min[1], x_max[1]),
    ybounds=(x_min[2], x_max[2]),
)


xys = merge_iterations(xs, ys)

create_anim(f, xys, xlims, ylims, "Anim_PGD2.gif";
    xbounds=(x_min[1], x_max[1]),
    ybounds=(x_min[2], x_max[2]),
)


# Priklad 2

P(x, c, r) = c + r*(x - c)/norm(x - c)

tbounds = 0:0.001:2π
fbounds(t,c,r) = c .+ r*[sin(t); cos(t)]

c = [-1; 0]
r = 1

xs, ys = optim(f, g, x -> P(x,c,r), [0;-1], 0.1)

xlims = (-3, 1)
ylims = (-2, 1)

create_anim(f, xs, xlims, ylims, "Anim_PGD3.gif";
    tbounds=tbounds,
    fbounds=t->fbounds(t,c,r),
)

grad1 = g(xs[:,end])
grad2 = xs[:,end] - c

grad1 ./ grad2

xys = merge_iterations(xs, ys)

create_anim(f, xys, xlims, ylims, "Anim_PGD4.gif";
    tbounds=tbounds,
    fbounds=t->fbounds(t,c,r),
)


xs, ys = optim(f, g, x -> P(x,c,r), [-2;1], 0.1)

create_anim(f, xs, xlims, ylims, "Anim_PGD5.gif";
    tbounds=tbounds,
    fbounds=t->fbounds(t,c,r),
)


# # SQP



f_con(x,c,r) = sum((x - c).^2) - r^2
g_con(x,c,r) = gradient(x -> f_con(x,c,r), x)[1]
h_con(x,c,r) = hessian(x -> f_con(x,c,r), x)


xs, λs = sqp(f, g, h, x->f_con(x,c,r), x->g_con(x,c,r), x->h_con(x,c,r), [0;-1], 1)

g(xs[:,end])

λs[:,end] .* g_con(xs[:,end],c,r)

create_anim(f, xs, xlims, ylims, "Anim_SQP1.gif";
    tbounds=tbounds,
    fbounds=t->fbounds(t,c,r),
)

hessian(f_con, x)






xs, λs = sqp(f, g, h, x->f_con(x,c,r), x->g_con(x,c,r), x->h_con(x,c,r), [-1;1], 1)


create_anim(f, xs, xlims, ylims, "Anim_SQP1.gif";
    tbounds=tbounds,
    fbounds=t->fbounds(t,c,r),
)


xs, λs = sqp(f, g, h, x->f_con(x,c,r), x->g_con(x,c,r), x->h_con(x,c,r), [-1;1], -2)


create_anim(f, xs, xlims, ylims, "Anim_SQP1.gif";
    tbounds=tbounds,
    fbounds=t->fbounds(t,c,r),
)



xs, λs = sqp(f, g, h, x->f_con(x,c,r), x->g_con(x,c,r), x->h_con(x,c,r), [-1;1], 3)


create_anim(f, xs, xlims, ylims, "Anim_SQP1.gif";
    tbounds=tbounds,
    fbounds=t->fbounds(t,c,r),
)

# Je dobre vedet, jak to funguje. Ale asi bych pouzil solvery, co delal nekdo jiny.



