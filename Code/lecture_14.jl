# # Lekce 14: Nelineární metody nejmenších čtverců 

# Nejdříve načtěme nutné balíčky a funkce z minulé hodiny.

using Random
using LinearAlgebra
using Plots

include("utilities.jl")

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
end;

# # Nejmenší čtverce

# Úkolem dnešní hodiny bude nafitovat $n$ dvojic $(x_i, y_i)$ pomocí lineárních a nelineárních funkcí. Data $x_i$ budou rovnoměrně rozdělené na intervalu $[-2,2]$ a budeme uvažovat přesnou závislost $$y_i = \sin x_i - 0.1x_i + 1.$$ Důležité je si uvědomit, že tato funkce je neznámá, a tedy ji nemůže použít pro transformaci dat $x_i$. Při řešení se parametrizuje prostor hledaných predikcí pomocí nějaké funkce $h(w;x)$. Zde je důležité si uvědomit rozdíl mezi parametry: zatímco $x$ jsou vstupní data, $w$ jsou hledané parametry. Poté řešíme optimalizační úlohu $$\operatorname{minimalizuj}\qquad \frac 1n\sum_{i=1}^n (h(w;x_i) - y_i)^2$$ přes všechny možné parametry $w$. Chceme tedy minimalizovat vzdálenost mezi predikcí $h(w;x_i)$ a labelem $y_i$.

# # Lineární nejmenší čtverce

# Vzhledem k tomu, že máme jednorozměrný vstup, pro lineární nejmenší čtverce máme $$h(w;x) = w_1x + w_2$$. Lineární nejmenší čtverce potom mají známý tvar $$\operatorname{minimalizuj}\qquad \frac 1n\sum_{i=1}^n (w_1x_i + w_2 - y_i)^2$$.

# Vytvořme nejdříve data a vykreleme je.

n = 1000
xs = range(-2, 2; length=n)
ys = sin.(xs) .- 0.1*xs .+ 1

scatter(xs, ys, label="Data", legend=:topleft)

# Nyní zadefinujme funkce se stejným značením jako na přednášce. Nejprve $$g_i(w) = w_1x_i + w_2 - y_i$$ ukazuje chybu při fitu i-tého pozorování. Poté $$f(w) = \frac 1n\sum_{i=1}^n g_i(w)^2$$ ukazuje průměrnou kvadratickou chybu přes všechny pozorování. Je důležité si uvědomit, že proměnná $x$ už označuje vstupní data, a tedy pro optimalizovanou proměnnou musíme použít jiné písmeno, například $w$. Nyní tyto funkce zadefinujeme. Použijeme maticový zápis s maticí `A`. Zároveň spočteme gradienty $f$ i $g$.

A = hcat(xs, ones(n))

g(w) = A*w .- ys
g_grad(w) = A

f(w) = g(w)'*g(w) / (2*length(g(w)))
f_grad(w) = g_grad(w)'*g(w) / length(g(w));

# Lineární nejmenší čtverce minimalizují funkci $f$. Vykresleme tedy její vrstevnice. Znovu si uvědomme, že optimalizujeme přes proměnnou $w$.

w1lim = range(0, 1; length=31)
w2lim = range(0, 2; length=31)

contourf(w1lim, w2lim, (w1,w2) -> f([w1;w2]); color=:jet)

# Lineární nejmenší čtverce mají řešení v uzavřené formě $w=(A^\top A)^{-1}A^\top y$. Když toto řešení spočteme a vykreslíme, není překvapivé, že se nachází v minimum funkce $f$.

w_opt1 = (A'*A) \ (A'*ys)

contourf(w1lim, w2lim, (w1,w2) -> f([w1;w2]); color=:jet)
scatter!([w_opt1[1]], [w_opt1[2]]; label="Optimum")

# # Odbočka k řešení soustavy lineárních čtverců

# Pro řešení jsme použili neznámý zápis `(A'*A) \ (A'*ys)`. Tento příkaz dá stejný výsledek jako `inv(A'*A)*A'*y`. Rozdíl mezi nimi je ten, že zatímco první příkaz používá specializované algoritmy pro řešení rovnic, druhý nejdrív spočte inverzi matice `A` a teprve potom ji vynásobí vektorem `b`.

using SparseArrays 

aux_s = 1000

aux_A1 = sprandn(aux_s, aux_s, 0.001)
aux_A1 += I
aux_b = randn(aux_s);

# 


aux_A2 = Matrix(aux_A1);

#

using LinearAlgebra

norm(inv(aux_A2)*aux_b - aux_A2\aux_b)

# 

using BenchmarkTools

@btime inv($aux_A2) * $aux_b;

# 

@btime $aux_A2 \ $aux_b;


# # ???

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


mimo obor




n = 3000

A = randn(n,n);
b = randn(n);

using BenchmarkTools

@time A \ b;
@time inv(A) * b;


