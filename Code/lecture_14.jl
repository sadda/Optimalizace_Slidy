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

# Úkolem dnešní hodiny bude nafitovat $n$ dvojic vzorků $(x_i, y_i)$ pomocí lineárních a nelineárních funkcí. Data $x_i$ budou rovnoměrně rozdělené na intervalu $[-2,2]$ a budeme uvažovat přesnou závislost $$y_i = h_{\rm true}(x_i) = \sin x_i - 0.1x_i + 1.$$ Důležité je si uvědomit, že tato funkce je neznámá, a tedy ji nemůžeme použít pro transformaci dat $x_i$. Vytvořme nejdříve data a vykresleme je.

h_true(x) = sin(x) - 0.1x + 1

n = 1000
xs = range(-2, 2; length=n)
ys = h_true.(xs)

plot(xs, ys, label="Data", legend=:topleft)

# Při řešení se parametrizuje prostor hledaných predikcí pomocí nějaké funkce $\text{predict}(w;x)$. Zde je důležité si uvědomit rozdíl mezi parametry: zatímco $x$ jsou vstupní data, $w$ jsou parametry, které budeme optimalizovat. Poté řešíme optimalizační úlohu $$\text{minimalizuj}_w\qquad \frac{1}{2n}\sum_{i=1}^n (\text{predict}(w;x_i) - y_i)^2$$ přes všechny možné parametry $w$. Chceme tedy minimalizovat vzdálenost mezi predikcí $\text{predict}(w;x_i)$ a labelem $y_i$.

# Nejprve zadefinujme nějakou obecnou predikční funkci $\text{predict}$, která se bude snažit aproximovat $h_{\rm true}$. Funkce $\text{predict}$ samozřejmě závisí na datech $x$, ale zároveň musí záviset na nějakých parametrech $w$, která budeme trénovat. Zadefinujeme dvě predikční funkce, lineární $$\text{predict}(w,x)=w_1x+w_2$$ a nelineární $$\text{predict}(w,x)=w_1\sin w_2x + w_3\cos w_4x + w_5x+ w_6.$$ První povede na lineární nejmenší čtverce, což se na přednášce dělalo několik týdnů zpátky. Zadefinujme tyto dvě funkce a spočtěme jejich derivace.

predict_lin(w,x) = w[1]*x + w[2]
predict_lin_grad(w,x) = [x 1]

predict_nonlin(w,x) = w[1]*sin(w[2]*x) + w[3]*cos(w[4]*x) + w[5]*x + w[6]
predict_nonlin_grad(w,x) = [sin(w[2]*x) x*w[1]*cos(w[2]*x) cos(w[4]*x) -x*w[3]*sin(w[4]*x) x 1];

# Nyní zadefinujme funkce se stejným značením jako na přednášce. Protože $$g_i(w) = \text{predict}(w,x_i) - y_i$$ ukazuje chybu při fitu i-tého vzorku, $$f(w) = \frac {1}{2n}\sum_{i=1}^n g_i(w)^2$$ ukazuje průměrnou kvadratickou chybu přes všechny vzorky. Je důležité si uvědomit, že proměnná $x$ už označuje vstupní data, a tedy pro optimalizovanou proměnnou jsme použili písmeno $w$. Nyní tyto funkce zadefinujeme. Zároveň spočteme gradienty $f$ i $g$.

g(w) = [predict(w,x) - y for (x,y) in zip(xs,ys)]
g_grad(w) = vcat([predict_grad(w,x) for x in xs]...)

f(w) = g(w)'*g(w) / (2*length(g(w)))
f_grad(w) = g_grad(w)'*g(w) / length(g(w))

f(x::Real,y::Real) = f([x,y]);

# V lineárních i nelineárních čtvercích chceme minimalizovat funkci $f$ a oba přístupy se liší pouze tím, jak je definovaná funkce $\text{predict}$.



# # Lineární nejmenší čtverce

# Pro lineární nejmenší čtverce definujme lineární predikci.

predict = predict_lin
predict_grad = predict_lin_grad;

# Lineární nejmenší čtverce minimalizují funkci $f$. Vykresleme tedy její vrstevnice. Znovu si uvědomme, že optimalizujeme přes proměnnou $w$.

w1lim = range(0, 1; length=31)
w2lim = range(0, 2; length=31)

contourf(w1lim, w2lim, f; color=:jet)

# Lineární nejmenší čtverce mají řešení v uzavřené formě $w=(A^\top A)^{-1}A^\top y$. Když toto řešení spočteme a vykreslíme, není překvapivé, že se nachází v minimum funkce $f$.

A = hcat(xs, ones(n))
w_opt1 = (A'*A) \ (A'*ys)

contourf(w1lim, w2lim, f; color=:jet)
scatter!([w_opt1[1]], [w_opt1[2]]; label="Optimum")

# Pro řešení jsme použili neznámý zápis `(A'*A) \ (A'*ys)`. Tento příkaz dá stejný výsledek jako `inv(A'*A)*A'*y`. Rozdíl mezi nimi je ten, že zatímco první příkaz používá specializované algoritmy pro řešení rovnic, druhý nejdrív spočte inverzi matice a teprve potom ji vynásobí vektorem. Tyto rozdíly budeme více komentuje na konci souboru.

# Z minulé hodiny máme naprogramovaný gradient descent. Pustíme ho tedy stejně jako minule.

w0 = [0;0]

w_opt2, w_all2, res2 = grad_descent(f_grad, w0);

# Dostali jsme optimální parametry, ale zajímá nás predikce. Tu dostaneme jako $w_1x+w_2$. Po vykreslení dostaneme nejlepší lineární aproximaci, která ale není moc dobrá.

plot(xs, ys, label="Data", legend=:topleft)
plot!(xs, x -> predict(w_opt2, x), label="Fit")

# Použijme opět stejnou funkci jako na minulé hodině a vykresleme konvergenci iterací.

create_anim(f, w_all2, w1lim, w2lim, "Anim_NC1.gif");

# ![](Anim_NC1.gif)

# # Metody založené na Newtonově metodě

# Pro použití Gauss-Newtonovy and Levenberg-Marquardtovy metody je dobré si uvědomit, že obě pracují stejně jako Newtonova metoda, tedy krok je $-A^{-1}\nabla f(x)$ pro nějakou matici $A$. Pro Newtonovu metodu se za $A$ bere Hessián, zatímco pro dvě výše zmíněné metoda to je nějaká jeho aproximace. Není tedy nutné psát novou optimalizační funkci, ale stačí použit již napsanou funkci `newton` se správným vstupem `h` druhých derivací. Tyto derivace jde spočíst následovně:

f_hess_approx(w,λ) = g_grad(w)'*g_grad(w) / length(g(w)) + λ*Diagonal(ones(length(w)));

# Gauss-Newtonova metoda používá `μ=0`, zatímco Levenberg-Marquardtova metoda používá `μ>0`. Poté již stačí zavolat Newtonovu metodu a dostaneme řešení.

w_opt3, w_all3, res3 = newton(f_grad, w -> f_hess_approx(w, 0), w0)
w_opt4, w_all4, res4 = newton(f_grad, w -> f_hess_approx(w, 0.5), w0);

# Je dobré si uvědomit, že Gauss-Newtonova metoda je přesná Newtonova metoda, neboť druhá derivace `g` je nulová. Vzhledem k tomu, že `f` je kvadratická, pro `w_opt3` dostáváme konvergenci v jedné iteraci. Pro Levenberg-Marquardtovu metodu je konvergence pomalejší. Navíc perturbace Hessiánu změnila superlineární (rychlou) konvergenci na pouhou lineární konvergenci.

plot(res4; yscale=:log10, label="Residual")

# Nakonec opět vykresleme jednotlivé iterace.

create_anim(f, w_all4, w1lim, w2lim, "Anim_NC2.gif");

# ![](Anim_NC2.gif)

# # Stochastický gradient descent

# Pro stochastický gradient descent definujme funkce stejně jako v první části přednášky, ale uvažujme pouze vzorky v nějaké indexové množině $I$. To odpovídá tomu, že uvažujeme pouze ty řádky matice $A$, které odpovídají těmto indexům.

g(w, I) = A[I,:]*w - ys[I] 
g_grad(w, I) = A[I,:]

f(w, I) = g(w, I)'*g(w, I) / (2*length(g(w, I)))
f_grad(w, I) = g_grad(w, I)'*g(w, I) / length(g(w, I));

# Stochastický gradient descent je stejný jako standardní gradient descent, ale gradient počítáme pouze ze zmenšeného počtu vzorků. Největší výhoda stochastického gradientu je rychlost počítání, neboť se pracuje pouze s malým počtem vzorků.

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
end;

# Pusťme nyní stochastický gradient descent.

w_opt5, w_all5, res5_1, res5_2 = stoch_grad_descent(f_grad, w0, n);

# Dále pak vykresleme rezidua na minibatchi a na celém souboru. Vidíme, že rezidua na minibatchi velmi skáčou. Důvod je ten, že minibatch je malý, což přidává nestabilitu.

plot(res5_1; yscale=:log10, label="Residual minibatch")
plot!(res5_2; yscale=:log10, label="Residual true")

# Iterace konvergují zpočátku rychle, ale když se dostanou poblíž řešení, tak začnou skákat. Důvodem je opět malá velikost minibatche. Pro konvergenci by bylo potřeba snižovat délku kroku.

create_anim(f, w_all5, w1lim, w2lim, "Anim_NC3.gif")

# ![](Anim_NC3.gif)

# I když jsme nedokonvergovali, následující obrázek ukazuje, že jsme pořád blízko dobrého řešení.

plot(xs, ys, label="Data", legend=:topleft)
plot!(xs, x -> predict(w_opt2, x), label="Fit: Optimální")
plot!(xs, x -> predict(w_opt5, x), label="Fit: SGD")

# # Nelineární nejmenší čtverce

# Nyní aktivujme nelineární predikci.

predict = predict_nonlin
predict_grad = predict_nonlin_grad;

# Zvolme počáteční bod jedniček a stejně jako v lineárním případě pusťme Levenberg-Marquardtovu metodu s parameterem $\lambda=0.001$

w0 = ones(6)

w_opt6, w_all6, res6 = newton(f_grad, w -> f_hess_approx(w, 0.001), w0);

# Po zaokrouhlední jsme dostali predikční funkci $$0.99\sin(1x) + 0.78\cos(0x) - 0.1x + 0.22 = 0.99\sin(x) - 0.1x + 1,$$ což je skoro perfektní fit. Vykresleme nyní tento fit. Data a fit jsou skoro identické.

plot(xs, ys, label="Data", legend=:topleft)
plot!(xs, x -> predict(w_opt6, x), label="Fit")

# # Neuronové sítě

# Nevýhoda předchozího přístupu je, že musíme přesně parametrizovat funkci `predict`. Ukážeme si nyní, jak nafitovat onu sinusoidu pomocí jednoduché neuronové sítě, kde tato parametrizace není nutná. Neuronová síť není nic jiného než nelineární zobrazení s nějakým speciálním předpisem. Načtěme nejdříve nutné balíky.

using Flux
using Flux: mse
using Base.Iterators: partition

# Vzhledem k tomu, že balík Flux vyžaduje, aby poslední dimenze vstupů byly vzorky, tak musíme vstupná data transformovat do řádkového vektoru. Zároveň je kvůli rychlostem výpočtu konvertujeme z `Float64` do `Float32`.

xs_row = Float32.(reshape(xs,1,:))
ys_row = Float32.(reshape(ys,1,:));

# Nyní zkonstruujme jednoduchou neuronovou síť s dvěma skrytými vrstvami.

n_hidden = 10
m = Chain(
    Dense(1, n_hidden, relu),
    Dense(n_hidden, n_hidden, relu),
    Dense(n_hidden, 1),
);

# Zadefinujme účelovou funkci jako mean square error, vytáhněme ze sítě parametry (které na začátku byly označeny jako $w$) a jako optimalizátor použijme gradient descent.

loss(x, y) = mse(m(x), y)
ps = params(m)
opt = Descent(1e-1);

# Nyní udělejme 100 iterací gradient descentu. Všimněme si, že Flux automaticky počítá derivace a provádí update parametrů.

max_iter = 100

Ls1 = zeros(max_iter)
for i in 1:max_iter
    gs = gradient(ps) do
        loss(xs_row, ys_row)
    end

    Flux.update!(opt, ps, gs)
    Ls1[i] = loss(xs_row, ys_row)
end

# Po vykreslení vidíme, že máme docela dobrý fit.

plot(xs, ys, label="Data", legend=:topleft)
plot!(xs, m(xs_row)[:], label="Fit")

# Nyní udělejme to samé, ale se stochastickým gradient descentem. Protože model `m` si v sobě nese optimalizované parametry, tak ho nejdříve znovu inicializujme.

m = Chain(
    Dense(1, n_hidden, relu),
    Dense(n_hidden, n_hidden, relu),
    Dense(n_hidden, 1),
);

# Nyní udělejme iterátor, který všechny vzorky rozdělí do minibachů, kde každý minibatch má velikost 10 vzorků.

batch_size = 10
batches_train = map(partition(randperm(size(ys_row, 2)), batch_size)) do inds
    return (xs_row[:, inds], ys_row[:, inds])
end;

# Pusťme stochastický gradient descent na 100 epoch. V jedné epoše by se optimalizátor měl podívat na každý vzorek právě jednou. Vzhledem k tomu, že máme 1000 dat a minibatch je velikosti 10, tak za 100 epoch uděláme 10000 gradientních updatů. Stochastický gradient tedy za stejný čas udělá daleko více updatů než gradient descent. I když jsou tyto updaty nepřesné, tak rychlostní bonus je většinou tak výrazný, že je dobré stochastickou verzi uvažovat.

Ls2 = zeros(max_iter)
for i in 1:max_iter
    Flux.train!(loss, params(m), batches_train, opt)
    Ls2[i] = loss(xs_row, ys_row)
end

# Porovnejme nyní běžný a stochastický gradient descent. Vidíme, že stochastická varianta má výrazně menší ztrátovou funkci.

plot(Ls1, label="GD", xlabel="Iterace", ylabel="Ztrátová funkce", legend=:topleft, yscale=:log10)
plot!(Ls2, label="SGD")

# Když vykreslíme predikci, dsotáváme skoro perfektní fit. Do odpovídá ne úplně běžné situaci, že stochastický gradient descent dokonvergoval do globálního minima.

plot(xs, ys, label="Data", legend=:topleft)
plot!(xs, x -> predict(w_opt6,x), label="Fit nonlinear")
plot!(xs, m(xs_row)[:], label="Fit neural")

# Zdálo by se, že všechno je růžové, ale co se stane, když vykreslíme fit mimo obor dat?

xs_ext = -10:0.01:10

plot(xs_ext, h_true, label="Data", legend=:topleft)
plot!(xs_ext, x -> predict(w_opt6,x), label="Fit nonlinear")
plot!(xs_ext, m(Float32.(reshape(xs_ext,1,:)))[:], label="Fit neural")

# Není vůbec dobrý. Toto je ale vlastnost všech modelů. Když učíme model na datech z intervalu $[-2,2]$ a pak ho testujeme mimo tento interval, nemůžeme očekávat, že tam bude fungovat dobře. Na drunou stranu fit pomocí parametrizované funkce je pořád dobrý.


# # Řešení soustavy lineárních rovnic

# Vraťme se nyní k tomu, jak se od sebe liší zápisy `inv(A)*b` a `A\b`. Vygenerujme náhodnou řídkou matici `aux_A1` a poté ji přetransformujme do husté matice.

using SparseArrays 

aux_s = 1000

aux_A1 = sprandn(aux_s, aux_s, 0.001)
aux_A1 += I
aux_A2 = Matrix(aux_A1);
aux_b = randn(aux_s);

# Následující kód ukazuje, že `inv(aux_A2)*aux_b` a `aux_A2\aux_b` dává stejný výsledek.

using LinearAlgebra

norm(inv(aux_A2)*aux_b - aux_A2\aux_b)

# Udělejme nyní časové porovnání pomocí balíku `BenchmarkTools`.

import BenchmarkTools: @btime

println("Dense matrix based on inv(A)*b")
@btime inv($aux_A2) * $aux_b;

println("Dense matrix based on A \\ b")
@btime $aux_A2 \ $aux_b;

println("Sparse matrix based on A \\ b")
@btime $aux_A1 \ $aux_b;

# Vidíme, že syntaxe `A \ b` je několikrát rychlejší a má menší nároky na paměť. Při použití řídké matice je rozdíl ještě markantnější, neboť `inv(A)` generuje hustou matici a není schopné využít řídkosti.


