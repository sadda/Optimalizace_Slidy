# # Lekce 16: Úlohy s omezeními 

# Nejdříve načtěme nutné balíčky.

using Plots
using LinearAlgebra
using Zygote
using Zygote: hessian

include("utilities.jl");

# Stejně jako v jedné z předchozích hodin uvažujme funkci $$f(x_1,x_2) = \sin(x_1+x_2) + \cos^2(x_1).$$ Definujme ji a spočtěme její derivaci a Hessián. Zároveň určeme limity pro vykreslování.

f(x) = sin(x[1] + x[2]) + cos(x[1])^2
g(x) = gradient(f, x)[1]
h(x) = hessian(f, x)

f(x1,x2) = f([x1;x2])

xlims = (-3, 1)
ylims = (-2, 1);

# # Projektované gradienty

# Projektované gradienty fungují úplně stejně jako klasický gradient descent, ale po každé iteraci se bod projektuje na množinu přípustných řešení. Funkce `optim` vrací iterace `xs` po projekci (na množině přípustných řešení) a iterace `ys` po gradientním kroku (potenciálně mimo množinu přípustných řešení).

function optim(f, g, P, x, α; max_iter=100)
    xs = zeros(length(x), max_iter+1)
    ys = zeros(length(x), max_iter)
    xs[:,1] = x
    for i in 1:max_iter
        ys[:,i] = xs[:,i] - α*g(xs[:,i])
        xs[:,i+1] = P(ys[:,i])
    end
    return xs, ys
end;

# Zaveďme ještě jednu funkci, která výstupy funkce `optim` dá za sebe. Tuto funkci budeme používat pouze pro vykreslování.

merge_iterations(xs, ys) = hcat(reshape([xs[:,1:end-1]; ys][:], 2, :), xs[:,end]);

# V prvním případě budeme minimalizovat funkci $f$ ma množině $[-1,0]^2$. Pro takovýto box se dá projekce spočíst po souřadnicích.

P(x, x_min, x_max) = min.(max.(x, x_min), x_max);

# Nyní pusťme optimalizaci.

x_min = [-1; -1]
x_max = [0; 0]

xs, ys = optim(f, g, x -> P(x,x_min,x_max), [0;-1], 0.1)

create_anim(f, xs, xlims, ylims, "Anim_PGD1.gif";
    xbounds=(x_min[1], x_max[1]),
    ybounds=(x_min[2], x_max[2]),
)

# ![](Anim_PGD1.gif)

# Vidíme, že optimalizace jede skoro celou dobu po hranici přípustné množiny. Když vykreslíme oba typy iterací, vidíme, že gradient iterace odnese mimo přípustnou množinu, ale projekce je vrátí zpět.

xys = merge_iterations(xs, ys)

create_anim(f, xys, xlims, ylims, "Anim_PGD2.gif";
    xbounds=(x_min[1], x_max[1]),
    ybounds=(x_min[2], x_max[2]),
)

# ![](Anim_PGD2.gif)

# Minimalizujme nyní stejnou funkci na kružnici se středem $c=(-1, 0)$ a poloměrem $r=1$. Projekce pouze znormuje vzdálenost bodu $x$ od středu kružnice $c$. Pro vykreslení přidejme ještě funkce na parametrizaci kružnice.

P(x, c, r) = c + r*(x - c)/norm(x - c)

c = [-1; 0]
r = 1

tbounds = 0:0.001:2π
fbounds(t,c,r) = c .+ r*[sin(t); cos(t)];

# Pusťme optimalizaci stejným stylem.

xs, ys = optim(f, g, x -> P(x,c,r), [0;-1], 0.1)

create_anim(f, xs, xlims, ylims, "Anim_PGD3.gif";
    tbounds=tbounds,
    fbounds=t->fbounds(t,c,r),
)

xys = merge_iterations(xs, ys)

create_anim(f, xys, xlims, ylims, "Anim_PGD4.gif";
    tbounds=tbounds,
    fbounds=t->fbounds(t,c,r),
)

# ![](Anim_PGD3.gif)

# ![](Anim_PGD4.gif)

# Vidíme podobné výsledky jako v případě boxu. Dostali jsme se na kružnici, po které jsme se následně pohybovali. Vypišme nyní podíl gradienty účelové funcce a omezení. Protože omezení jde napsat jako $\frac12(x-c)^2 - \frac12r^2=0$, jeho gradient je $x-c$.

grad1 = g(xs[:,end])
grad2 = xs[:,end] - c

grad1 ./ grad2

# Vzhledem k tomu, že tento podíl je (přibližně) stejný v obou složkách, dostali jsme stacionární bod. Není těžké ukázat, že toto číslo se rovná lagrangeovu multiplikátoru.


# # Sequential quadratic programming (SQP)

# Naprogramujme nyní jednoduchou verzi SQP fungující pouze pro jedno omezení.

function sqp(f, f_grad, f_hess, g, g_grad, g_hess, x, λ::Real; max_iter=100, ϵ_tol=1e-8)
    for i in 1:max_iter
        A = [f_hess(x) + λ.*g_hess(x) g_grad(x); g_grad(x)' 0]
        b = [f_grad(x) + λ.*g_grad(x); g(x)]
        if norm(g(x)) <= ϵ_tol && norm(f_grad(x) + λ.*g_grad(x)) <= ϵ_tol
            break
        end
        step = A \ b
        x -= step[1:length(x)]
        λ -= step[length(x)+1]
    end
    return x
end;

# Podobně jako u ostatních metod ukažme nyní variantu metody, která vrací všechny iterace.

function sqp(f, f_grad, f_hess, g, g_grad, g_hess, x, λ::Real; max_iter=100, ϵ_tol=1e-8)
    xs = zeros(length(x), max_iter+1)
    λs = zeros(length(λ), max_iter+1)
    xs[:,1] = x
    λs[:,1] = [λ]
    for i in 1:max_iter
        x = xs[:,i]
        λ = λs[:,i]
        if norm(g(x)) <= ϵ_tol && norm(f_grad(x) + λ.*g_grad(x)) <= ϵ_tol
            xs = xs[:,1:i]
            λs = λs[:,1:i]
            break
        end
        A = [f_hess(x) + λ.*g_hess(x) g_grad(x); g_grad(x)' 0]
        b = [f_grad(x) + λ.*g_grad(x); g(x)]
        step = A \ b
        xs[:,i+1] = xs[:,i] - step[1:length(x)]
        λs[:,i+1] = λs[:,i] - step[length(x)+1:end]
    end
    return xs, λs
end;

# Zadefinujme nyní omezení, jeho derivaci a Hessián a konečně pusťme SQP.

f_con(x,c,r) = sum((x - c).^2) - r^2
g_con(x,c,r) = gradient(x -> f_con(x,c,r), x)[1]
h_con(x,c,r) = hessian(x -> f_con(x,c,r), x)

xs, λs = sqp(f, g, h, x->f_con(x,c,r), x->g_con(x,c,r), x->h_con(x,c,r), [0;-1], 1)

create_anim(f, xs, xlims, ylims, "Anim_SQP1.gif";
    tbounds=tbounds,
    fbounds=t->fbounds(t,c,r),
)

# ![](Anim_SQP1.gif)

# Iterace konvergují opět k lokálnímu minimu. Zadefinujme Lagrangeovu funkci a její derivaci podle $x$. V optimální hodnotě, tedy poslední iteraci, je derivace Lagrangeovy funkce rovna nule, což je podmínka optimality.

L(x,λ,c,r) = f(x) + λ.*f_con(x,c,r)
L_grad(x,λ,c,r) = g(x) + λ.*g_con(x,c,r)

L_grad(xs[:,end], λs[:,end], c, r)

# Je dobré si uvědomit, že pro nastartování SQP potřebujeme jak počáteční hodnotu $x$ tak i $\lambda$. Pokud zvolíme stejné $x$ jako v předchozím případě, ale jiné $\lambda$, můžeme konvergovat do jiného bodu.

xs, λs = sqp(f, g, h, x->f_con(x,c,r), x->g_con(x,c,r), x->h_con(x,c,r), [0;-1], 3)

create_anim(f, xs, xlims, ylims, "Anim_SQP2.gif";
    tbounds=tbounds,
    fbounds=t->fbounds(t,c,r),
)

L_grad(xs[:,end], λs[:,end], c, r)

# ![](Anim_SQP2.gif)

# # Závěrem

# Toto je poslední notebook na optimalizační metody. Jaké je shrnutí? Je vždy dobré používat už naprogramované věci, protože tyto implemtace obsahují spoustu vychytávek, bez kterých optimalizace v některých případech nemusí konvergovat. Zároveň je ale podle mně důležité metodám alespoň trochu rozumět. Naše pochopení metod může ovlivnit volbu vhodného solveru, zkombinovat dva solvery dohromady nebo v případě problémů naznačit, kde je problém a co by se mělo změnit.

