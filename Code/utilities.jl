function create_anim(file_name, f, path, xlims, ylims; fps=15)
    xs = length(xlims) > 2 ? range(xlims[1], xlims[end]; length = 100) : range(xlims...; length = 100)
    ys = length(ylims) > 2 ? range(ylims[1], ylims[end]; length = 100) : range(ylims...; length = 100)

    plt = contourf(xs, ys, f, color = :jet, axis = false, ticks = false, cbar = false)

    # adds an empty plot to plt
    plot!(Float64[], Float64[]; line = (4, :green), label = "")

    # extracts last plot series
    plt_path = plt.series_list[end]

    # creates the  animation
    anim = Animation()
    for x in eachcol(path)
        push!(plt_path, x[1], x[2]) # add new point to plt_grad
        frame(anim)
    end
    gif(anim, file_name; fps = fps, show_msg=false)
    return nothing
end