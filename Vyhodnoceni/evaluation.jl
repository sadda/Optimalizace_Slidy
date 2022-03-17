using CSV
using DataFrames
using Query

function con(a::String)
    if a == "X"
        return 0
    else
        return parse(Float64, replace(a, "," => "."))
    end
end

function grade(row)
    points = row.points + 0.01

    if points < 50 || row.res3 < 25
        return "F"
    elseif points < 60
        return "E"
    elseif points < 70
        return "D"
    elseif points < 80
        return "C"
    elseif points < 90
        return "B"
    else
        return "A"
    end
end

exam_name = "zk3"
file_name = "B0B33OPT.csv"
data = CSV.read(file_name, DataFrame)

data = data[.!ismissing.(data[:, exam_name]),:]

res = @from row in data begin
    @select {
        name = row.Student,
        res1 = getproperty(row, Symbol("Sum assig")),
        res2 = getproperty(row, Symbol("Sum tests")),
        res3 = getproperty(row, Symbol(exam_name))
    }
    @collect DataFrame
end
insertcols!(res, 5, :points => res.res1 * 10 / 6 + res.res2 + res.res3)
insertcols!(res, 2, :grade => [grade(row) for row in eachrow(res)])

res_succ = res[res.grade .!= "F", :]
