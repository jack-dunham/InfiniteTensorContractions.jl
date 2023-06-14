push!(LOAD_PATH, "../src/")

using Documenter
using InfiniteTensorContractions

makedocs(
    sitename = "InfiniteTensorContractions.jl",
    authors = "Jack Dunham",
    pages = [
        "Home" => "index.md",
        "Library" => "library.md",
        "Index" => "_index.md"
    ]
)

deploydocs(
    repo = "github.com/jack-dunham/InfiniteTensorContractions.jl.git",
)
