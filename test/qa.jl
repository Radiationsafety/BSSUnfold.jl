# Package-quality checks (Aqua is recommended by the General registry FAQ).
using Aqua
using BSSUnfold

@testset "Aqua" begin
    Aqua.test_all(BSSUnfold; ambiguities = (recursive = false,))
end
