#include "CosseratRod.hpp"
#include "CosseratRodWithCrossSectionalDeformation.hpp"
#include "CosseratRodWithCrossSectionalDeformationLinearized.hpp"
#include "RodUtils.hpp"

#include <chrono>
#include <array>

#define N_ITER 10

constexpr std::array<int, 4> NODE_NUMBERS = {5, 10, 25, 50};

template <int N>
void benchmarkRods()
{
    std::cout << "\n\n === N = " << N << " ===" << std::endl;
    EllipseCrossSection circle_cross_section(0.5, 0.5);

    Real length = 3.0;
    Real E = 3e6;
    Real nu = 0.45;

    CosseratRod<N> rod(length, circle_cross_section, E, nu);
    CosseratRodWithCrossSectionalDeformation<N> rod_with_deformation(length, circle_cross_section, E, nu);
    CosseratRodWithCrossSectionalDeformationLinearized<N> rod_with_deformation_linearized(length, circle_cross_section, E, nu);

    Vec3r tip_force(10000, 0, 0);
    std::cout << "\n=== Standard Cosserat Rod ===" << std::endl;
    RodUtils::solveOptimizationProblem(rod, tip_force, N_ITER);
    std::cout << "\n=== Linearized Constant Modes of Cross-Sectional Deformation ===" << std::endl;
    RodUtils::solveOptimizationProblem(rod_with_deformation_linearized, tip_force, N_ITER);
    std::cout << "\n=== Constant Modes of Cross-Sectional Deformation === " << std::endl;
    RodUtils::solveOptimizationProblem(rod_with_deformation, tip_force, N_ITER);
}

template<std::size_t I = 0>
void runBenchmarks() {
    if constexpr (I < NODE_NUMBERS.size()) {
        benchmarkRods<NODE_NUMBERS[I]>();
        runBenchmarks<I + 1>();
    }
}

int main()
{
    
    runBenchmarks();
    return EXIT_SUCCESS;
}