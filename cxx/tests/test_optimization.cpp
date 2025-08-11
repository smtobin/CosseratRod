#include "CosseratRod.hpp"
#include "CosseratRodWithCrossSectionalDeformation.hpp"
#include "CosseratRodWithCrossSectionalDeformationLinearized.hpp"
#include "CosseratRodWithLinearModesOfCrossSectionalDeformation.hpp"
#include "RodUtils.hpp"

#include <chrono>
#include <array>

#define N_ITER 100

// constexpr std::array<int, 5> NODE_NUMBERS = {5, 10, 20, 50, 100};
constexpr std::array<int, 1> NODE_NUMBERS = {21};

template <int N>
void benchmarkRods()
{
    std::cout << "\n\n === N = " << N << " ===" << std::endl;
    EllipseCrossSection cross_section(0.5, 0.5);
    // RectCrossSection cross_section(1.0, 0.5);

    Real length = 2.0;
    Real E = 1e5;
    Real nu = 0.49;

    // CosseratRod<N> rod(length, cross_section, E, nu);
    CosseratRod<N> rod1(length, cross_section, E, nu);
    CosseratRod<N> rod2(length, cross_section, E, nu);
    CosseratRod<N> rod3(length, cross_section, E, nu);
    CosseratRod<N> rod4(length, cross_section, E, nu);
    CosseratRodWithCrossSectionalDeformationLinearized<N> rod_with_deformation_linearized_no_bending_correction(
        length, cross_section, E, nu, true, false, false);
    CosseratRodWithCrossSectionalDeformationLinearized<N> rod_with_deformation_linearized(length, cross_section, E, nu);
    CosseratRodWithCrossSectionalDeformation<N> rod_with_deformation(length, cross_section, E, nu, true, false, true);
    // CosseratRodWithCrossSectionalDeformation<N> rod_with_deformation1(length, cross_section, E, nu, true, true, true);
    // CosseratRodWithCrossSectionalDeformation<N> rod_with_deformation2(length, cross_section, E, nu, true, true, true);
    // CosseratRodWithCrossSectionalDeformation<N> rod_with_deformation3(length, cross_section, E, nu, true, true, true);
    // CosseratRodWithCrossSectionalDeformation<N> rod_with_deformation4(length, cross_section, E, nu, true, true, true);
    CosseratRodWithLinearModesOfCrossSectionalDeformation<N> rod_with_linear_modes(length, cross_section, E, nu);


    // Vec3r tip_force(0,500,0);

    Vec3r tip_force1(0, 0, 50e3);
    Vec3r tip_force2(0, 0, 20e3);
    Vec3r tip_force3(0, 0, -5e3);
    Vec3r tip_force4(0, 0, -15e3);
    const std::string folder_path("../output/0.5x2cyl_N=21_E=1e5_nu=0.49/");
    // std::cout << "\n=== Standard Cosserat Rod ===" << std::endl;
    // RodUtils::solveOptimizationProblemAndWriteToFile(rod, tip_force, folder_path);
    // RodUtils::solveOptimizationProblem(rod, tip_force, N_ITER);
    // std::cout << "\n=== Linearized Constant Modes of Cross-Sectional Deformation No Bending Correction ===" << std::endl;
    // RodUtils::solveOptimizationProblemAndWriteToFile(rod_with_deformation_linearized_no_bending_correction, tip_force, folder_path);
    // RodUtils::solveOptimizationProblem(rod_with_deformation_linearized_no_bending_correction, tip_force, N_ITER);
    // std::cout << "\n=== Linearized Constant Modes of Cross-Sectional Deformation ===" << std::endl;
    // RodUtils::solveOptimizationProblemAndWriteToFile(rod_with_deformation_linearized, tip_force, folder_path);
    // RodUtils::solveOptimizationProblem(rod_with_deformation_linearized, tip_force, N_ITER);

    // std::cout << "\n=== Constant Modes of Cross-Sectional Deformation === " << std::endl;
    // RodUtils::solveOptimizationProblemAndWriteToFile(rod_with_deformation, tip_force, folder_path);
    // RodUtils::solveOptimizationProblem(rod_with_deformation, tip_force, N_ITER);
    // std::cout << "\n=== Linear Modes of Cross-Sectional Deformation ===" << std::endl;
    // RodUtils::solveOptimizationProblemAndWriteToFile(rod_with_linear_modes, tip_force, folder_path);
    // RodUtils::solveOptimizationProblem(rod_with_linear_modes, tip_force, N_ITER);

    RodUtils::solveOptimizationProblemAndWriteToFile(rod1, tip_force1, folder_path);
    RodUtils::solveOptimizationProblemAndWriteToFile(rod2, tip_force2, folder_path);
    RodUtils::solveOptimizationProblemAndWriteToFile(rod3, tip_force3, folder_path);
    RodUtils::solveOptimizationProblemAndWriteToFile(rod4, tip_force4, folder_path);
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