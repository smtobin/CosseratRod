#include "CosseratRod.hpp"
#include "CosseratRodWithCrossSectionalDeformation.hpp"
#include "CosseratRodWithCrossSectionalDeformationLinearized.hpp"
#include "CosseratRodWithLinearModesOfCrossSectionalDeformation.hpp"
#include "RodUtils.hpp"

#include <chrono>
#include <array>

#define N 15

int main()
{
    std::string folder_path = "../output/";

    std::cout << "\n\n === N = " << N << " ===" << std::endl;
    EllipseCrossSection circle_cross_section(0.5, 0.5);
    RectCrossSection rect_cross_section(1.0, 0.5);

    Real length = 2.0;
    Real E = 1e5;
    Real nu = 0.3;

    CosseratRod<N> rod(length, rect_cross_section, E, nu);
    CosseratRodWithCrossSectionalDeformation<N> rod_with_deformation(length, rect_cross_section, E, nu);
    CosseratRodWithCrossSectionalDeformationLinearized<N> rod_with_deformation_linearized(length, rect_cross_section, E, nu);
    CosseratRodWithLinearModesOfCrossSectionalDeformation<N> rod_with_linear_modes(length, rect_cross_section, E, nu);

    Vec3r tip_force(0, 500, 0);
    RodUtils::solveOptimizationProblemAndWriteToFile(rod, tip_force, folder_path);
    RodUtils::solveOptimizationProblemAndWriteToFile(rod_with_deformation, tip_force, folder_path);
    RodUtils::solveOptimizationProblemAndWriteToFile(rod_with_deformation_linearized, tip_force, folder_path);
    RodUtils::solveOptimizationProblemAndWriteToFile(rod_with_linear_modes, tip_force, folder_path);
    
    return EXIT_SUCCESS;
}