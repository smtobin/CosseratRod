#include "CosseratRod.hpp"
#include "CosseratRodWithCrossSectionalDeformation.hpp"
#include "CosseratRodWithCrossSectionalDeformationLinearized.hpp"
#include "RodUtils.hpp"

#include <chrono>
#include <array>

#define N 11

int main()
{
    std::string folder_path = "../output/";

    std::cout << "\n\n === N = " << N << " ===" << std::endl;
    EllipseCrossSection circle_cross_section(0.5, 0.5);

    Real length = 3.0;
    Real E = 3e6;
    Real nu = 0.45;

    CosseratRod<N> rod(length, circle_cross_section, E, nu);
    CosseratRodWithCrossSectionalDeformation<N> rod_with_deformation(length, circle_cross_section, E, nu);
    CosseratRodWithCrossSectionalDeformationLinearized<N> rod_with_deformation_linearized(length, circle_cross_section, E, nu);

    Vec3r tip_force(10000, 0, 0);
    RodUtils::solveOptimizationProblemAndWriteToFile(rod, tip_force, folder_path);
    RodUtils::solveOptimizationProblemAndWriteToFile(rod_with_deformation, tip_force, folder_path);
    RodUtils::solveOptimizationProblemAndWriteToFile(rod_with_deformation_linearized, tip_force, folder_path);
    
    return EXIT_SUCCESS;
}