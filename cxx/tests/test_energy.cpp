#include "common.hpp"
#include "Cosserat.hpp"

#define NUM_NODES 11

int main()
{
    EllipseCrossSection circle_cross_section(0.5, 0.5);

    Real length = 3.0;
    Real E = 3e6;
    Real nu = 0.45;
    CosseratRod<NUM_NODES> rod(length, circle_cross_section, E, nu);
    std::cout << "Total energy: " << CosseratRod<NUM_NODES>::totalEnergy(rod, Vec3r(0,0,0)) << std::endl;
    
    return EXIT_SUCCESS;
}