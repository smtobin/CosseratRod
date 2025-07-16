#include "../LBFGSpp/include/LBFGS.h"   // TODO: change this path
#include "Cosserat.hpp"

#include <chrono>

#define N 11

int main()
{
    EllipseCrossSection circle_cross_section(0.5, 0.5);

    Real length = 3.0;
    Real E = 3e6;
    Real nu = 0.45;
    CosseratRod<N> rod(length, circle_cross_section, E, nu);

    Vec3r applied_tip_force(10000, 0, 0);
    CosseratRodOptimizationFunctor<N> functor(&rod, applied_tip_force);

    // Set up parameters
    LBFGSpp::LBFGSParam<Real> param;
    param.epsilon = 1;
    param.max_iterations = 10000;

    // Create solver object
    LBFGSpp::LBFGSSolver<Real> solver(param);

    // initial guess is rod's current state
    VecXr x = rod.state().state_vec;
    Real fx;

    
    auto t_start = std::chrono::high_resolution_clock::now();
    try 
    {
        // solve the optimization problem
        int niter = solver.minimize(functor, x, fx);
        std::cout << "Number of iterations: " << niter << std::endl;
    }
    catch(const std::runtime_error& e)
    {
        // if we don't converge, print out the error (maybe epsilon was set too small)
        std::cout << "Error occurred: " << e.what() << std::endl;
    }

    // print out info
    auto t_end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count() / 1.0e6;
    std::cout << "Elapsed time for optimization: " << time_ms << " ms" << std::endl;

    std::cout << "Total energy: " << rod.minimizationEnergy(applied_tip_force) << std::endl;

    Vec3r tip_pos = rod.tipPosition();
    std::cout << "Tip position: (" << tip_pos[0] << ", " << tip_pos[1] << ", " << tip_pos[2] << ")" << std::endl;




    return EXIT_SUCCESS;
}