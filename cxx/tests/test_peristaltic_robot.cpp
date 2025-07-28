#include "PeristalticRobot.hpp"
#include "../LBFGSpp/include/LBFGS.h"   // TODO: change this path
#include "RodUtils.hpp"

#include <chrono>

#define N 21

int main()
{
    EllipseCrossSection rod_cs(0.25, 0.25);
    EllipseCrossSection actuator_cs(0.15, 0.15);
    Real rod_length = 2.0;
    Real actuator_length = 0.7;
    int num_actuators = 2;

    Real E = 1e5;
    Real nu = 0.45;

    PeristalticRobot<N> robot(rod_length, rod_cs, E, nu, num_actuators, actuator_length, actuator_cs);

    std::vector<Real> actuation_pressures(num_actuators);
    actuation_pressures[0] = 70e3;
    actuation_pressures[1] = 70e3;
    // Real energy = robot.minimizationEnergy(actuation_pressures);

    int n_iter = 1;
    PeristalticRobot_OptimizationFunctor functor(&robot, actuation_pressures);

    // Set up parameters
    LBFGSpp::LBFGSParam<Real> param;
    param.epsilon = 0;
    param.max_iterations = 10000;

    // Create solver object
    LBFGSpp::LBFGSSolver<Real> solver(param);

    VecXr orig_x = robot.state().state_vec;

    auto t_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_iter; i++)
    {
        // initial guess is rod's original state
        VecXr x = orig_x;
        Real fx;

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
    }

    // print out info
    auto t_end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count() / 1.0e6;
    std::cout << "Elapsed time for optimization: " << time_ms / n_iter << " ms" << std::endl;

    std::cout << "Total energy: " << robot.minimizationEnergy(actuation_pressures) << std::endl;

    std::cout << "Final state:\n" << robot.state() << std::endl;

    RodUtils::writeToFile("../output/peristaltic.txt", robot);
}