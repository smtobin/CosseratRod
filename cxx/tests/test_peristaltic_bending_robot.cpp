#include "PeristalticBendingRobot.hpp"
#include "../LBFGSpp/include/LBFGS.h"   // TODO: change this path
#include "../alglib-cpp/src/optimization.h"
#include "RodUtils.hpp"

#include <chrono>

#define N 49

int main()
{
    EllipseCrossSection rod_cs(0.1, 0.1);
    EllipseCrossSection actuator_cs(0.035, 0.08);
    Real rod_length = 2.0;

    Real h = rod_length / (N-1);
    int num_segments_per_actuator = 4;
    Real actuator_length = num_segments_per_actuator*h;
    int num_actuators = (N-1) / (num_segments_per_actuator+2);

    std::cout << "Num Actuators: " << num_actuators << std::endl;

    Real E = 1e5;
    Real nu = 0.45;

    PeristalticBendingRobot<N> robot(rod_length, rod_cs, E, nu, num_actuators, actuator_length, actuator_cs);

    std::vector<Vec2r> actuation_pressures(num_actuators, Vec2r::Zero());
    for (int i = 0; i < num_actuators; i++)
    {
        // if (i % 2 == 0)
            actuation_pressures[i][0] = 200e3;
            actuation_pressures[i][1] = 100e3;

    }
    
    // Real energy = robot.minimizationEnergy(actuation_pressures);

    ///////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////
    std::cout << "\n=== LBFGSpp Optimization ===\n" << std::endl;

    int n_iter = 1;
    PeristalticBendingRobot_OptimizationFunctor functor(&robot, actuation_pressures);

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

    RodUtils::writeToFile("../output/peristaltic_bending.txt", robot);
}