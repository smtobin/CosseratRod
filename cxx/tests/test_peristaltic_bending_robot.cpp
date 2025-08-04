#include "PeristalticBendingRobot.hpp"
#include "../LBFGSpp/include/LBFGS.h"   // TODO: change this path
#include "../alglib-cpp/src/optimization.h"
#include "RodUtils.hpp"

#include <chrono>

#define N 7

Real findCorrespondingHighPressureForCurvature(PeristalticBendingRobot<N>* robot, Real low_pressure, Real desired_max_curvature)
{
    std::vector<Vec2r> actuation_pressures(1, Vec2r::Zero());

    Real last_eff_curvature = 0;
    for (Real high_pressure = low_pressure; high_pressure < low_pressure+200e3; high_pressure+=0.5e3)
    {
        if (desired_max_curvature < 0)
        {
            actuation_pressures[0][0] = low_pressure;
            actuation_pressures[0][1] = high_pressure;
        }
        else
        {
            actuation_pressures[0][0] = high_pressure;
            actuation_pressures[0][1] = low_pressure;
        }
            

        int n_iter = 1;
        PeristalticBendingRobot_OptimizationFunctor functor(robot, actuation_pressures);

        // Set up parameters
        LBFGSpp::LBFGSParam<Real> param;
        param.epsilon = 0;
        param.max_iterations = 10000;

        // Create solver object
        LBFGSpp::LBFGSSolver<Real> solver(param);

        VecXr orig_x = robot->state().state_vec;

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
            }
            catch(const std::runtime_error& e)
            {
                // if we don't converge, print out the error (maybe epsilon was set too small)
                // std::cout << "Error occurred: " << e.what() << std::endl;
            }
        }

        Real eff_curvature;
        if (desired_max_curvature < 0)
        {
            eff_curvature = robot->state().u2().minCoeff() / robot->state().v3().maxCoeff();
        }
        else
        {
            eff_curvature = robot->state().u2().maxCoeff() / robot->state().v3().maxCoeff();
        }
        if (std::abs(eff_curvature) >= std::abs(desired_max_curvature))
        {
            return high_pressure;
        }
            
    }

    return low_pressure;
}

int main()
{
    EllipseCrossSection rod_cs(0.1, 0.1);
    EllipseCrossSection actuator_cs(0.035, 0.08);
    // Real rod_length = 2.0;

    
    int num_segments_per_actuator = 4;
    Real actuator_length = 0.166666;
    int num_actuators = 1;
    Real h = actuator_length / num_segments_per_actuator;
    Real rod_length = h*(N-1);

    std::cout << "Rod length: " << rod_length << std::endl;
    std::cout << "h: " << rod_length / (N-1) << std::endl;
    std::cout << "Num Actuators: " << num_actuators << std::endl;

    Real E = 1e5;
    Real nu = 0.45;

    PeristalticBendingRobot<N> robot(rod_length, rod_cs, E, nu, num_actuators, actuator_length, actuator_cs);

    VecXr orig_x = robot.state().state_vec;

    Real desired_max_curvature = -0.5;
    std::vector<Real> low_pressures = {0e3, 50e3, 100e3, 150e3};
    std::vector<Real> high_pressures;
    
    std::cout << "For desired curvature=" << desired_max_curvature << ": " << std::endl;
    for(const auto& pressure : low_pressures)
    {
        Real high_pressure = findCorrespondingHighPressureForCurvature(&robot, pressure, desired_max_curvature);
        high_pressures.push_back(high_pressure);
        std::cout << "  Corresponding high pressure for low pressure of " << pressure << " Pa: " << high_pressure << std::endl;
    }

    std::vector<Vec2r> actuation_pressures(1);
    actuation_pressures[0][0] = low_pressures[0];
    actuation_pressures[0][1] = high_pressures[0];
    PeristalticBendingRobot_OptimizationFunctor functor(&robot, actuation_pressures);

    // Set up parameters
    LBFGSpp::LBFGSParam<Real> param;
    param.epsilon = 0;
    param.max_iterations = 10000;

    // Create solver object
    LBFGSpp::LBFGSSolver<Real> solver(param);
    Real fx;

    try 
    {
        // solve the optimization problem
        int niter = solver.minimize(functor, orig_x, fx);
    }
    catch(const std::runtime_error& e)
    {
        // if we don't converge, print out the error (maybe epsilon was set too small)
        // std::cout << "Error occurred: " << e.what() << std::endl;
    }

    std::cout << robot.state() << std::endl;
    
    
}