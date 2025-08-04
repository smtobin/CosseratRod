#include "PeristalticBendingRobot.hpp"
#include "../LBFGSpp/include/LBFGS.h"   // TODO: change this path
#include "../alglib-cpp/src/optimization.h"
#include "RodUtils.hpp"
#include "CosseratRodBase.hpp"

#include <chrono>

#define N 13

EllipseCrossSection rod_cs(0.1, 0.1);
EllipseCrossSection actuator_cs(0.035, 0.08);
Real rod_length = 1.0;

Real h = rod_length / (N-1);
constexpr int num_segments_per_actuator = 4;
Real actuator_length = num_segments_per_actuator*h;
int num_actuators = (N-1) / (num_segments_per_actuator+2);

Real E = 1e5;
Real nu = 0.45;

Real distanceFromPath(const Vec2r& x)
{
    // distance from circle with radius 1.0 centered at (1,0)
    Real radius = 1;
    Vec2r center(radius,0);
    return std::abs( (x-center).norm() - radius);
}

Real pathCurvatureAtPoint(const Vec2r& x)
{
    // get closest point on path and get curvature at that point
    // path is just a circle, so curvature is just the 1/radius
    return 1;
}

Vec2r findPressures(const Vec3r& actuator_base, const Mat3r& actuator_orientation, Real low_pressure)
{
    PeristalticBendingRobot<num_segments_per_actuator+1> robot(actuator_length, rod_cs, E, nu, 1, actuator_length, actuator_cs);
    std::vector<Vec2r> actuation_pressures(1, Vec2r::Zero());

    Real min_dist = std::numeric_limits<Real>::max();
    Vec2r min_pressures(low_pressure, low_pressure);
    for (Real high_pressure = low_pressure; high_pressure < low_pressure+200e3; high_pressure+=0.5e3)
    {
        // just assume we have positive curvature for now
        actuation_pressures[0][0] = high_pressure;
        actuation_pressures[0][1] = low_pressure;
            
        PeristalticBendingRobot_OptimizationFunctor functor(&robot, actuation_pressures);

        // Set up parameters
        LBFGSpp::LBFGSParam<Real> param;
        param.epsilon = 0;
        param.max_iterations = 10000;

        // Create solver object
        LBFGSpp::LBFGSSolver<Real> solver(param);

        VecXr x = robot.state().state_vec;
        Real fx;
        try 
        {
            // solve the optimization problem
            int niter = solver.minimize(functor, x, fx);
        }
        catch(const std::runtime_error& e)
        {
        }
    
        Vec3r pos = CosseratRod<num_segments_per_actuator+1>::nodePosition(
            h, num_segments_per_actuator/2,
            actuator_base, actuator_orientation,
            robot.state().v1(), robot.state().v2(), robot.state().v3(),
            robot.state().u1(), robot.state().u2(), robot.state().u3()
        );

        Real dist = distanceFromPath(Vec2r(pos[0], pos[1]));
        if (dist < min_dist)
        {
            min_dist = dist;
            min_pressures = actuation_pressures[0];
        }
            
    }

    return min_pressures;
}

int main()
{

    PeristalticBendingRobot<N> robot(rod_length, rod_cs, E, nu, num_actuators, actuator_length, actuator_cs);
    PeristalticBendingRobot<N>::State initial_state = robot.state();
    initial_state.set_p(Vec3r(0,-0.1,rod_cs.rx()*1.1));
    initial_state.set_ori(Vec3r(-M_PI/2,0,0));
    robot.setState(initial_state);

    Vec3r actuator1_position = robot.actuatorPosition(1);

    Vec6r actuator1_base = robot.actuatorBasePositionAndOrientation(1);
    Vec2r actuator1_pressures = findPressures(actuator1_base.head<3>(), Math::Exp_so3(actuator1_base.tail<3>()), 0e3);
    std::cout << "Actuator1 pressures: " << actuator1_pressures.transpose() << std::endl;
    VecXr orig_x = robot.state().state_vec;

    std::vector<Vec2r> actuation_pressures(2, Vec2r::Zero());
    actuation_pressures[1] = actuator1_pressures;
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
    
    Vec3r new_actuator1_position = robot.actuatorPosition(1);

    std::cout << "Initial actuator1 position: " << actuator1_position.transpose() << std::endl;
    std::cout << "New actuator1 position: " << new_actuator1_position.transpose() << std::endl;

    std::cout << "Distance from path: " << distanceFromPath(Vec2r(new_actuator1_position[0], new_actuator1_position[1])) << std::endl;
}