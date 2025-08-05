#pragma once

#include "CosseratRod.hpp"


template <int N>
std::vector<Vec2r> PeristalticBendingRobotPathOptimizer<N>::findOptimalPressures(
    const std::vector<Real>& avg_pressures,
    int fixed_actuator_index, const Vec3r& fixed_actuator_position, const Mat3r& fixed_actuator_rotation
)
{
    PeristalticBendingRobotPathOptimizer_OptimizationFunctor functor(
        this, avg_pressures,
        fixed_actuator_index, fixed_actuator_position, fixed_actuator_rotation
    );

    // Set up parameters
    LBFGSpp::LBFGSParam<Real> param;
    param.epsilon = 0;
    param.max_iterations = 10000;

    // Create solver object
    LBFGSpp::LBFGSSolver<Real> solver(param);

    VecXr x = VecXr::Zero(avg_pressures.size());
    Real fx;

    try 
    {
        // solve the optimization problem
        solver.minimize(functor, x, fx);
    }
    catch(const std::runtime_error& e)
    {
        std::cout << "Error in findOptimalPressures: " << e.what() << std::endl;
    }

    // std::cout << "Optimal pressure differentials: " << x.transpose() << std::endl;

    // output the optimal actuation pressures, not just the differentials
    // can easily get these from the average pressures that were passed in
    std::vector<Vec2r> actuation_pressures(avg_pressures.size());
    for(unsigned i = 0; i < actuation_pressures.size(); i++)
    {
        actuation_pressures[i] = Vec2r(avg_pressures[i] + 0.5*x[i], avg_pressures[i] - 0.5*x[i]);
    }

    return actuation_pressures;
}

template <int N>
Real PeristalticBendingRobotPathOptimizer<N>::minimizationCost(
    const std::vector<Vec2r>& actuation_pressures,
    int fixed_actuator_index, const Vec3r& fixed_actuator_position, const Mat3r& fixed_actuator_rotation)
{
    PeristalticBendingRobot_OptimizationFunctor functor(&_robot_copy, actuation_pressures);

    // Set up parameters
    LBFGSpp::LBFGSParam<Real> param;
    param.epsilon = 0;
    param.max_iterations = 10000;

    // Create solver object
    LBFGSpp::LBFGSSolver<Real> solver(param);

    VecXr x = _robot_copy.state().state_vec;
    Real fx;

    try 
    {
        // solve the optimization problem
        solver.minimize(functor, x, fx);
    }
    catch(const std::runtime_error& e)
    {
    }

    // evaluate the cost
    int free_actuator_index = (fixed_actuator_index == 0) ? 1 : 0;

    Vec6r free_actuator_pos = CosseratRod<N>::nodePositionAndOrientationGivenStartingNode(
        _robot_copy.length() / (N-1),
        _robot_copy.actuatorNode(fixed_actuator_index), fixed_actuator_position, fixed_actuator_rotation,
        _robot_copy.actuatorNode(free_actuator_index),
        _robot_copy.state().v1(), _robot_copy.state().v2(), _robot_copy.state().v3(), 
        _robot_copy.state().u1(), _robot_copy.state().u2(), _robot_copy.state().u3()
    );

    Mat3r free_orientation = Math::Exp_so3(free_actuator_pos.tail<3>());
    Vec2r robot_tangent = free_orientation.col(2).head<2>();
    Vec2r robot_position = free_actuator_pos.head<2>();

    Vec2r path_tangent = _tangentAtClosestPointOnPath(robot_position);
    Vec2r path_position = _closestPointOnPath(robot_position);

    Mat4r W = 1e6*Vec4r(5, 5, 1, 1).asDiagonal();
    Vec4r vec;
    vec.head<2>() = path_tangent - robot_tangent;
    vec.tail<2>() = path_position - robot_position;
    return vec.transpose() * W * vec;
}

template <int N>
VecXr PeristalticBendingRobotPathOptimizer<N>::minimizationGradient(
    const std::vector<Vec2r>& actuation_pressures,
    int fixed_actuator_index, const Vec3r& fixed_actuator_position, const Mat3r& fixed_actuator_rotation)
{
    VecXr grad(actuation_pressures.size());
    std::vector<Vec2r> new_actuation_pressures(actuation_pressures);
    Real pressure_delta = 1.0; // Pa
    Real orig_cost = minimizationCost(actuation_pressures, fixed_actuator_index, fixed_actuator_position, fixed_actuator_rotation);
    for (unsigned i = 0; i < actuation_pressures.size(); i++)
    {
        new_actuation_pressures[i][0] += pressure_delta/2;
        new_actuation_pressures[i][1] -= pressure_delta/2;
        Real new_cost = minimizationCost(new_actuation_pressures, fixed_actuator_index, fixed_actuator_position, fixed_actuator_rotation);
        
        // std::cout << "cost diff: " << new_cost - orig_cost << std::endl;
        grad[i] = (new_cost - orig_cost) / pressure_delta;
        new_actuation_pressures[i][0] -= pressure_delta/2;
        new_actuation_pressures[i][1] += pressure_delta/2;
    }

    return grad;
}