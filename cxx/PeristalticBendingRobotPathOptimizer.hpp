#pragma once

#include "PeristalticBendingRobot.hpp"
#include "../LBFGSpp/include/LBFGS.h"   // TODO: change this path
#include "../alglib-cpp/src/optimization.h"

template <int N>
class PeristalticBendingRobotPathOptimizer_OptimizationFunctor;

template <int N>
class PeristalticBendingRobotPathOptimizer
{
public:
    PeristalticBendingRobotPathOptimizer(const PeristalticBendingRobot<N>* robot)
        : _robot(robot), _robot_copy(*robot)
    {

    }

    std::vector<Vec2r> findOptimalPressures(
        const std::vector<Real>& avg_pressures,
        int fixed_actuator_index, 
        const Vec3r& fixed_actuator_position,
        const Mat3r& fixed_actuator_orientation);



    // void _robotStateFromActuatorPressures(const std::vector<Vec2r>& actuation_pressures);

    Real minimizationCost(const std::vector<Vec2r>& actuation_pressures, int fixed_actuator_index, 
        const Vec3r& fixed_actuator_position,
        const Mat3r& fixed_actuator_orientation);

    VecXr minimizationGradient(const std::vector<Vec2r>& actuation_pressures, int fixed_actuator_index, 
        const Vec3r& fixed_actuator_position,
        const Mat3r& fixed_actuator_orientation);
private:
    // path functions
    // right now, path is hard-coded
    // and the path is just a vertical line through (0,0)
    Real _distanceToPath(const Vec2r& x)
    {
        return std::abs(x[0]);
    }

    Vec2r _closestPointOnPath(const Vec2r& x)
    {
        return Vec2r(0, x[1]);
    }

    Vec2r _tangentAtClosestPointOnPath(const Vec2r& x)
    {
        return Vec2r(0,1);
    }

    const PeristalticBendingRobot<N>* _robot; // for getting the current robot state
    PeristalticBendingRobot<N> _robot_copy; // for doing the test optimizations for determing the optimal pressures
};

#include "PeristalticBendingRobotPathOptimizer.impl.hpp"

/** Functor used in the LBFGS optimization of the minimiziation energy.
 * Given x (the state), the () operator computes f(x) and the gradient.
 */
template <int N>
class PeristalticBendingRobotPathOptimizer_OptimizationFunctor
{
public:
    PeristalticBendingRobotPathOptimizer_OptimizationFunctor(
        PeristalticBendingRobotPathOptimizer<N>* optimizer,
        const std::vector<Real>& avg_pressures,
        int fixed_actuator_index, const Vec3r& fixed_actuator_position, const Mat3r& fixed_actuator_orientation)
        : _optimizer(optimizer), _avg_pressures(avg_pressures),
            _fixed_actuator_index(fixed_actuator_index), _fixed_actuator_position(fixed_actuator_position),
            _fixed_actuator_orientation(fixed_actuator_orientation)
    {}

    Real operator() (const VecXr& x, VecXr& grad)
    {
        // convert vector to std::vector<Vec2r>
        std::vector<Vec2r> actuation_pressures(x.size());
        for (int i = 0; i < x.size(); i++)
        {
            actuation_pressures[i] = Vec2r(_avg_pressures[i] + 0.5*x[i], _avg_pressures[i] - 0.5*x[i]);
        }

        // compute gradient and energy
        grad = _optimizer->minimizationGradient(actuation_pressures, _fixed_actuator_index, _fixed_actuator_position, _fixed_actuator_orientation);
        Real cost = _optimizer->minimizationCost(actuation_pressures, _fixed_actuator_index, _fixed_actuator_position, _fixed_actuator_orientation);

        return cost;
    }

private:
    PeristalticBendingRobotPathOptimizer<N>* _optimizer;
    const std::vector<Real>& _avg_pressures;
    int _fixed_actuator_index;
    const Vec3r& _fixed_actuator_position;
    const Mat3r& _fixed_actuator_orientation;
};