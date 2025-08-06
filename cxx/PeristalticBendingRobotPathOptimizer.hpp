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
    static Real _distanceToPath(const Vec2r& x)
    {
        // return std::abs(x[0]);   // vertical line through (0,0)

        // circle with center (2,0) and radius = 2
        // Vec2r center(2,0);
        // return std::abs( (center - x).norm() - 2);

        return 0;
    }

    static Real _circleAngle(const Vec2r& x, const Vec2r& center)
    {
        Vec2r diff = x - center;
        Real angle = std::atan2(diff[1], diff[0]);
        return angle;
    }

    static Vec2r _closestPointToCircle(const Vec2r& x, const Vec2r& center, Real radius)
    {
        Vec2r diff = x - center;
        Real angle = std::atan2(diff[1], diff[0]);
        Vec2r closest_point = Vec2r(std::cos(angle)*radius, std::sin(angle)*radius) + center;
        return closest_point;
    }

    static Vec2r _tangentAtClosestPointToCircle(const Vec2r& x, const Vec2r& center, Real radius, bool cw=false)
    {
        Vec2r diff = x - center;
        Real angle = std::atan2(diff[1], diff[0]);
        if (cw)
            return Vec2r(std::sin(-angle + M_PI), std::cos(-angle + M_PI));
        else
            return Vec2r(-std::sin(angle), std::cos(angle));
        
    }

    static Vec2r _closestPointOnPath(const Vec2r& x, int path_num)
    {
        // return Vec2r(0, x[1]); // vertical line through (0,0)

        // circle with center (2,0) and radius = 2
        // Vec2r center(2,0);
        // Vec2r diff = x - center;
        // Real angle = std::atan2(diff[1], diff[0]);
        // Vec2r closest_point = Vec2r(std::cos(angle)*2, std::sin(angle)*2) + center;
        // // std::cout << "x: " << x.transpose() << " closest point: " << closest_point.transpose() << std::endl;
        // return closest_point;

        // some complicated curvy path
        Real cos45 = 0.5*std::sqrt(2.0);
        Real sin45 = 0.5*std::sqrt(2.0);
        Real r1 = 4.0; Real r2 = 3.0; Real r3 = 2.5;
        Vec2r c1(0, r1);
        Vec2r c2 = c1 + r1*Vec2r(cos45, -sin45) + r2*Vec2r(cos45, -sin45);
        Vec2r c3 = c2 + r2*Vec2r(1, 0) + r3*Vec2r(1, 0);

        Vec2r cp1 = _closestPointToCircle(x, c1, r1);
        Vec2r cp2 = _closestPointToCircle(x, c2, r2);
        Vec2r cp3 = _closestPointToCircle(x, c3, r3);

        Real d1 = (x-cp1).norm();
        Real d2 = (x-cp2).norm();
        Real d3 = (x-cp3).norm();

        Real a1 = _circleAngle(x, c1);
        Real a2 = _circleAngle(x, c2);
        Real a3 = _circleAngle(x, c3);

        bool in_a1_range = (a1 >= -M_PI/2-0.1 && a1 <= -M_PI/4);
        bool in_a2_range = (a2 <= 3*M_PI/4 && a2 >= 0);
        bool in_a3_range = (a3 >= -M_PI && a3 <= 0);

        // if (path_num == 1)
        // {
        //     if (in_a1_range)
        //         return cp1;
        //     else
        //     {
        //         // path_num = 2;
        //         return cp2;
        //     }
        // }
        // else if (path_num == 2)
        // {
        //     if (!in_a2_range && in_a3_range && a2 > 3*M_PI/4)
        //     {
        //         // path_num = 3;
        //         return cp3;
        //     }
        //     else
        //         return cp2;
            
        // }
        // else
        // {
        //     return cp3;
        // }
        // std::cout << in_a1_range << " " << in_a2_range << " " << in_a3_range << std::endl;
        if (d1 <= d2 && d1 <= d3 && in_a1_range)
            return cp1;
        else if (d3 <= d1 && d3 <= d2 && in_a3_range)
            return cp3;
        else if (d2 <= d1 && d2 <= d3 && in_a2_range)
            return cp2;
        
        else if (!in_a1_range && in_a2_range && in_a3_range)
        {
            if (d3 <= d1 && d3 <= d2)
                return cp3;
            if (d2 <= d1 && d2 <= d3)
                return cp2;
        }

        else if (!in_a2_range && in_a1_range && in_a3_range)
        {
            if (d3 <= d1 && d3 <= d2)
                return cp3;
            if (d1 <= d2 && d1 <= d3)
                return cp1;
        }

        else if (!in_a3_range && in_a1_range && in_a2_range)
        {
            if (d2 <= d1 && d2 <= d3)
                return cp2;
            if (d1 <= d2 && d1 <= d3)
                return cp1;
        }
        else if (!in_a1_range && !in_a2_range)
            return cp3;
        else if (!in_a1_range && !in_a3_range)
            return cp2;
        else if (!in_a2_range && !in_a3_range)
            return cp1;
        else
        {
            std::cout << "closest point on path" << std::endl;
            std::cout << "Ahh! " << x.transpose() << " is not on a circle?" << std::endl;
            std::cout << "a1: " << a1*180/M_PI << " a2: " << a2*180/M_PI << " a3: " << a3*180/M_PI << std::endl;
            std::cout << "in a1: " << in_a1_range << " in a2: " << in_a2_range << " in a3: " << in_a3_range << std::endl;
            std::cout << "d1: " << d1 << " d2: " << d2 << " d3: " << d3 << std::endl;
            assert(0);
        }
        
    }

    static Vec2r _tangentAtClosestPointOnPath(const Vec2r& x, int path_num)
    {
        // return Vec2r(0,1); // vertical line through (0,0)

        // circle with center (2,0) and radius = 2
        // Vec2r center(2,0);
        // Vec2r diff = x - center;
        // Real angle = std::atan2(diff[1], diff[0]);
        // Vec2r tangent(std::sin(-angle + M_PI), std::cos(-angle + M_PI));
        // return tangent;

        // some complicated curvy path
        Real cos45 = 0.5*std::sqrt(2.0);
        Real sin45 = 0.5*std::sqrt(2.0);
        Real r1 = 4.0; Real r2 = 3.0; Real r3 = 2.5;
        Vec2r c1(0, r1);
        Vec2r c2 = c1 + r1*Vec2r(cos45, -sin45) + r2*Vec2r(cos45, -sin45);
        Vec2r c3 = c2 + r2*Vec2r(1, 0) + r3*Vec2r(1, 0);

        Vec2r cp1 = _closestPointToCircle(x, c1, r1);
        Vec2r cp2 = _closestPointToCircle(x, c2, r2);
        Vec2r cp3 = _closestPointToCircle(x, c3, r3);

        Real d1 = (x-cp1).norm();
        Real d2 = (x-cp2).norm();
        Real d3 = (x-cp3).norm();

        Real a1 = _circleAngle(x, c1);
        Real a2 = _circleAngle(x, c2);
        Real a3 = _circleAngle(x, c3);

        bool in_a1_range = (a1 >= -M_PI/2-0.1 && a1 <= -M_PI/4);
        bool in_a2_range = (a2 <= 3*M_PI/4 && a2 >= 0);
        bool in_a3_range = (a3 >= -M_PI && a3 <= 0);

        // if (path_num == 1)
        // {
        //     if (in_a1_range)
        //         return std::make_pair(_tangentAtClosestPointToCircle(x, c1, r1, false), 1);
        //     else
        //     {
        //         // path_num = 2;
        //         return std::make_pair(_tangentAtClosestPointToCircle(x, c2, r2, false), 2);
        //     }
        // }
        // else if (path_num == 2)
        // {
        //     if (!in_a2_range && in_a3_range)
        //     {
        //         // path_num = 3;
        //         return std::make_pair(_tangentAtClosestPointToCircle(x, c3, r3, true), 3);
        //     }
        //     else
        //         return std::make_pair(_tangentAtClosestPointToCircle(x, c2, r2, false), 2);
            
        // }
        // else
        // {
        //     return std::make_pair(_tangentAtClosestPointToCircle(x, c3, r3, true), 3);
        // }

        if (d1 <= d2 && d1 <= d3 && in_a1_range)
            return _tangentAtClosestPointToCircle(x, c1, r1, false);
        else if (d3 <= d1 && d3 <= d2 && in_a3_range)
            return _tangentAtClosestPointToCircle(x, c3, r3, false);
        else if (d2 <= d1 && d2 <= d3 && in_a2_range)
            return _tangentAtClosestPointToCircle(x, c2, r2, true);
        else if (!in_a1_range && in_a2_range && in_a3_range)
        {
            if (d3 <= d1 && d3 <= d2)
                return _tangentAtClosestPointToCircle(x, c3, r3, false);
            if (d2 <= d1 && d2 <= d3)
                return _tangentAtClosestPointToCircle(x, c2, r2, false);
        }

        else if (!in_a2_range && in_a1_range && in_a3_range)
        {
            if (d3 <= d1 && d3 <= d2)
                return _tangentAtClosestPointToCircle(x, c3, r3, false);
            if (d1 <= d2 && d1 <= d3)
                return _tangentAtClosestPointToCircle(x, c1, r1, false);
        }

        else if (!in_a3_range && in_a1_range && in_a2_range)
        {
            if (d2 <= d1 && d2 <= d3)
                return _tangentAtClosestPointToCircle(x, c2, r2, true);
            if (d1 <= d2 && d1 <= d3)
                return _tangentAtClosestPointToCircle(x, c1, r1, false);
        }
        else if (!in_a1_range && !in_a2_range)
            return _tangentAtClosestPointToCircle(x, c3, r3, false);
        else if (!in_a1_range && !in_a3_range)
            return _tangentAtClosestPointToCircle(x, c2, r2, true);
        else if (!in_a2_range && !in_a3_range)
            return _tangentAtClosestPointToCircle(x, c1, r1, false);
        else
        {
            std::cout << "tangent" << std::endl;
            std::cout << "Ahh! " << x.transpose() << " is not on a circle?" << std::endl;
            std::cout << "a1: " << a1*180/M_PI << " a2: " << a2*180/M_PI << " a3: " << a3*180/M_PI << std::endl;
            std::cout << "in a1: " << in_a1_range << " in a2: " << in_a2_range << " in a3: " << in_a3_range << std::endl;
            std::cout << "d1: " << d1 << " d2: " << d2 << " d3: " << d3 << std::endl;
            std::cout << "cp1: " << cp1.transpose() << " cp2: " << cp2.transpose() << " cp3: " << cp3.transpose() << std::endl;
            assert(0);
        }
        
        
        
        
    }

    const PeristalticBendingRobot<N>* _robot; // for getting the current robot state
    PeristalticBendingRobot<N> _robot_copy; // for doing the test optimizations for determing the optimal pressures
    int _front_actuator_path = 1;
    int _back_actuator_path = 1;
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
    {
        _actuation_pressures.resize(avg_pressures.size(), Vec2r::Zero());
    }

    Real operator() (const VecXr& x, VecXr& grad)
    {
        // convert vector to std::vector<Vec2r>
        for (int i = 0; i < x.size(); i++)
        {
            _actuation_pressures[i][0] = _avg_pressures[i] + 0.5*x[i];
            _actuation_pressures[i][1] = _avg_pressures[i] - 0.5*x[i];
        }

        // compute gradient and energy
        grad = _optimizer->minimizationGradient(_actuation_pressures, _fixed_actuator_index, _fixed_actuator_position, _fixed_actuator_orientation);
        Real cost = _optimizer->minimizationCost(_actuation_pressures, _fixed_actuator_index, _fixed_actuator_position, _fixed_actuator_orientation);

        return cost;
    }

private:
    PeristalticBendingRobotPathOptimizer<N>* _optimizer;
    std::vector<Real> _avg_pressures;
    std::vector<Vec2r> _actuation_pressures;
    int _fixed_actuator_index;
    Vec3r _fixed_actuator_position;
    Mat3r _fixed_actuator_orientation;
};