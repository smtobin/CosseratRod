#ifndef __PERISTALTIC_ROBOT_GAIT_OPTIMIZER
#define __PERISTALTIC_ROBOT_GAIT_OPTIMIZER

#include "PeristalticRobotSimulator.hpp"

template<int N>
class PeristalticRobotGaitOptimizer
{
public:
    /** Represents a gait cycle for a 2-actuator peristaltic robot.
     * Each actuator follows a trapezoidal wave.
     *              __________
     * Actuator 1: /          \__________
     *                       _____
     * Actuator 2: _________/     \______
     * 
     * Actuator 1 can be parameterized by (hi pressure, low pressure, number of high states)
     * Actuator 2 can be parameterized by (hi pressure, low pressure, number of initial low states, number of final low states)
     */
    struct GaitCycle
    {
        Real actuator1_minP;
        Real actuator1_maxP;
        Real actuator1_numHighStates;

        Real actuator2_minP;
        Real actuator2_maxP;
        Real actuator2_numLowStates1;
        Real actuator2_numLowStates2;
    };

    PeristalticRobotGaitOptimizer(PeristalticRobot<N>* robot, Real min_actuator_pressure, Real max_actuator_pressure)
        : _robot(robot), _min_actuator_pressure(min_actuator_pressure), _max_actuator_pressure(max_actuator_pressure)
    {

    }

    Real distanceTraveledWithGaitCycle(const GaitCycle& gait_cycle)
    {
        return 0;
    }


private:
    // std::vector<std::vector<Real>> _actuationPressuresFromGaitCycle(const GaitCyle& gait_cycle)
    // {

    // }


    PeristalticRobot<N>* _robot;
    Real _min_actuator_pressure;
    Real _max_actuator_pressure;
    
    constexpr static int MAX_PRESSURE_CHANGE = 10e3;
};

#endif // __PERISTALTIC_ROBOT_GAIT_OPTIMIZER