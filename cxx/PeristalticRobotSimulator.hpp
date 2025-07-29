#ifndef __PERISTALTIC_ROBOT_SIMULATOR_HPP
#define __PERISTALTIC_ROBOT_SIMULATOR_HPP

#include "PeristalticRobot.hpp"

template<int N>
class PeristalticRobotSimulator
{

public:

    PeristalticRobotSimulator(PeristalticRobot<N>* robot, const std::vector<std::vector<Real>>& actuator_pressures, Real pipe_radius, Real critical_radius_ratio)
        : _robot(robot), _actuator_pressures(actuator_pressures), _pipe_radius(pipe_radius), _critical_radius_ratio(critical_radius_ratio)
    {
        assert(_actuator_pressures.size() == robot->numActuators());

        _critical_pressures.resize(robot->numActuators());
        _findCriticalPressures();
    }

    std::vector<typename PeristalticRobot<N>::State> runSimulation();

private:
    void _findCriticalPressures(Real pressure_increment=5e3);
    


    PeristalticRobot<N>* _robot;
    std::vector<std::vector<Real>> _actuator_pressures;
    Real _pipe_radius;
    Real _critical_radius_ratio;
    std::vector<Real> _critical_pressures;

};

#include "PeristalticRobotSimulator.impl.hpp"


#endif // __PERISTALTIC_ROBOT_SIMULATOR_HPP