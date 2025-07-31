#ifndef __PERISTALTIC_ROBOT_SIMULATOR_HPP
#define __PERISTALTIC_ROBOT_SIMULATOR_HPP

#include "PeristalticRobot.hpp"

#include <sstream>
#include <iomanip>

template<int N>
class PeristalticRobotPipeSimulator
{

public:

    PeristalticRobotPipeSimulator(PeristalticRobot<N>* robot, const std::vector<std::vector<Real>>& actuator_pressures, Real pipe_radius, Real critical_radius_ratio)
        : _robot(robot), _actuator_pressures(actuator_pressures), _pipe_radius(pipe_radius), _critical_radius_ratio(critical_radius_ratio)
    {
        assert(_actuator_pressures.size() == robot->numActuators());

        _critical_pressures.resize(robot->numActuators());
        _findCriticalPressures();
    }

    /** Performs the optimization at each time step, using the previous time step's state as the initial state
     * for the optimization.
     * 
     * If the pressure in one of the actuators is greater than its critical pressure, the node at that actuator's
     * center is fixed. This corresponds to the outward force on the pipe by the robot being enough for static 
     * friction to prevent the contacting part from sliding along the pipe.
     */
    std::vector<typename PeristalticRobot<N>::State> runSimulation();

    /** Writes the finished simulation results to a series of output files, one for each step. 
     * Make sure to call this AFTER calling runSimulation() */
    void writeToFile(const std::string& output_folder) const
    {
        std::string robot_filename = output_folder + "robot.txt";
        _robot->writeToFile(robot_filename);
        for (unsigned i = 0; i < _states.size(); i++)
        {
            std::stringstream ss;
            ss << std::setw(4) << std::setfill('0') << i;
            std::string filename = output_folder + "step" + ss.str() + ".txt";
            std::ofstream file(filename);
            if (file.is_open())
            {
                for (int a = 0; a < _robot->numActuators(); a++)
                {
                    file << _actuator_pressures[a][i] << " ";
                }
                file << "\n" << _states[i].state_vec;
            }
        }
    }

private:
    /** Automatically determines the "critical pressure" for each actuator, i.e. the pressure at which the
     * unconstrained radius of the robot around the actuator in question is some fraction larger than the pipe radius.
     * 
     * To do this we just test pressures in increasing increments until the radius of the robot passes some threshold.
     */
    void _findCriticalPressures(Real pressure_increment=5e3);
    


    PeristalticRobot<N>* _robot;    // the robot
    std::vector<std::vector<Real>> _actuator_pressures; // the series of actuator pressures for each time step
    Real _pipe_radius;  // the radius of the pipe the robot is in
    Real _critical_radius_ratio;    // the ratio of (unconstrained robot radius)/(pipe radius) past which the robot doesn't slide along the pipe
    std::vector<Real> _critical_pressures;  // the critical pressures for each actuator, past which the actuator "sticks" to the wall of the pipe and doesn't slide

    std::vector<typename PeristalticRobot<N>::State> _states;   // store the state of the robot at each time step

};

#include "PeristalticRobotPipeSimulator.impl.hpp"


#endif // __PERISTALTIC_ROBOT_SIMULATOR_HPP