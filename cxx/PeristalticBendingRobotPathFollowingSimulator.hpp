#ifndef __PERISTALTIC_BENDING_ROBOT_PATH_FOLLOWING_SIMULATION_HPP
#define __PERISTALTIC_BENDING_ROBOT_PATH_FOLLOWING_SIMULATION_HPP

#include "PeristalticBendingRobot.hpp"

#include <sstream>
#include <iomanip>

template<int N>
class PeristalticBendingRobotPathFollowingSimulator
{

public:

    PeristalticBendingRobotPathFollowingSimulator(PeristalticBendingRobot<N>* robot, const std::vector<std::vector<Real>>& actuator_low_pressures)
        : _robot(robot), _actuator_pressures(actuator_low_pressures)
    {
        assert(_actuator_pressures.size() == robot->numActuators());
    }

    /** Performs the optimization at each time step, using the previous time step's state as the initial state
     * for the optimization.
     * 
     * If the pressure in one of the actuators is greater than its critical pressure, the node at that actuator's
     * center is fixed. This corresponds to the outward force on the pipe by the robot being enough for static 
     * friction to prevent the contacting part from sliding along the pipe.
     */
    std::vector<typename PeristalticBendingRobot<N>::State> runSimulation(int num_steps);

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
                    file << _actuator_pressures[a][i].transpose() << " ";
                }
                file << "\n" << _states[i].state_vec;
            }
        }
    }

private:
    


    PeristalticBendingRobot<N>* _robot;    // the robot
    std::vector<std::vector<Real>> _actuator_pressures; // the series of actuator pressures for each time step
    
    std::vector<typename PeristalticBendingRobot<N>::State> _states;   // store the state of the robot at each time step

};

#include "PeristalticBendingRobotPathFollowingSimulator.impl.hpp"

#endif // __PERISTALTIC_BENDING_ROBOT_PATH_FOLLOWING_SIMULATION_HPP