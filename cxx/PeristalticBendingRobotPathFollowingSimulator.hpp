#ifndef __PERISTALTIC_BENDING_ROBOT_PATH_FOLLOWING_SIMULATION_HPP
#define __PERISTALTIC_BENDING_ROBOT_PATH_FOLLOWING_SIMULATION_HPP

#include "PeristalticBendingRobot.hpp"

#include <sstream>
#include <iomanip>
#include <memory>

template<int N, int M>  // N = nodes in robot, M = nodes per actuator
class PeristalticBendingRobotPathFollowingSimulator
{

public:

    PeristalticBendingRobotPathFollowingSimulator(PeristalticBendingRobot<N>* robot, const std::vector<std::vector<Real>>& actuator_low_pressures)
        : _robot(robot), _actuator_pressures(actuator_low_pressures)
    {
        assert(_actuator_pressures.size() == robot->numActuators());

        Real h = _robot->length() / (N-1);
        Real actuator_length = h*(M-1);

        const EllipseCrossSection* robot_cs = dynamic_cast<const EllipseCrossSection*>(_robot->crossSection());
        const EllipseCrossSection* act_cs = dynamic_cast<const EllipseCrossSection*>(_robot->actuatorCrossSection());
        _single_actuator_robot = std::make_unique<PeristalticBendingRobot<M>>(
            actuator_length, *robot_cs, _robot->E(), _robot->nu(), 1, actuator_length, *act_cs);
    }

    /** Performs the optimization at each time step, using the previous time step's state as the initial state
     * for the optimization.
     * 
     * If the pressure in one of the actuators is greater than its critical pressure, the node at that actuator's
     * center is fixed. This corresponds to the outward force on the pipe by the robot being enough for static 
     * friction to prevent the contacting part from sliding along the pipe.
     */
    std::vector<typename PeristalticBendingRobot<N>::State> runSimulation();

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
    Vec2r _findPressuresForCurvature(Real low_pressure, Real desired_curvature);
    Vec2r _findPressuresForPath(Real low_pressure, int actuator_index, int node=M/2, bool from_center=false);

    Real _pathCurvature(const Vec3r& pos)
    {
        return 0.5;
    }
    Real _distanceFromPath(const Vec3r& pos)
    {
        // distance from circle with radius 1.0 centered at (1,0)
        Real radius = 2;
        Vec2r center(radius,0);
        return std::abs( (pos.head<2>()-center).norm() - radius);
    }


    PeristalticBendingRobot<N>* _robot;    // the robot
    std::vector<std::vector<Real>> _actuator_pressures; // the series of low actuator pressures for each time step - high pressures will be calculated

    std::unique_ptr<PeristalticBendingRobot<M>> _single_actuator_robot;
    
    std::vector<typename PeristalticBendingRobot<N>::State> _states;   // store the state of the robot at each time step

};

#include "PeristalticBendingRobotPathFollowingSimulator.impl.hpp"

#endif // __PERISTALTIC_BENDING_ROBOT_PATH_FOLLOWING_SIMULATION_HPP