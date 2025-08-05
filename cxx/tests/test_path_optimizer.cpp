#include "PeristalticBendingRobotPathOptimizer.hpp"
#include "RodUtils.hpp"

#define N 13

void runSimulation(PeristalticBendingRobot<N>* robot, const std::vector<std::vector<Real>>& avg_pressures)
{
    PeristalticBendingRobot<N> robot_copy(robot);
    for (unsigned step = 0; step < avg_pressures.size(); step++)
    {
        const std::vector<Real>& cur_avg_pressures = avg_pressures[step];
        // fix the back actuator (actuator 0)
        int fixed_actuator_index = (cur_avg_pressures[0] > cur_avg_pressures[1]) ? 0 : 1;
        Vec6r fixed_actuator_pos_and_ori = robot.actuatorPositionAndOrientation(fixed_actuator_index);
        Vec3r fixed_actuator_position = fixed_actuator_pos_and_ori.head<3>();
        Mat3r fixed_actuator_orientation = Math::Exp_so3(fixed_actuator_pos_and_ori.tail<3>());

        std::cout << "Fixed actuator position: " << fixed_actuator_position.transpose() << std::endl;
        std::cout << "Fixed actuator rotation:\n" << fixed_actuator_orientation << std::endl;

        // run the path optimizer
        PeristalticBendingRobotPathOptimizer<N> optimizer(robot);
        std::vector<Vec2r> optimal_pressures = optimizer.findOptimalPressures(cur_avg_pressures, fixed_actuator_index, fixed_actuator_position, fixed_actuator_orientation);

        std::cout << "Actuator 0 optimal pressures: " << optimal_pressures[0].transpose() << std::endl;
        std::cout << "Actuator 1 optimal pressures: " << optimal_pressures[1].transpose() << std::endl;

        // apply the optimal pressures
        PeristalticBendingRobot_OptimizationFunctor functor(&robot_copy, optimal_pressures);
        // Set up parameters
        LBFGSpp::LBFGSParam<Real> param;
        param.epsilon = 0;
        param.max_iterations = 10000;

        // Create solver object
        LBFGSpp::LBFGSSolver<Real> solver(param);
        Real fx;
        VecXr x = robot->state().state_vec;
        try 
        {
            // solve the optimization problem
            solver.minimize(functor, x, fx);
        }
        catch(const std::runtime_error& e)
        {
            // if we don't converge, print out the error (maybe epsilon was set too small)
            // std::cout << "Error occurred: " << e.what() << std::endl;
        }

        // calculate the new center position and orientation
        Vec6r pos_and_ori = CosseratRod<N>::nodePositionAndOrientationGivenStartingNode(
            robot->actuatorNode(fixed_actuator_index),
            fixed_actuator_position,
            fixed_actuator_orientation,
            N/2,
            robot_copy.state().v1(), robot_copy.state().v2(), robot_copy.state().v3(),
            robot_copy.state().u1(), robot_copy.state().u2(), robot_copy.state().u3()
        );

        // set the robot state
        PeristalticBendingRobot<N>::State copy_state = robot_copy->state();
        copy_state.set_p(pos_and_ori.head<3>());
        copy_state.set_ori(pos_and_ori.tail<3>());
        robot->setState(copy_state);
    }
}

int main()
{
    EllipseCrossSection rod_cs(0.1, 0.1);
    EllipseCrossSection actuator_cs(0.035, 0.08);
    Real rod_length = 1.0;

    Real h = rod_length / (N-1);
    int num_segments_per_actuator = 4;
    Real actuator_length = num_segments_per_actuator*h;
    int num_actuators = (N-1) / (num_segments_per_actuator+2);

    std::cout << "Num Actuators: " << num_actuators << std::endl;

    Real E = 1e5;
    Real nu = 0.45;

    PeristalticBendingRobot<N> robot(rod_length, rod_cs, E, nu, num_actuators, actuator_length, actuator_cs);
    PeristalticBendingRobot<N>::State initial_state = robot.state();
    initial_state.set_p(Vec3r(0.5,0,rod_cs.rx()*1.1));
    initial_state.set_ori(Vec3r(-M_PI/2,0,0));
    robot.setState(initial_state);

    

    RodUtils::writeToFile("../output/peristaltic_bending.txt", robot);
}