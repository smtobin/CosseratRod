#include "PeristalticBendingRobotPathOptimizer.hpp"
#include "RodUtils.hpp"

#include <iomanip>
#include <filesystem>

#define N 13

void runSimulation(PeristalticBendingRobot<N>* robot, const std::vector<std::vector<Real>>& avg_pressures)
{
    // write robot to file
    const std::string output_folder = "../output/sim/";
    std::filesystem::remove_all(output_folder);
    std::filesystem::create_directory(output_folder);
    std::string robot_filename = output_folder + "robot.txt";
    robot->writeToFile(robot_filename);

    PeristalticBendingRobotPathOptimizer<N> optimizer(robot);

    PeristalticBendingRobot<N> robot_copy(*robot);
    for (unsigned step = 0; step < avg_pressures.size(); step++)
    {
        const std::vector<Real>& cur_avg_pressures = avg_pressures[step];
        // fix the back actuator (actuator 0)
        int fixed_actuator_index = (cur_avg_pressures[0] > cur_avg_pressures[1]) ? 0 : 1;
        Vec6r fixed_actuator_pos_and_ori = robot->actuatorPositionAndOrientation(fixed_actuator_index);
        Vec3r fixed_actuator_position = fixed_actuator_pos_and_ori.head<3>();
        Mat3r fixed_actuator_orientation = Math::Exp_so3(fixed_actuator_pos_and_ori.tail<3>());

        std::cout << "step " << step << std::endl;
        // std::cout << "Fixed actuator position: " << fixed_actuator_position.transpose() << std::endl;
        // std::cout << "Fixed actuator rotation:\n" << fixed_actuator_orientation << std::endl;

        // run the path optimizer
        
        std::vector<Vec2r> optimal_pressures = optimizer.findOptimalPressures(cur_avg_pressures, fixed_actuator_index, fixed_actuator_position, fixed_actuator_orientation);

        // std::cout << "Actuator 0 optimal pressures: " << optimal_pressures[0].transpose() << std::endl;
        // std::cout << "Actuator 1 optimal pressures: " << optimal_pressures[1].transpose() << std::endl;

        // apply the optimal pressures to the robot copy to get curvature and cross-section deformation
        if (robot->numActuators() == 2)
        {
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
                robot->length() / (N-1),
                robot->actuatorNode(fixed_actuator_index),
                fixed_actuator_position,
                fixed_actuator_orientation,
                N/2,
                robot_copy.state().v1(), robot_copy.state().v2(), robot_copy.state().v3(),
                robot_copy.state().u1(), robot_copy.state().u2(), robot_copy.state().u3()
            );

            // set the robot state
            PeristalticBendingRobot<N>::State copy_state = robot_copy.state();
            copy_state.set_p(pos_and_ori.head<3>());
            copy_state.set_ori(pos_and_ori.tail<3>());
            robot->setState(copy_state);
        }
        else
        {
            // try to make it general
            int num_actuators = robot->numActuators();

            // set up optimization
            int num_states = PeristalticBendingRobot<N>::State::NumStates;
            
            // scaling
            VecXr ones = VecXr::Ones(num_states);
            alglib::real_1d_array s;
            s.setcontent(num_states, ones.data());
            // optimizer parameters
            double epsx = 0.000001;
            alglib::ae_int_t maxits = 0;
            alglib::minnlcstate state;

            // set constraint bounds
            alglib::real_1d_array nl, nu;
            nl.setlength(num_actuators*6);
            nu.setlength(num_actuators*6);
            // default bounds are [-inf, +inf]
            for (int i = 0; i < num_actuators*6; i++)
            {
                nl[i] = std::numeric_limits<Real>::lowest();
                nu[i] = std::numeric_limits<Real>::max();
            }

            alglib::real_1d_array x0;
            x0.setcontent(num_states, robot->state().state_vec.data());
            // create optimizer object
            alglib::minnlccreate(num_states, x0, state);
            alglib::minnlcsetcond(state, epsx, maxits);
            alglib::minnlcsetscale(state, s);
            alglib::minnlcsetalgosqp(state);


            // calculate current actuator positions
            std::vector<Vec6r> current_actuator_positions(robot->numActuators());
            for (int a = 0; a < robot->numActuators(); a++)
            {
                current_actuator_positions[a] = robot->actuatorPositionAndOrientation(a);
                std::cout << "Actuator " << a << " position: " << current_actuator_positions[a].transpose() << std::endl;
            }

            // find the max pressure
            Real max_pressure = 0;
            for (int a = 0; a < robot->numActuators(); a++)
            {
                if (optimal_pressures[a].mean() > max_pressure)
                    max_pressure = optimal_pressures[a].mean();
            }
            // fix an actuator if it has the max pressure
            // keep track of actuators that are "free" i.e. allowed to move
            std::vector<int> free_actuator_indices;
            for (int a = 0; a < robot->numActuators(); a++)
            {
                if (optimal_pressures[a].mean() == max_pressure && max_pressure > 100e3)
                {
                    nl[6*a] = 0; nl[6*a+1] = 0; nl[6*a+2] = 0; nl[6*a+3] = 0; nl[6*a+4] = 0; nl[6*a+5] = 0;
                    nu[6*a] = 0; nu[6*a+1] = 0; nu[6*a+2] = 0; nu[6*a+3] = 0; nu[6*a+4] = 0; nu[6*a+5] = 0;
                }
                else
                {
                    nl[6*a] = std::numeric_limits<Real>::lowest(); nl[6*a+1] = std::numeric_limits<Real>::lowest(); nl[6*a+2] = std::numeric_limits<Real>::lowest();
                    nl[6*a+3] = std::numeric_limits<Real>::lowest(); nl[6*a+4] = std::numeric_limits<Real>::lowest(); nl[6*a+5] = std::numeric_limits<Real>::lowest();
                    nu[6*a] = std::numeric_limits<Real>::max(); nu[6*a+1] = std::numeric_limits<Real>::max(); nu[6*a+2] = std::numeric_limits<Real>::max();
                    nu[6*a+3] = std::numeric_limits<Real>::max(); nu[6*a+4] = std::numeric_limits<Real>::max(); nu[6*a+5] = std::numeric_limits<Real>::max();
                
                    free_actuator_indices.push_back(a);
                }
            }

            // alglib::minnlcsetbc(state, bndl, bndu);
            alglib::minnlcsetnlc2(state, nl, nu);

            // optimize

            // set up user info that will be passed as additional info to the optimization
            // _robot_copy.setState(_robot->state());
            typename PeristalticBendingRobot_Optimization<N>::UserInfo info;
            info.robot = robot;
            info.actuation_pressures = optimal_pressures;
            info.actuation_positions = current_actuator_positions;
            

            alglib::minnlcreport rep;
            alglib::real_1d_array x1;
            alglib::minnlcoptimize(state, PeristalticBendingRobot_Optimization<N>::ground_func, nullptr, &info);
            alglib::minnlcresults(state, x1, rep);
        }

        // write current robot state to file
        std::stringstream ss;
        ss << std::setw(4) << std::setfill('0') << step;
        std::string filename = output_folder + "step" + ss.str() + ".txt";
        std::ofstream file(filename);
        if (file.is_open())
        {
            for (int a = 0; a < robot->numActuators(); a++)
            {
                file << optimal_pressures[a].transpose() << " ";
            }
            file << "\n" << robot->state().state_vec;
        }
        file.close();
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
    initial_state.set_p(Vec3r(0,-0.5,rod_cs.rx()*1.1));
    initial_state.set_ori(Vec3r(-M_PI/2,0,0));
    robot.setState(initial_state);

    std::vector<std::vector<Real>> avg_pressures;
    int num_cycles = 500;
    for (int i = 0; i < num_cycles; i++)
    {
        std::vector<Real> avg_pressures1(num_actuators);
        std::vector<Real> avg_pressures2(num_actuators);
        std::vector<Real> avg_pressures3(num_actuators);
        std::vector<Real> avg_pressures4(num_actuators);
        std::vector<Real> avg_pressures5(num_actuators);
        std::vector<Real> avg_pressures6(num_actuators);
        for (int a = 0; a < num_actuators; a++)
        {
            avg_pressures1[a] = (a%2==0) ? 250e3 : 50e3;
            avg_pressures2[a] = (a%2==0) ? 250e3 : 75e3;
            avg_pressures3[a] = (a%2==0) ? 95e3 : 100e3;
            avg_pressures4[a] = (a%2==0) ? 60e3 : 100e3;
            avg_pressures5[a] = (a%2==0) ? 100e3 : 75e3;
            avg_pressures6[a] = (a%2==0) ? 200e3 : 50e3;
        }
        
        avg_pressures.push_back(avg_pressures1);
        avg_pressures.push_back(avg_pressures2);
        avg_pressures.push_back(avg_pressures3);
        avg_pressures.push_back(avg_pressures4);
        avg_pressures.push_back(avg_pressures5);
        avg_pressures.push_back(avg_pressures6);
    }

    runSimulation(&robot, avg_pressures);
}