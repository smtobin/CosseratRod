#ifndef __PERISTALTIC_BENDING_ROBOT_GROUND_SIMULATOR_IMPL_HPP
#define __PERISTALTIC_BENDING_ROBOT_GROUND_SIMULATOR_IMPL_HPP

#include "../alglib-cpp/src/optimization.h"

template<int N>
std::vector<typename PeristalticBendingRobot<N>::State> PeristalticBendingRobotGroundSimulator<N>::runSimulation()
{
    // create output vector
    int num_steps = _actuator_pressures[0].size();
    _states.resize(num_steps);

    int num_actuators = _robot->numActuators();

    // set up optimization
    int num_states = PeristalticBendingRobot<N>::State::NumStates;
    
    // scaling
    VecXr ones = VecXr::Ones(num_states);
    alglib::real_1d_array s;
    s.setcontent(num_states, ones.data());
    // optimizer parameters
    double epsx = 0.0000001;
    alglib::ae_int_t maxits = 0;
    alglib::minnlcstate state;

    // set bounds
    // alglib::real_1d_array bndl, bndu;
    // bndl.setlength(num_states);
    // bndu.setlength(num_states);
    // // default bounds are [-inf, +inf]
    // for (int i = 0; i < num_states; i++)
    // {
    //     bndl[i] = std::numeric_limits<Real>::lowest();
    //     bndu[i] = std::numeric_limits<Real>::max();
    // }

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
    // for (int i = num_actuators*3; i < num_actuators*3 + N; i++)
    // {
    //     nl[i] = 0;
    //     nu[i] = std::numeric_limits<Real>::max();
    // }

    // keeps track of actuator positions
    std::vector<Vec6r> current_actuator_positions(_robot->numActuators());

    for (int step = 0; step < num_steps; step++)
    try
    {
        std::cout << "\n=== Step " << step << " ===" << std::endl;
        alglib::real_1d_array x0;
        x0.setcontent(num_states, _robot->state().state_vec.data());
        // create optimizer object
        alglib::minnlccreate(num_states, x0, state);
        alglib::minnlcsetcond(state, epsx, maxits);
        alglib::minnlcsetscale(state, s);
        alglib::minnlcsetalgosqp(state);

        std::vector<Vec2r> current_actuator_pressures(_robot->numActuators());
        for (int a = 0; a < _robot->numActuators(); a++)
        {
            current_actuator_pressures[a] = _actuator_pressures[a][step];
            std::cout << "Actuator pressures " << a << ": " << current_actuator_pressures[a].transpose() << std::endl;
        }

        // if (step == 0)
        // {
        //     // run optimization without constraints to set the shape of the robot
        //     PeristalticBendingRobot_OptimizationFunctor functor(_robot, current_actuator_pressures);

        //     // Set up parameters
        //     LBFGSpp::LBFGSParam<Real> param;
        //     param.epsilon = 0;
        //     param.max_iterations = 10000;

        //     // Create solver object
        //     LBFGSpp::LBFGSSolver<Real> solver(param);

        //     VecXr orig_x = _robot->state().state_vec;
        //     Real fx;

        //     try 
        //     {
        //         // solve the optimization problem
        //         int niter = solver.minimize(functor, orig_x, fx);
        //     }
        //     catch(const std::runtime_error& e)
        //     {
        //         // if we don't converge, print out the error (maybe epsilon was set too small)
        //         std::cout << "Error occurred: " << e.what() << std::endl;
        //     }
        // }


        // calculate current actuator positions
        for (int a = 0; a < _robot->numActuators(); a++)
        {
            current_actuator_positions[a] = _robot->actuatorPositionAndOrientation(a);
            std::cout << "Actuator " << a << " position: " << current_actuator_positions[a].transpose() << std::endl;
        }

        // find the max pressure
        Real max_pressure = 0;
        for (int a = 0; a < _robot->numActuators(); a++)
        {
            if (_actuator_pressures[a][step].maxCoeff() > max_pressure)
                max_pressure = _actuator_pressures[a][step].maxCoeff();
        }
        // fix an actuator if it has the max pressure
        for (int a = 0; a < _robot->numActuators(); a++)
        {
            if (_actuator_pressures[a][step].maxCoeff() == max_pressure && max_pressure > 100e3)
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
            }
        }

        // alglib::minnlcsetbc(state, bndl, bndu);
        alglib::minnlcsetnlc2(state, nl, nu);

        // optimize

        // set up user info that will be passed as additional info to the optimization
        typename PeristalticBendingRobot_Optimization<N>::UserInfo info;
        info.robot = _robot;
        info.actuation_pressures = current_actuator_pressures;
        info.actuation_positions = current_actuator_positions;
        

        alglib::minnlcreport rep;
        alglib::real_1d_array x1;
        alglib::minnlcoptimize(state, PeristalticBendingRobot_Optimization<N>::ground_func, nullptr, &info);
        alglib::minnlcresults(state, x1, rep);

        // std::cout << "Final state:\n" << x1.tostring(5).c_str() << std::endl;

        _states[step] = _robot->state();
    }
    catch(alglib::ap_error alglib_exception)
    {
        std::cerr << alglib_exception.msg.c_str() << '\n';
        return _states;
    }

    return _states;
}

#endif // __PERISTALTIC_ROBOT_GROUND_SIMULATOR_IMPL_HPP