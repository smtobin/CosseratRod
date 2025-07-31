#ifndef __PERISTALTIC_ROBOT_GROUND_SIMULATOR_IMPL_HPP
#define __PERISTALTIC_ROBOT_GROUND_SIMULATOR_IMPL_HPP

#include "../alglib-cpp/src/optimization.h"

template<int N>
std::vector<typename PeristalticRobot<N>::State> PeristalticRobotGroundSimulator<N>::runSimulation()
{
    // create output vector
    int num_steps = _actuator_pressures[0].size();
    _states.resize(num_steps);

    int num_actuators = _robot->numActuators();

    // set up optimization
    int num_states = PeristalticRobot<N>::State::NumStates;
    
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
    nl.setlength(num_actuators*3);
    nu.setlength(num_actuators*3);
    // default bounds are [-inf, +inf]
    for (int i = 0; i < num_actuators*3; i++)
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
    std::vector<Vec3r> current_actuator_positions(_robot->numActuators());

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

        // calculate current actuator positions
        for (int a = 0; a < _robot->numActuators(); a++)
        {
            current_actuator_positions[a] = _robot->actuatorPosition(a);
            std::cout << "Actuator " << a << " position: " << current_actuator_positions[a].transpose() << std::endl;
        }

        // find the max pressure
        Real max_pressure = 0;
        for (int a = 0; a < _robot->numActuators(); a++)
        {
            if (_actuator_pressures[a][step] > max_pressure)
                max_pressure = _actuator_pressures[a][step];
        }
        // fix an actuator if it has the max pressure
        for (int a = 0; a < _robot->numActuators(); a++)
        {
            if (_actuator_pressures[a][step] == max_pressure)
            {
                nl[3*a] = 0; nl[3*a+1] = 0; nl[3*a+2] = 0;
                nu[3*a] = 0; nu[3*a+1] = 0; //nu[3*a+2] = 0;
            }
            else
            {
                nl[3*a] = std::numeric_limits<Real>::lowest(); nl[3*a+1] = std::numeric_limits<Real>::lowest(); nl[3*a+2] = std::numeric_limits<Real>::lowest();
                nu[3*a] = std::numeric_limits<Real>::max(); nu[3*a+1] = std::numeric_limits<Real>::max(); nu[3*a+2] = std::numeric_limits<Real>::max();
            }
        }

        // alglib::minnlcsetbc(state, bndl, bndu);
        alglib::minnlcsetnlc2(state, nl, nu);

        // optimize
        std::vector<Real> current_actuator_pressures(_robot->numActuators());
        for (int a = 0; a < _robot->numActuators(); a++)
        {
            current_actuator_pressures[a] = _actuator_pressures[a][step];
            std::cout << "Actuator pressure " << a << ": " << current_actuator_pressures[a] << std::endl;
        }

        // set up user info that will be passed as additional info to the optimization
        typename PeristalticRobot_Optimization<N>::UserInfo info;
        info.robot = _robot;
        info.actuation_pressures = current_actuator_pressures;
        info.actuation_positions = current_actuator_positions;
        

        alglib::minnlcreport rep;
        alglib::real_1d_array x1;
        // auto t_start = std::chrono::high_resolution_clock::now();
        alglib::minnlcoptimize(state, PeristalticRobot_Optimization<N>::ground_func, nullptr, &info);
        // t_end = std::chrono::high_resolution_clock::now();
        // auto time_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count() / 1.0e6;
        // std::cout << "Elapsed time for optimization: " << time_ms << " ms" << std::endl;
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