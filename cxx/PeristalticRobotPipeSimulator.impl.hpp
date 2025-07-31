#ifndef __PERISTALTIC_ROBOT_SIMULATOR_IMPL_HPP
#define __PERISTALTIC_ROBOT_SIMULATOR_IMPL_HPP

#include "../LBFGSpp/include/LBFGS.h"   // TODO: change this path
#include "../alglib-cpp/src/optimization.h"
#include "RodUtils.hpp"

template<int N>
void PeristalticRobotPipeSimulator<N>::_findCriticalPressures(Real pressure_increment)
{
    // Set up parameters
    LBFGSpp::LBFGSParam<Real> param;
    param.epsilon = 0;
    param.max_iterations = 10000;

    // Create solver object
    LBFGSpp::LBFGSSolver<Real> solver(param);

    typename PeristalticRobot<N>::State::StateVecType orig_state_vec = _robot->state().state_vec;
    for (int i = 0; i < _robot->numActuators(); i++)
    {
        for (int p = 0; p < 1000; p++)
        {
            std::vector<Real> actuation_pressures(_robot->numActuators(), 0);
            actuation_pressures[i] = p*pressure_increment;
            PeristalticRobot_OptimizationFunctor functor(_robot, actuation_pressures);

            // use LBFGSpp because it is cheap and easy and we don't need to use constraints
            // initial guess is rod's original state
            VecXr x = orig_state_vec;
            Real fx;

            try 
            {
                // solve the optimization problem
                int niter = solver.minimize(functor, x, fx);
            }
            catch(const std::runtime_error& e)
            {
                // if we don't converge, print out the error (maybe epsilon was set too small)
                // std::cout << "Error occurred: " << e.what() << std::endl;
            }
            
            // check the max value of a
            Real max_a = _robot->state().a().maxCoeff();

            if (max_a*_robot->crossSection()->rx() > _critical_radius_ratio*_pipe_radius)
            {
                _critical_pressures[i] = actuation_pressures[i];
                std::cout << "Critical pressure for actuator " << i << ": " << actuation_pressures[i] << " Pa" << std::endl;
                break;
            }
        }
        
    }

    // reset the robot state
    _robot->setState(orig_state_vec);
}

template<int N>
std::vector<typename PeristalticRobot<N>::State> PeristalticRobotPipeSimulator<N>::runSimulation()
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
    alglib::real_1d_array bndl, bndu;
    bndl.setlength(num_states);
    bndu.setlength(num_states);
    // default bounds are [-inf, +inf]
    for (int i = 0; i < num_states; i++)
    {
        bndl[i] = std::numeric_limits<Real>::lowest();
        bndu[i] = std::numeric_limits<Real>::max();
    }
    // set bounds for a and b to be less than (pipe radius)/(rod radius)
    Real a_max = _pipe_radius / _robot->crossSection()->rx();
    for (int i = PeristalticRobot<N>::State::aStart; i < PeristalticRobot<N>::State::aStart + PeristalticRobot<N>::State::NumNodes; i++)
    {
        bndu[i] = a_max;
    }
    for (int i = PeristalticRobot<N>::State::bStart; i < PeristalticRobot<N>::State::bStart + PeristalticRobot<N>::State::NumNodes; i++)
    {
        bndu[i] = a_max;
    }

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

        // fix an actuator if its pressure is greater than its pre-determined critical pressure
        // otherwise let it be free
        for (int a = 0; a < _robot->numActuators(); a++)
        {
            if (_actuator_pressures[a][step] >= _critical_pressures[a])
            {
                nl[3*a] = 0; nl[3*a+1] = 0; nl[3*a+2] = 0;
                nu[3*a] = 0; nu[3*a+1] = 0; nu[3*a+2] = 0;
            }
            else
            {
                nl[3*a] = std::numeric_limits<Real>::lowest(); nl[3*a+1] = std::numeric_limits<Real>::lowest(); nl[3*a+2] = std::numeric_limits<Real>::lowest();
                nu[3*a] = std::numeric_limits<Real>::max(); nu[3*a+1] = std::numeric_limits<Real>::max(); nu[3*a+2] = std::numeric_limits<Real>::max();
            }
        }

        alglib::minnlcsetbc(state, bndl, bndu);
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
        alglib::minnlcoptimize(state, PeristalticRobot_Optimization<N>::pipe_func, nullptr, &info);
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

#endif // __PERISTALTIC_ROBOT_SIMULATOR_IMPL_HPP