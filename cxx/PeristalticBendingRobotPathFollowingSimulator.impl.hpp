#ifndef __PERISTALTIC_BENDING_ROBOT_PATH_FOLLOWING_SIMULATION_IMPL_HPP
#define __PERISTALTIC_BENDING_ROBOT_PATH_FOLLOWING_SIMULATION_IMPL_HPP

#include "../alglib-cpp/src/optimization.h"

Real PeristalticBendingRobotPathFollowingSimulator<N>::_findCorrespondingHighPressureForCurvature(
    Real low_pressure, Real desired_curvature)
{
    PeristalticBendingRobot<num_segments_per_actuator+1> robot(actuator_length, rod_cs, E, nu, 1, actuator_length, actuator_cs);
    std::vector<Vec2r> actuation_pressures(1, Vec2r::Zero());

    Real last_eff_curvature = 0;
    for (Real high_pressure = low_pressure; high_pressure < low_pressure+200e3; high_pressure+=0.5e3)
    {
        if (desired_max_curvature < 0)
        {
            actuation_pressures[0][0] = low_pressure;
            actuation_pressures[0][1] = high_pressure;
        }
        else
        {
            actuation_pressures[0][0] = high_pressure;
            actuation_pressures[0][1] = low_pressure;
        }
            

        int n_iter = 1;
        PeristalticBendingRobot_OptimizationFunctor functor(robot, actuation_pressures);

        // Set up parameters
        LBFGSpp::LBFGSParam<Real> param;
        param.epsilon = 0;
        param.max_iterations = 10000;

        // Create solver object
        LBFGSpp::LBFGSSolver<Real> solver(param);

        VecXr orig_x = robot->state().state_vec;

        auto t_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < n_iter; i++)
        {
            // initial guess is rod's original state
            VecXr x = orig_x;
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
        }

        Real eff_curvature;
        if (desired_max_curvature < 0)
        {
            eff_curvature = robot->state().u2().minCoeff() / robot->state().v3().maxCoeff();
        }
        else
        {
            eff_curvature = robot->state().u2().maxCoeff() / robot->state().v3().maxCoeff();
        }
        if (std::abs(eff_curvature) >= std::abs(desired_max_curvature))
        {
            return high_pressure;
        }
            
    }

    return low_pressure;
}

template<int N>
std::vector<typename PeristalticBendingRobot<N>::State> PeristalticBendingRobotPathFollowingSimulator<N>::runSimulation(int num_steps)
{
    // create output vector
    _states.resize(num_steps);

    int num_actuators = _robot->numActuators();

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

    // keeps track of current actuator positions and pressures
    std::vector<Vec6r> current_actuator_positions(_robot->numActuators());
    std::vector<Real> current_actuator_low_pressures(_robot->numActuators());

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

        
        for (int a = 0; a < _robot->numActuators(); a++)
        {
            current_actuator_low_pressures[a] = _actuator_pressures[a][step];
            std::cout << "Actuator low pressure " << a << ": " << current_actuator_low_pressures[a].transpose() << std::endl;
        }


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
            if (_actuator_pressures[a][step] > max_pressure)
                max_pressure = _actuator_pressures[a][step];
        }
        // fix an actuator if it has the max pressure
        for (int a = 0; a < _robot->numActuators(); a++)
        {
            if (_actuator_pressures[a][step] == max_pressure && max_pressure > 100e3)
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

        // find high pressures for path following
        for (int a = 0; a < _robot->numActuators(); a++)
        {
            // if the actuator has maximum pressure, then calculate the high pressure based on desired curvature
            // we take the desired curvature to be the curvature of the path at the actuator's current position (since it will be fixed)
            if (_actuator_pressures[a][step] == max_pressure && max_pressure > 100e3)
            {
                _findHighPressureFromCurvature(_actuator_pressures[a][step])
            }
            else
            {

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

#endif // __PERISTALTIC_BENDING_ROBOT_PATH_FOLLOWING_SIMULATION_IMPL_HPP