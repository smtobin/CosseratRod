#ifndef __PERISTALTIC_BENDING_ROBOT_PATH_FOLLOWING_SIMULATION_IMPL_HPP
#define __PERISTALTIC_BENDING_ROBOT_PATH_FOLLOWING_SIMULATION_IMPL_HPP

#include "../alglib-cpp/src/optimization.h"
#include "../LBFGSpp/include/LBFGS.h"  
#include "CosseratRod.hpp"

template<int N, int M>
Vec2r PeristalticBendingRobotPathFollowingSimulator<N,M>::_findPressuresForCurvature(Real low_pressure, Real desired_curvature)
{
    std::vector<Vec2r> actuation_pressures(1, Vec2r::Zero());

    for (Real high_pressure = low_pressure; high_pressure < low_pressure+200e3; high_pressure+=0.5e3)
    {
        if (desired_curvature < 0)
        {
            actuation_pressures[0][0] = low_pressure;
            actuation_pressures[0][1] = high_pressure;
        }
        else
        {
            actuation_pressures[0][0] = high_pressure;
            actuation_pressures[0][1] = low_pressure;
        }
            
        PeristalticBendingRobot_OptimizationFunctor functor(_single_actuator_robot.get(), actuation_pressures);

        // Set up parameters
        LBFGSpp::LBFGSParam<Real> param;
        param.epsilon = 0;
        param.max_iterations = 10000;

        // Create solver object
        LBFGSpp::LBFGSSolver<Real> solver(param);

        VecXr x = _single_actuator_robot->state().state_vec;
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
        

        Real eff_curvature;
        if (desired_curvature < 0)
        {
            eff_curvature = _single_actuator_robot->state().u2().minCoeff() / _single_actuator_robot->state().v3().maxCoeff();
        }
        else
        {
            eff_curvature = _single_actuator_robot->state().u2().maxCoeff() / _single_actuator_robot->state().v3().maxCoeff();
        }
        if (std::abs(eff_curvature) >= std::abs(desired_curvature))
        {
            return actuation_pressures[0];
        }
            
    }

    return Vec2r(low_pressure, low_pressure);
}

template<int N, int M>
Vec2r PeristalticBendingRobotPathFollowingSimulator<N,M>::_findPressuresForPath(Real low_pressure, int actuator_index, int node, bool from_center)
{
    std::vector<Vec2r> actuation_pressures(1, Vec2r::Zero());
    

    Real min_dist = std::numeric_limits<Real>::max();
    Vec2r min_pressures(low_pressure, low_pressure);
    for (Real high_pressure = low_pressure; high_pressure < low_pressure+100e3; high_pressure+=0.5e3)
    {
        // just assume we have positive curvature for now
        actuation_pressures[0][0] = high_pressure;
        actuation_pressures[0][1] = low_pressure;
            
        PeristalticBendingRobot_OptimizationFunctor functor(_single_actuator_robot.get(), actuation_pressures);

        // Set up parameters
        LBFGSpp::LBFGSParam<Real> param;
        param.epsilon = 0;
        param.max_iterations = 10000;

        // Create solver object
        LBFGSpp::LBFGSSolver<Real> solver(param);

        VecXr x = _single_actuator_robot->state().state_vec;
        Real fx;
        try 
        {
            // solve the optimization problem
            int niter = solver.minimize(functor, x, fx);
        }
        catch(const std::runtime_error& e)
        {
        }
    
        Vec3r pos;

        if (from_center)
        {
            Vec6r actuator_pos = _robot->actuatorPositionAndOrientation(actuator_index);
            std::vector<Vec3r> node_positions = _single_actuator_robot->nodePositions(
                actuator_pos.head<3>(), actuator_pos.tail<3>(),
                _single_actuator_robot->state().v1(), _single_actuator_robot->state().v2(), _single_actuator_robot->state().v3(),
                _single_actuator_robot->state().u1(), _single_actuator_robot->state().u2(), _single_actuator_robot->state().u3()
            );
            pos = node_positions[node];
        }
        else
        {
            Vec6r actuator_base = _robot->actuatorBasePositionAndOrientation(actuator_index);
            pos = CosseratRod<M>::nodePosition(
                _robot->length() / (N-1), node,
                actuator_base.head<3>(), Math::Exp_so3(actuator_base.tail<3>()),
                _single_actuator_robot->state().v1(), _single_actuator_robot->state().v2(), _single_actuator_robot->state().v3(),
                _single_actuator_robot->state().u1(), _single_actuator_robot->state().u2(), _single_actuator_robot->state().u3()
            );
        }

        Real dist = _distanceFromPath(pos);
        if (dist < min_dist)
        {
            min_dist = dist;
            min_pressures = actuation_pressures[0];
        }
            
    }
    // I'm lazy
    for (Real high_pressure = low_pressure; high_pressure < low_pressure+100e3; high_pressure+=0.5e3)
    {
        // just assume we have positive curvature for now
        actuation_pressures[0][1] = high_pressure;
        actuation_pressures[0][0] = low_pressure;
            
        PeristalticBendingRobot_OptimizationFunctor functor(_single_actuator_robot.get(), actuation_pressures);

        // Set up parameters
        LBFGSpp::LBFGSParam<Real> param;
        param.epsilon = 0;
        param.max_iterations = 10000;

        // Create solver object
        LBFGSpp::LBFGSSolver<Real> solver(param);

        VecXr x = _single_actuator_robot->state().state_vec;
        Real fx;
        try 
        {
            // solve the optimization problem
            int niter = solver.minimize(functor, x, fx);
        }
        catch(const std::runtime_error& e)
        {
        }
    
        Vec3r pos;

        if (from_center)
        {
            Vec6r actuator_pos = _robot->actuatorPositionAndOrientation(actuator_index);
            std::vector<Vec3r> node_positions = _single_actuator_robot->nodePositions(
                actuator_pos.head<3>(), actuator_pos.tail<3>(),
                _single_actuator_robot->state().v1(), _single_actuator_robot->state().v2(), _single_actuator_robot->state().v3(),
                _single_actuator_robot->state().u1(), _single_actuator_robot->state().u2(), _single_actuator_robot->state().u3()
            );
            pos = node_positions[node];
        }
        else
        {
            Vec6r actuator_base = _robot->actuatorBasePositionAndOrientation(actuator_index);
            pos = CosseratRod<M>::nodePosition(
                _robot->length() / (N-1), node,
                actuator_base.head<3>(), Math::Exp_so3(actuator_base.tail<3>()),
                _single_actuator_robot->state().v1(), _single_actuator_robot->state().v2(), _single_actuator_robot->state().v3(),
                _single_actuator_robot->state().u1(), _single_actuator_robot->state().u2(), _single_actuator_robot->state().u3()
            );
        }

        Real dist = _distanceFromPath(pos);
        if (dist < min_dist)
        {
            min_dist = dist;
            min_pressures = actuation_pressures[0];
        }
            
    }

    std::cout << "Actuator " << actuator_index << " min achieved distance: " << min_dist << std::endl;

    return min_pressures;
}

template<int N, int M>
std::vector<typename PeristalticBendingRobot<N>::State> PeristalticBendingRobotPathFollowingSimulator<N,M>::runSimulation()
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
    std::vector<Vec2r> current_actuator_pressures(_robot->numActuators());

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
            std::cout << "Actuator low pressure " << a << ": " << current_actuator_low_pressures[a] << std::endl;
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
            if (_actuator_pressures[a][step] == max_pressure && max_pressure >= 100e3)
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
            if (_actuator_pressures[a][step] == max_pressure && max_pressure >= 100e3)
            {
                Vec2r curvature_pressures = _findPressuresForCurvature(current_actuator_low_pressures[a], _pathCurvature(current_actuator_positions[a].head<3>()));
                if (a == 0)
                {
                    // find pressures such that the tip is on the path
                    Vec2r path_pressures = _findPressuresForPath(current_actuator_low_pressures[a], a, M-1, true);
                    current_actuator_pressures[a] = curvature_pressures*0.9 + path_pressures*0.1;
                }
                else
                {
                    Vec2r path_pressures = _findPressuresForPath(current_actuator_low_pressures[a], a, 0, true);
                    current_actuator_pressures[a] = curvature_pressures*0.9 + path_pressures*0.1;
                }
                
                
                // int actuator_node = _robot->actuatorNode(a);
                // Real last_curvature = _robot->state().u2()[actuator_node];
                // current_actuator_pressures[a] = _findPressuresForCurvature(current_actuator_low_pressures[a], last_curvature);
                current_actuator_pressures[a] = curvature_pressures;
            }
            // if the actuator does not have maximum pressure, then calculate the high pressure in order to stay on the path
            else
            {
                Vec2r curvature_pressures = _findPressuresForCurvature(current_actuator_low_pressures[a], _pathCurvature(current_actuator_positions[a].head<3>()));
                Vec2r path_pressures = _findPressuresForPath(current_actuator_low_pressures[a], a);
                // current_actuator_pressures[a] = curvature_pressures*0.7 + path_pressures*0.3;
                current_actuator_pressures[a] = curvature_pressures;
            }

            std::cout << "Actuator " << a << " pressures: " << current_actuator_pressures[a].transpose() << std::endl;
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