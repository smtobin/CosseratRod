#pragma once

#include "CosseratRod.hpp"


template <int N>
std::vector<Vec2r> PeristalticBendingRobotPathOptimizer<N>::findOptimalPressures(
    const std::vector<Real>& avg_pressures,
    int fixed_actuator_index, const Vec3r& fixed_actuator_position, const Mat3r& fixed_actuator_rotation
)
{
    PeristalticBendingRobotPathOptimizer_OptimizationFunctor functor(
        this, avg_pressures,
        fixed_actuator_index, fixed_actuator_position, fixed_actuator_rotation
    );

    _robot_copy.setState(_robot->state());

    // Set up parameters
    LBFGSpp::LBFGSParam<Real> param;
    if (avg_pressures.size() == 2)
        param.epsilon = 0;
    else
        param.epsilon = 0.001;
    param.max_iterations = 10000;

    // Create solver object
    LBFGSpp::LBFGSSolver<Real> solver(param);

    VecXr x = VecXr::Zero(avg_pressures.size());
    Real fx;

    try 
    {
        // solve the optimization problem
        solver.minimize(functor, x, fx);
    }
    catch(const std::runtime_error& e)
    {
        std::cout << "Error in findOptimalPressures: " << e.what() << std::endl;
    }

    // std::cout << "Optimal pressure differentials: " << x.transpose() << std::endl;

    // output the optimal actuation pressures, not just the differentials
    // can easily get these from the average pressures that were passed in
    std::vector<Vec2r> actuation_pressures(avg_pressures.size());
    for(unsigned i = 0; i < actuation_pressures.size(); i++)
    {
        actuation_pressures[i] = Vec2r(avg_pressures[i] + 0.5*x[i], avg_pressures[i] - 0.5*x[i]);
    }

    return actuation_pressures;
}

template <int N>
Real PeristalticBendingRobotPathOptimizer<N>::minimizationCost(
    const std::vector<Vec2r>& actuation_pressures,
    int fixed_actuator_index, const Vec3r& fixed_actuator_position, const Mat3r& fixed_actuator_rotation)
{
    if (actuation_pressures.size() == 2)
    {
        PeristalticBendingRobot_OptimizationFunctor functor(&_robot_copy, actuation_pressures);

        {
            // Set up parameters
            LBFGSpp::LBFGSParam<Real> param;
            param.epsilon = 0;
            param.max_iterations = 10000;

            // Create solver object
            LBFGSpp::LBFGSSolver<Real> solver(param);

            VecXr x;
            x = _robot_copy.state().state_vec;
            VecXr x2(x);
            Real fx;

            try 
            {
                // solve the optimization problem
                solver.minimize(functor, x2, fx);
            }
            catch(const std::runtime_error& e)
            {
            }
        }

        // evaluate the cost
        int free_actuator_index = (fixed_actuator_index == 0) ? 1 : 0;

        Vec6r free_actuator_pos = CosseratRod<N>::nodePositionAndOrientationGivenStartingNode(
            _robot_copy.length() / (N-1),
            _robot_copy.actuatorNode(fixed_actuator_index), fixed_actuator_position, fixed_actuator_rotation,
            _robot_copy.actuatorNode(free_actuator_index),
            _robot_copy.state().v1(), _robot_copy.state().v2(), _robot_copy.state().v3(), 
            _robot_copy.state().u1(), _robot_copy.state().u2(), _robot_copy.state().u3()
        );

        Mat3r free_orientation = Math::Exp_so3(free_actuator_pos.tail<3>());
        Vec2r robot_tangent = free_orientation.col(2).head<2>();
        Vec2r robot_position = free_actuator_pos.head<2>();

        Vec2r path_tangent, path_position;
        if (free_actuator_index == 0)
        {
            auto vec1 = _tangentAtClosestPointOnPath(robot_position, _back_actuator_path);
            auto vec2 = _closestPointOnPath(robot_position, _back_actuator_path);
            // _back_actuator_path = num;
            path_tangent = vec1;
            path_position = vec2;
        }
        else
        {
            auto vec1 = _tangentAtClosestPointOnPath(robot_position, _front_actuator_path);
            auto vec2 = _closestPointOnPath(robot_position, _front_actuator_path);
            // _front_actuator_path = num;
            path_tangent = vec1;
            path_position = vec2;
        }
        

        Mat4r W = 1e4*Vec4r(3, 3, 10, 10).asDiagonal();
        Vec4r vec;
        vec.head<2>() = path_tangent - robot_tangent;
        vec.tail<2>() = path_position - robot_position;
        Real cost = vec.transpose() * W * vec;
        return cost;
    }
    else
    {
        // try to make it general
        int num_actuators = _robot->numActuators();

        // set up optimization
        int num_states = PeristalticBendingRobot<N>::State::NumStates;
        
        // scaling
        VecXr ones = VecXr::Ones(num_states);
        alglib::real_1d_array s;
        s.setcontent(num_states, ones.data());
        // optimizer parameters
        double epsx = 0.0001;
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
        x0.setcontent(num_states, _robot->state().state_vec.data());
        // create optimizer object
        alglib::minnlccreate(num_states, x0, state);
        alglib::minnlcsetcond(state, epsx, maxits);
        alglib::minnlcsetscale(state, s);
        alglib::minnlcsetalgosqp(state);


        // calculate current actuator positions
        std::vector<Vec6r> current_actuator_positions(_robot->numActuators());
        for (int a = 0; a < _robot->numActuators(); a++)
        {
            current_actuator_positions[a] = _robot->actuatorPositionAndOrientation(a);
            // std::cout << "Actuator " << a << " position: " << current_actuator_positions[a].transpose() << std::endl;
        }

        // find the max pressure
        Real max_pressure = 0;
        for (int a = 0; a < _robot->numActuators(); a++)
        {
            if (actuation_pressures[a].mean() > max_pressure)
                max_pressure = actuation_pressures[a].mean();
        }
        // fix an actuator if it has the max pressure
        // keep track of actuators that are "free" i.e. allowed to move
        std::vector<int> free_actuator_indices;
        for (int a = 0; a < _robot->numActuators(); a++)
        {
            if (actuation_pressures[a].mean() == max_pressure && max_pressure > 100e3)
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
        _robot_copy.setState(_robot->state());
        typename PeristalticBendingRobot_Optimization<N>::UserInfo info;
        info.robot = &_robot_copy;
        info.actuation_pressures = actuation_pressures;
        info.actuation_positions = current_actuator_positions;
        

        alglib::minnlcreport rep;
        alglib::real_1d_array x1;
        alglib::minnlcoptimize(state, PeristalticBendingRobot_Optimization<N>::ground_func, nullptr, &info);
        alglib::minnlcresults(state, x1, rep);

        // evaluate the cost

        VecXr robot_tangents(free_actuator_indices.size()*2);
        VecXr robot_positions(free_actuator_indices.size()*2);
        VecXr path_tangents(free_actuator_indices.size()*2);
        VecXr path_positions(free_actuator_indices.size()*2);
        VecXr weights = 1e6*VecXr::Ones(free_actuator_indices.size()*4);
        weights(Eigen::seqN(free_actuator_indices.size()*2, free_actuator_indices.size()*2)) *= 100; // prioritize position

        for (unsigned i = 0; i < free_actuator_indices.size(); i++)
        {
            Vec6r free_actuator_pos = _robot_copy.actuatorPositionAndOrientation(free_actuator_indices[i]);

            Mat3r free_orientation = Math::Exp_so3(free_actuator_pos.tail<3>());
            robot_tangents(Eigen::seqN(2*i, 2)) = free_orientation.col(2).head<2>();
            Vec2r robot_position = free_actuator_pos.head<2>();
            robot_positions(Eigen::seqN(2*i, 2)) = robot_position;

            // path_tangents(Eigen::seqN(2*i, 2)) = _tangentAtClosestPointOnPath(robot_position, _front_actuator_path);
            // path_positions(Eigen::seqN(2*i, 2)) = _closestPointOnPath(robot_position, _front_actuator_path);
        }

        VecXr vec(free_actuator_indices.size()*4);
        vec(Eigen::seqN(0, free_actuator_indices.size()*2)) = path_tangents - robot_tangents;
        vec(Eigen::seqN(free_actuator_indices.size()*2, free_actuator_indices.size()*2)) = path_positions - robot_positions;
        Real cost = vec.transpose() * weights.asDiagonal() * vec;

        std::cout << "Cost: " << cost << std::endl;
        return cost;
    }

    
}

template <int N>
VecXr PeristalticBendingRobotPathOptimizer<N>::minimizationGradient(
    const std::vector<Vec2r>& actuation_pressures,
    int fixed_actuator_index, const Vec3r& fixed_actuator_position, const Mat3r& fixed_actuator_rotation)
{
    VecXr grad(actuation_pressures.size());
    std::vector<Vec2r> new_actuation_pressures(actuation_pressures);
    Real pressure_delta = 1.0; // Pa
    Real orig_cost = minimizationCost(new_actuation_pressures, fixed_actuator_index, fixed_actuator_position, fixed_actuator_rotation);
    for (unsigned i = 0; i < actuation_pressures.size(); i++)
    {
        new_actuation_pressures[i][0] += pressure_delta/2;
        new_actuation_pressures[i][1] -= pressure_delta/2;
        Real new_cost = minimizationCost(new_actuation_pressures, fixed_actuator_index, fixed_actuator_position, fixed_actuator_rotation);
        
        // std::cout << "cost diff: " << new_cost - orig_cost << std::endl;
        grad[i] = (new_cost - orig_cost) / pressure_delta;
        new_actuation_pressures[i][0] -= pressure_delta/2;
        new_actuation_pressures[i][1] += pressure_delta/2;
    }

    return grad;
}