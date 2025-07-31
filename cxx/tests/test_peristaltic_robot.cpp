#include "PeristalticRobot.hpp"
#include "../LBFGSpp/include/LBFGS.h"   // TODO: change this path
#include "../alglib-cpp/src/optimization.h"
#include "RodUtils.hpp"

#include <chrono>

#define N 109

int main()
{
    EllipseCrossSection rod_cs(0.1, 0.1);
    EllipseCrossSection actuator_cs(0.08, 0.08);
    Real rod_length = 2.0;

    Real h = rod_length / (N-1);
    int num_segments_per_actuator = 2;
    Real actuator_length = num_segments_per_actuator*h;
    int num_actuators = (N-1) / (num_segments_per_actuator+2);

    std::cout << "Num Actuators: " << num_actuators << std::endl;

    Real E = 1e5;
    Real nu = 0.45;

    PeristalticRobot<N> robot(rod_length, rod_cs, E, nu, num_actuators, actuator_length, actuator_cs);

    std::vector<Real> actuation_pressures(num_actuators, 0);
    for (int i = 0; i < num_actuators; i++)
    {
        if (i % 2 == 0)
            actuation_pressures[i] = 100e3;

    }
    
    // Real energy = robot.minimizationEnergy(actuation_pressures);

    ///////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////
    std::cout << "\n=== LBFGSpp Optimization ===\n" << std::endl;

    int n_iter = 1;
    PeristalticRobot_OptimizationFunctor functor(&robot, actuation_pressures);

    // Set up parameters
    LBFGSpp::LBFGSParam<Real> param;
    param.epsilon = 0;
    param.max_iterations = 10000;

    // Create solver object
    LBFGSpp::LBFGSSolver<Real> solver(param);

    VecXr orig_x = robot.state().state_vec;

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
            std::cout << "Number of iterations: " << niter << std::endl;
        }
        catch(const std::runtime_error& e)
        {
            // if we don't converge, print out the error (maybe epsilon was set too small)
            std::cout << "Error occurred: " << e.what() << std::endl;
        }
    }

    // print out info
    auto t_end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count() / 1.0e6;
    std::cout << "Elapsed time for optimization: " << time_ms / n_iter << " ms" << std::endl;

    std::cout << "Total energy: " << robot.minimizationEnergy(actuation_pressures) << std::endl;

    std::cout << "Final state:\n" << robot.state() << std::endl;

    RodUtils::writeToFile("../output/peristaltic.txt", robot);

    // test out actuator positions
    Vec3r act0_pos = robot.actuatorPosition(0);
    Vec3r act1_pos = robot.actuatorPosition(1);

    ///////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////

    std::cout << "\n=== Alglib Optimization ====\n" << std::endl;

    try
    {
        int num_states = PeristalticRobot<N>::State::NumStates;
        alglib::real_1d_array x0;
        x0.setcontent(num_states, orig_x.data());
        
        VecXr ones = VecXr::Ones(num_states);
        alglib::real_1d_array s;
        s.setcontent(num_states, ones.data());

        double epsx = 0.000000001;
        alglib::ae_int_t maxits = 0;
        alglib::minnlcstate state;

        // create optimizer object
        alglib::minnlccreate(num_states, x0, state);
        alglib::minnlcsetcond(state, epsx, maxits);
        alglib::minnlcsetscale(state, s);
        alglib::minnlcsetalgosqp(state);

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
        for (int i = PeristalticRobot<N>::State::aStart; i < PeristalticRobot<N>::State::aStart + PeristalticRobot<N>::State::NumNodes; i++)
        {
            bndu[i] = 1.2;
        }
        for (int i = PeristalticRobot<N>::State::bStart; i < PeristalticRobot<N>::State::bStart + PeristalticRobot<N>::State::NumNodes; i++)
        {
            bndu[i] = 1.2;
        }
        alglib::minnlcsetbc(state, bndl, bndu);

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

        // fix the first actuator
        // nl[0] = 0; nl[1] = 0; nl[2] = 0;
        // nu[0] = 0; nu[1] = 0; nu[2] = 0;

        // fix the second actuator
        // nl[3] = 0; nl[4] = 0; nl[5] = 0;
        // nu[3] = 0; nu[4] = 0; nu[5] = 0;

        alglib::minnlcsetnlc2(state, nl, nu);

        // optimize
        std::vector<Vec3r> actuator_positions(num_actuators);
        actuator_positions[0] = Vec3r(0,0,0.141973);
        actuator_positions[1] = Vec3r(0,0,1.52716);
        PeristalticRobot_Optimization<N>::UserInfo info;
        info.robot = &robot;
        info.actuation_pressures = actuation_pressures;
        info.actuation_positions = actuator_positions;
        

        alglib::minnlcreport rep;
        alglib::real_1d_array x1;
        t_start = std::chrono::high_resolution_clock::now();
        alglib::minnlcoptimize(state, PeristalticRobot_Optimization<N>::pipe_func, nullptr, &info);
        t_end = std::chrono::high_resolution_clock::now();
        time_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count() / 1.0e6;
        std::cout << "Elapsed time for optimization: " << time_ms << " ms" << std::endl;
        alglib::minnlcresults(state, x1, rep);

        std::cout << "Final state:\n" << x1.tostring(5).c_str() << std::endl;

        std::cout << "Robot state:\n" << robot.state() << std::endl;

        std::cout << "Minimization energy: " << robot.minimizationEnergy(actuation_pressures) << std::endl;
    }
    catch(alglib::ap_error alglib_exception)
    {
        std::cerr << alglib_exception.msg.c_str() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}