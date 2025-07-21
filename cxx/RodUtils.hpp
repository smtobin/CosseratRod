#ifndef __ROD_UTILS_HPP
#define __ROD_UTILS_HPP

#include "common.hpp"
#include "../LBFGSpp/include/LBFGS.h"   // TODO: change this path

#include <chrono>

struct RodUtils
{
    template <typename RodType>
    static void writeToFile(const std::string& filename, const RodType& rod)
    {
        std::ofstream file(filename);
        if (file.is_open())
        {
            file << RodType::NumNodes << "\n" << rod.length() << "\n" << rod.E() << "\n" << rod.nu() << "\n" <<
                rod.crossSection()->type() << "\n" << rod.crossSection()->rx() << "\n" << rod.crossSection()->ry() << "\n" <<
                rod.state().state_vec;
        }
    }

    template <typename RodType>
    static void solveOptimizationProblem(RodType& rod, const Vec3r& tip_force, int n_iter=1)
    {
        typename RodType::OptimizationFunctor functor(&rod, tip_force);

        // Set up parameters
        LBFGSpp::LBFGSParam<Real> param;
        param.epsilon = RodType::OptTol;
        param.max_iterations = 10000;

        // Create solver object
        LBFGSpp::LBFGSSolver<Real> solver(param);

        VecXr orig_x = rod.state().state_vec;

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

        std::cout << "Total energy: " << rod.minimizationEnergy(tip_force) << std::endl;

        Vec3r tip_pos = rod.tipPosition();
        std::cout << "Tip position: (" << tip_pos[0] << ", " << tip_pos[1] << ", " << tip_pos[2] << ")" << std::endl;
    }

    template <typename RodType>
    static void solveOptimizationProblemAndWriteToFile(RodType& rod, const Vec3r& tip_force, const std::string& folder_path)
    {
        solveOptimizationProblem(rod, tip_force);

        // form a descriptive filename
        std::string rod_type;
        if constexpr (std::is_same_v<RodType, CosseratRod<RodType::NumNodes>>)
            rod_type = "Rod";
        if constexpr (std::is_same_v<RodType, CosseratRodWithCrossSectionalDeformation<RodType::NumNodes>>)
            rod_type = "RodCSD";
        if constexpr (std::is_same_v<RodType, CosseratRodWithCrossSectionalDeformationLinearized<RodType::NumNodes>>)
            rod_type = "RodCSDLin";
        if constexpr (std::is_same_v<RodType, CosseratRodWithLinearModesOfCrossSectionalDeformation<RodType::NumNodes>>)
            rod_type = "RodCSDLinModes";
        
        std::string N_str = "N=" + std::to_string(RodType::NumNodes);
        std::stringstream force_ss;
        force_ss << "F=(" << tip_force[0] << "," << tip_force[1] << "," << tip_force[2] << ")";

        std::stringstream filename_ss;
        filename_ss << folder_path << rod_type << "_" << N_str << "_" << force_ss.str() << ".txt";
        // write to file
        writeToFile(filename_ss.str(), rod);
    }
};

#endif // __ROD_UTILS_HPP