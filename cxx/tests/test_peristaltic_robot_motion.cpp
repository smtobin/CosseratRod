#include "PeristalticRobot.hpp"
#include "../LBFGSpp/include/LBFGS.h"   // TODO: change this path
#include "../alglib-cpp/src/optimization.h"
#include "PeristalticRobotSimulator.hpp"

#define N 25

int main()
{
    EllipseCrossSection rod_cs(0.25, 0.25);
    EllipseCrossSection actuator_cs(0.22, 0.22);
    Real rod_length = 2.0;
    Real actuator_length = 0.7;
    int num_actuators = 2;

    Real E = 1e5;
    Real nu = 0.45;

    PeristalticRobot<N> robot(rod_length, rod_cs, E, nu, num_actuators, actuator_length, actuator_cs);

    // set series of actuation pressures for the actuators
    std::vector<Real> actuator1_pressures;
    actuator1_pressures.push_back(0);
    std::vector<Real> actuator2_pressures;
    actuator2_pressures.push_back(0);
    int num_cycles = 10;
    for (int i = 0; i < num_cycles; i++)
    {

        if (i == 0)
        {
            // phase 1 (initial): actuator 1 expands to lock while actuator2 is unactuated
            actuator1_pressures.push_back(25e3); actuator2_pressures.push_back(0);
            actuator1_pressures.push_back(50e3); actuator2_pressures.push_back(0);
            actuator1_pressures.push_back(70e3); actuator2_pressures.push_back(0);

            // phase 2 (initial): actuator 1 pushes more
            actuator1_pressures.push_back(100e3); actuator2_pressures.push_back(0e3);
            actuator1_pressures.push_back(110e3); actuator2_pressures.push_back(0e3);
        }
        else
        {
            // phase 1: actuator 1 expands to lock
            actuator1_pressures.push_back(25e3); actuator2_pressures.push_back(70e3);
            actuator1_pressures.push_back(50e3); actuator2_pressures.push_back(70e3);
            actuator1_pressures.push_back(70e3); actuator2_pressures.push_back(70e3);

            // phase 2: actuator 2 contracts while actuator 1 pushes more
            actuator1_pressures.push_back(100e3); actuator2_pressures.push_back(50e3);
            actuator1_pressures.push_back(110e3); actuator2_pressures.push_back(25e3);
            actuator1_pressures.push_back(110e3); actuator2_pressures.push_back(0e3);
        }

        // phase 3: actuator 2 expands to lock
        actuator1_pressures.push_back(110e3); actuator2_pressures.push_back(25e3);
        actuator1_pressures.push_back(110e3); actuator2_pressures.push_back(50e3);
        actuator1_pressures.push_back(110e3); actuator2_pressures.push_back(70e3);

        // phase 4: actuator 1 contracts
        actuator1_pressures.push_back(100e3); actuator2_pressures.push_back(70e3);
        actuator1_pressures.push_back(75e3); actuator2_pressures.push_back(70e3);
        actuator1_pressures.push_back(50e3); actuator2_pressures.push_back(70e3);
        actuator1_pressures.push_back(25e3); actuator2_pressures.push_back(70e3);
        actuator1_pressures.push_back(0e3); actuator2_pressures.push_back(70e3);
    }

    std::vector<std::vector<Real>> actuator_pressures(num_actuators);
    actuator_pressures[0] = actuator1_pressures;
    actuator_pressures[1] = actuator2_pressures;

    // set simulation parameters
    Real pipe_radius = rod_cs.rx()*1.2;
    Real critical_radius_ratio = 1.25;

    PeristalticRobotSimulator sim(&robot, actuator_pressures, pipe_radius, critical_radius_ratio);
    std::vector<PeristalticRobot<N>::State> results = sim.runSimulation();

    unsigned num_steps = actuator1_pressures.size();
    for (int i = 0; i < num_steps; i++)
    {
        std::cout << "State " << i << ":\n" << results[i] << std::endl;
    }

    // write to file
    sim.writeToFile("../output/sim/");
}