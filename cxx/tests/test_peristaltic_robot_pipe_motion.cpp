#include "PeristalticRobot.hpp"
#include "../LBFGSpp/include/LBFGS.h"   // TODO: change this path
#include "../alglib-cpp/src/optimization.h"
#include "PeristalticRobotPipeSimulator.hpp"

#define N 109

int main()
{
    EllipseCrossSection rod_cs(0.1, 0.1);
    EllipseCrossSection actuator_cs(0.08, 0.08);
    Real rod_length = 2.0;

    Real h = rod_length / (N-1);
    int num_segments_per_actuator = 4;
    Real actuator_length = num_segments_per_actuator*h;
    int num_actuators = (N-1) / (num_segments_per_actuator+2);

    std::cout << "Num Actuators: " << num_actuators << std::endl;

    Real E = 1e5;
    Real nu = 0.45;

    PeristalticRobot<N> robot(rod_length, rod_cs, E, nu, num_actuators, actuator_length, actuator_cs);

    // set series of actuation pressures for the actuators
    std::vector<std::vector<Real>> actuator_pressures(num_actuators);
    int num_cycles = 3;
    int num_steps_per_cycle = (2*num_actuators);
    int gap = 5;
    for (int a = 0; a < num_actuators; a++)
    {
        actuator_pressures[a].resize(num_cycles * num_steps_per_cycle, 0);
    }

    for (int a = 0; a < num_actuators; a++)
    {
        int ind = 2*(a%gap);
        while (ind+4 < num_cycles*num_steps_per_cycle)
        {
            actuator_pressures[a][ind++] = 70e3;
            actuator_pressures[a][ind++] = 160e3;
            actuator_pressures[a][ind++] = 220e3;
            actuator_pressures[a][ind++] = 220e3;
            actuator_pressures[a][ind++] = 70e3;

            ind += 2*(gap-2);
        }
    }

    // set simulation parameters
    Real pipe_radius = rod_cs.rx()*1.2;
    Real critical_radius_ratio = 1.25;

    PeristalticRobotPipeSimulator sim(&robot, actuator_pressures, pipe_radius, critical_radius_ratio);
    std::vector<PeristalticRobot<N>::State> results = sim.runSimulation();

    unsigned num_steps = num_cycles * num_steps_per_cycle;
    for (int i = 0; i < num_steps; i++)
    {
        std::cout << "State " << i << ":\n" << results[i] << std::endl;
    }

    // write to file
    sim.writeToFile("../output/sim/");
}