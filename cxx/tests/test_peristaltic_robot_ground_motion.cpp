#include "PeristalticRobot.hpp"
#include "../LBFGSpp/include/LBFGS.h"   // TODO: change this path
#include "../alglib-cpp/src/optimization.h"
#include "PeristalticRobotGroundSimulator.hpp"

#define N 49

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
    PeristalticRobot<N>::State initial_state = robot.state();
    initial_state.set_p(Vec3r(0,0,rod_cs.rx()));
    initial_state.set_ori(Vec3r(-M_PI/2,0,0));
    robot.setState(initial_state);

    // set series of actuation pressures for the actuators
    std::vector<std::vector<Real>> actuator_pressures(num_actuators);
    int num_cycles = 5;
    int num_steps_per_cycle = (2*num_actuators);
    int gap = 4;
    for (int a = 0; a < num_actuators; a++)
    {
        actuator_pressures[a].resize(num_cycles * num_steps_per_cycle, 0);
    }

    for (int a = 0; a < num_actuators; a++)
    {
        int ind = (a%gap);
        while (ind+4 < num_cycles*num_steps_per_cycle)
        {
            actuator_pressures[a][ind++] = 0e3;
            actuator_pressures[a][ind++] = 30e3;
            actuator_pressures[a][ind++] = 75e3;
            actuator_pressures[a][ind++] = 30e3;
            actuator_pressures[a][ind++] = 0e3;

            ind += (gap-5);
        }
    }

    // set simulation parameters
    Real pipe_radius = rod_cs.rx()*1.2;
    Real critical_radius_ratio = 1.25;

    PeristalticRobotGroundSimulator sim(&robot, actuator_pressures);
    std::vector<PeristalticRobot<N>::State> results = sim.runSimulation();

    unsigned num_steps = num_cycles * num_steps_per_cycle;
    for (int i = 0; i < num_steps; i++)
    {
        std::cout << "State " << i << ":\n" << results[i] << std::endl;
    }

    // write to file
    sim.writeToFile("../output/sim/");
}