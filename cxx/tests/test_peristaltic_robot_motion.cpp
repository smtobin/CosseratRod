#include "PeristalticRobot.hpp"
#include "../LBFGSpp/include/LBFGS.h"   // TODO: change this path
#include "../alglib-cpp/src/optimization.h"
#include "PeristalticRobotSimulator.hpp"

#define N 25

int main()
{
    EllipseCrossSection rod_cs(0.25, 0.25);
    EllipseCrossSection actuator_cs(0.15, 0.15);
    Real rod_length = 2.0;
    Real actuator_length = 0.7;
    int num_actuators = 2;

    Real E = 1e5;
    Real nu = 0.45;

    PeristalticRobot<N> robot(rod_length, rod_cs, E, nu, num_actuators, actuator_length, actuator_cs);

    // set series of actuation pressures for the actuators
    int num_steps = 6;
    std::vector<Real> actuator1_pressures(num_steps);
    actuator1_pressures[0] = 100e3;
    actuator1_pressures[1] = 200e3;
    actuator1_pressures[2] = 300e3;
    actuator1_pressures[3] = 400e3;
    actuator1_pressures[4] = 500e3;
    actuator1_pressures[5] = 600e3;
    std::vector<Real> actuator2_pressures(num_steps);
    actuator2_pressures[0] = 0;
    actuator2_pressures[1] = 0;
    actuator2_pressures[2] = 0;
    actuator2_pressures[3] = 0;
    actuator2_pressures[4] = 0;
    actuator2_pressures[5] = 0;

    std::vector<std::vector<Real>> actuator_pressures(num_actuators);
    actuator_pressures[0] = actuator1_pressures;
    actuator_pressures[1] = actuator2_pressures;

    // set simulation parameters
    Real radius_ratio = 1.2;
    Real pipe_radius = rod_cs.rx()*radius_ratio;
    Real critical_radius_ratio = 1.4;

    PeristalticRobotSimulator sim(&robot, actuator_pressures, pipe_radius, critical_radius_ratio);
    std::vector<PeristalticRobot<N>::State> results = sim.runSimulation();

    for (int i = 0; i < num_steps; i++)
    {
        std::cout << "State " << i << ":\n" << results[i] << std::endl;
    }
}