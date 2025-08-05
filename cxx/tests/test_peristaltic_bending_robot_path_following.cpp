#include "PeristalticBendingRobot.hpp"
#include "PeristalticBendingRobotPathFollowingSimulator.hpp"
#include "../LBFGSpp/include/LBFGS.h"   // TODO: change this path
#include "../alglib-cpp/src/optimization.h"
#include "RodUtils.hpp"
#include "CosseratRodBase.hpp"

#include <chrono>

#define N 13

EllipseCrossSection rod_cs(0.1, 0.1);
EllipseCrossSection actuator_cs(0.035, 0.08);
Real rod_length = 1.0;

Real h = rod_length / (N-1);
constexpr int num_segments_per_actuator = 4;
Real actuator_length = num_segments_per_actuator*h;
int num_actuators = (N-1) / (num_segments_per_actuator+2);

Real E = 1e5;
Real nu = 0.45;

int main()
{

    PeristalticBendingRobot<N> robot(rod_length, rod_cs, E, nu, num_actuators, actuator_length, actuator_cs);
    PeristalticBendingRobot<N>::State initial_state = robot.state();
    initial_state.set_p(Vec3r(0,0,rod_cs.rx()*1.1));
    initial_state.set_ori(Vec3r(-M_PI/2,0,0));
    robot.setState(initial_state);

    std::vector<std::vector<Real>> actuator_low_pressures(num_actuators);
    actuator_low_pressures[0].push_back(0e3);      actuator_low_pressures[1].push_back(0e3);
    int num_cycles = 100;
    for (int ci = 0; ci < num_cycles; ci++)
    {
        actuator_low_pressures[0].push_back(50e3);      actuator_low_pressures[1].push_back(100e3); 
        // actuator_low_pressures[0].push_back(150e3);     actuator_low_pressures[1].push_back(50e3);
        actuator_low_pressures[0].push_back(150e3);     actuator_low_pressures[1].push_back(50e3);
        actuator_low_pressures[0].push_back(150e3);     actuator_low_pressures[1].push_back(100e3);
        // actuator_low_pressures[0].push_back(50e3);      actuator_low_pressures[1].push_back(100e3);
    }

    PeristalticBendingRobotPathFollowingSimulator<N,num_segments_per_actuator+1> sim(&robot, actuator_low_pressures);
    sim.runSimulation();

    sim.writeToFile("../output/sim/");
}