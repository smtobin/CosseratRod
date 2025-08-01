#include "PeristalticBendingRobot.hpp"
#include "../LBFGSpp/include/LBFGS.h"   // TODO: change this path
#include "../alglib-cpp/src/optimization.h"
#include "PeristalticBendingRobotGroundSimulator.hpp"

#define N 15

int main()
{
    EllipseCrossSection rod_cs(0.1, 0.1);
    EllipseCrossSection actuator_cs(0.035, 0.08);
    Real rod_length = 1.0;

    Real h = rod_length / (N-1);
    int num_segments_per_actuator = 4;
    Real actuator_length = num_segments_per_actuator*h;
    int num_actuators = (N-1) / (num_segments_per_actuator+2);

    std::cout << "Num Actuators: " << num_actuators << std::endl;

    Real E = 1e5;
    Real nu = 0.45;

    PeristalticBendingRobot<N> robot(rod_length, rod_cs, E, nu, num_actuators, actuator_length, actuator_cs);
    PeristalticBendingRobot<N>::State initial_state = robot.state();
    initial_state.set_p(Vec3r(0,0,rod_cs.rx()*1.1));
    initial_state.set_ori(Vec3r(-M_PI/2,0,0));
    robot.setState(initial_state);

    // set series of actuation pressures for the actuators
    std::vector<std::vector<Vec2r>> actuator_pressures(num_actuators);
    

    // assuming 2 actuators
    // actuator 1 - always straight
    int num_cycles = 20;
    int ns = 4;
    for (int i = 0; i < num_cycles; i++)
    {
        // if (i%2 == 0)
        // {
            actuator_pressures[0].push_back(Vec2r(75e3, 100e3));         actuator_pressures[1].push_back(Vec2r(50e3, 50e3));
            actuator_pressures[0].push_back(Vec2r(125e3, 155e3));       actuator_pressures[1].push_back(Vec2r(50e3, 50e3));
            actuator_pressures[0].push_back(Vec2r(45e3, 65e3));       actuator_pressures[1].push_back(Vec2r(100e3, 100e3));
            
        // }
        // else
        // {
        //     actuator_pressures[0].push_back(Vec2r(100e3, 100e3));       actuator_pressures[1].push_back(Vec2r(50e3, 50e3));
        //     actuator_pressures[0].push_back(Vec2r(160e3, 160e3));       actuator_pressures[1].push_back(Vec2r(50e3, 50e3));
        //     actuator_pressures[0].push_back(Vec2r(60e3, 60e3));       actuator_pressures[1].push_back(Vec2r(100e3, 100e3));
        // }
        // actuator_pressures[0].push_back(Vec2r(120e3, 120e3));       actuator_pressures[1].push_back(Vec2r(45e3, 75e3));
    }


    // int num_cycles = 5;
    // int num_steps_per_cycle = (2*num_actuators);
    // int gap = 6;
    // for (int a = 0; a < num_actuators; a++)
    // {
    //     actuator_pressures[a].resize(num_cycles * num_steps_per_cycle, Vec2r(0e3, 20e3));
    // }

    // for (int a = 0; a < num_actuators; a++)
    // {
    //     int ind = (a%gap);
    //     while (ind+4 < num_cycles*num_steps_per_cycle)
    //     {
    //         actuator_pressures[a][ind++] = Vec2r(0e3, 20e3);
    //         actuator_pressures[a][ind++] = Vec2r(50e3, 80e3);
    //         actuator_pressures[a][ind++] = Vec2r(120e3, 160e3);
    //         actuator_pressures[a][ind++] = Vec2r(50e3, 80e3);
    //         actuator_pressures[a][ind++] = Vec2r(0e3, 20e3);
    //         // actuator_pressures[a][ind++] = Vec2r(0e3, 0e3);
    //         // actuator_pressures[a][ind++] = Vec2r(60e3, 60e3);
    //         // actuator_pressures[a][ind++] = Vec2r(150e3, 150e3);
    //         // actuator_pressures[a][ind++] = Vec2r(60e3, 60e3);
    //         // actuator_pressures[a][ind++] = Vec2r(0e3, 0e3);

    //         ind += (gap-5);
    //     }
    //     // if (a <= num_actuators/2)
    //     // {
    //     //     for (int i = 0; i < num_cycles*num_steps_per_cycle; i++)
    //     //     {
    //     //         actuator_pressures[a][i] = Vec2r(0e3, 20e3);
    //     //     }
    //     //     while (ind+4 < num_cycles*num_steps_per_cycle)
    //     //     {
    //     //         actuator_pressures[a][ind++] = Vec2r(0e3, 20e3);
    //     //         actuator_pressures[a][ind++] = Vec2r(60e3, 80e3);
    //     //         actuator_pressures[a][ind++] = Vec2r(120e3, 140e3);
    //     //         actuator_pressures[a][ind++] = Vec2r(60e3, 80e3);
    //     //         actuator_pressures[a][ind++] = Vec2r(0e3, 20e3);
    //     //         // actuator_pressures[a][ind++] = Vec2r(0e3, 0e3);
    //     //         // actuator_pressures[a][ind++] = Vec2r(60e3, 60e3);
    //     //         // actuator_pressures[a][ind++] = Vec2r(150e3, 150e3);
    //     //         // actuator_pressures[a][ind++] = Vec2r(60e3, 60e3);
    //     //         // actuator_pressures[a][ind++] = Vec2r(0e3, 0e3);

    //     //         ind += (gap-5);
    //     //     }
    //     // }
    //     // else
    //     // {
    //     //     // for (int i = 0; i < num_cycles*num_steps_per_cycle; i++)
    //     //     // {
    //     //     //     actuator_pressures[a][i] = Vec2r(120e3, 120e3);
    //     //     // }
    //     //     while (ind+4 < num_cycles*num_steps_per_cycle)
    //     //     {
    //     //         actuator_pressures[a][ind++] = Vec2r(0e3, 0e3);
    //     //         actuator_pressures[a][ind++] = Vec2r(80e3, 80e3);
    //     //         actuator_pressures[a][ind++] = Vec2r(140e3, 140e3);
    //     //         actuator_pressures[a][ind++] = Vec2r(80e3, 80e3);
    //     //         actuator_pressures[a][ind++] = Vec2r(0e3, 0e3);

    //     //         ind += (gap-5);
    //     //     }
    //     // }
    // }

    // set simulation parameters
    PeristalticBendingRobotGroundSimulator sim(&robot, actuator_pressures);
    std::vector<PeristalticBendingRobot<N>::State> results = sim.runSimulation();

    unsigned num_steps = actuator_pressures[0].size();
    for (int i = 0; i < num_steps; i++)
    {
        std::cout << "State " << i << ":\n" << results[i] << std::endl;
    }

    // write to file
    sim.writeToFile("../output/sim/");
}