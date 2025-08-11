#include "PeristalticBendingRobot.hpp"
#include "../LBFGSpp/include/LBFGS.h"   // TODO: change this path
#include "../alglib-cpp/src/optimization.h"
#include "PeristalticBendingRobotGroundSimulator.hpp"

#define N 49

int main()
{
    EllipseCrossSection rod_cs(0.1, 0.1);
    EllipseCrossSection actuator_cs(0.035, 0.08);
    Real rod_length = 2.0;

    Real h = rod_length / (N-1);
    int num_segments_per_actuator = 6;
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
    // int num_cycles = 20;
    // int ns = 4;
    // for (int i = 0; i < num_cycles; i++)
    // {
    //     // if (i%2 == 0)
    //     // {
    //         // actuator_pressures[0].push_back(Vec2r(75e3, 100e3));         actuator_pressures[1].push_back(Vec2r(50e3, 50e3));
    //         // actuator_pressures[0].push_back(Vec2r(125e3, 155e3));       actuator_pressures[1].push_back(Vec2r(50e3, 50e3));
    //         // actuator_pressures[0].push_back(Vec2r(45e3, 65e3));       actuator_pressures[1].push_back(Vec2r(100e3, 100e3));
            
    //     // }
    //     // else
    //     // {
    //     //     actuator_pressures[0].push_back(Vec2r(100e3, 100e3));       actuator_pressures[1].push_back(Vec2r(50e3, 50e3));
    //     //     actuator_pressures[0].push_back(Vec2r(160e3, 160e3));       actuator_pressures[1].push_back(Vec2r(50e3, 50e3));
    //     //     actuator_pressures[0].push_back(Vec2r(60e3, 60e3));       actuator_pressures[1].push_back(Vec2r(100e3, 100e3));
    //     // }
    //     // actuator_pressures[0].push_back(Vec2r(120e3, 120e3));       actuator_pressures[1].push_back(Vec2r(45e3, 75e3));
    // }


    int num_cycles = 40;
    int num_steps_per_cycle = (2*num_actuators);
    int gap = 6;
    for (int a = 0; a < num_actuators; a++)
    {
        actuator_pressures[a].resize(num_cycles * num_steps_per_cycle, Vec2r(0e3, 0));
    }

    // differentials for a constant curvature of 0.5
    // std::vector<std::vector<Real>> actuator_differentials1 = {
    //     {11.52e3, 11.52e3, 11.52e3, 11.52e3, 11.52e3, 11.52e3},  // differentials for low pressure = 0e3
    //     {12.32e3, 12.32e3, 12.32e3, 12.32e3, 12.32e3, 12.32e3},  // differentials for low pressure = 50e3
    //     {13.39e3, 13.39e3, 13.39e3, 13.39e3, 13.39e3, 13.39e3}   // differentials for low pressure = 100e3
    // };
    
    // std::vector<std::vector<Real>> actuator_differentials1 = {
    //     {0e3, 0e3, 60e3, 60e3, 0e3, 0e3}, // differentials for low pressure = 0e3
    //     {0e3, 0e3, 62e3, 62e3, 0e3, 0e3}, // differentials for low pressure = 50e3
    //     {0e3, 0e3, 64e3, 64e3, 0e3, 0e3}  // differentials for low pressure = 100e3
    // };

    // differentials for a constant curvature of 1.5
    // std::vector<std::vector<Real>> actuator_differentials2 = {
    //     {34.24e3, 34.24e3, 34.24e3, 34.24e3, 34.24e3, 34.24e3},  // differentials for low pressure = 0e3
    //     {36.95e3, 36.95e3, 36.95e3, 36.95e3, 36.95e3, 36.95e3},  // differentials for low pressure = 50e3
    //     {40.51e3, 40.51e3, 40.51e3, 40.51e3, 40.51e3, 40.51e3}   // differentials for low pressure = 100e3
    // };

    std::vector<std::vector<Real>> actuator_differentials1 = {
        {0e3, 0e3, 0e3, 0e3, 0e3, 0e3}, // differentials for low pressure = 0e3
        {0e3, 0e3, 0e3, 0e3, 0e3, 0e3}, // differentials for low pressure = 50e3
        {0e3, 0e3, 0e3, 0e3, 0e3, 0e3}  // differentials for low pressure = 100e3
    };

    // differentials for half of a U
    // std::vector<std::vector<Real>> actuator_differentials3 = {
    //      {0e3, 0e3, 5e3, 25e3, 37e3, 37e3},
    //      {0e3, 0e3, 6e3, 26.5e3, 39e3, 39e3},
    //      {0e3, 0e3, 6.5e3, 27.2e3, 40e3, 40e3}
    // };


    std::vector<std::vector<std::vector<Real>>> actuator_differentials = {
        actuator_differentials1
        /*actuator_differentials2*/
        /*actuator_differentials3*/
    };
    
    std::vector<int> transition_cycles = {45, 80, 100};
    for (int a = 0; a < num_actuators; a++)
    {
        int differential_index = 0;
        
        int ind = (a%gap);
        while (ind+4 < num_cycles*num_steps_per_cycle)
        {
            const std::vector<std::vector<Real>>& differentials = actuator_differentials[differential_index];
            int next_differential_index = (differential_index < actuator_differentials.size()-1) ? differential_index+1 : actuator_differentials.size()-1;
            const std::vector<std::vector<Real>>& next_differentials = actuator_differentials[next_differential_index];

            Real cycle_num = Real(ind) / num_steps_per_cycle;
            Real t = (4 - std::min(4.0, std::max(0.0,transition_cycles[differential_index]-cycle_num)))/4.0;
            std::cout << t << std::endl;

            Real diff_0e3 = (1-t)*differentials[0][a]+t*next_differentials[0][a];
            Real diff_75e3 = (1-t)*differentials[1][a]+t*next_differentials[1][a];
            Real diff_150e3 = (1-t)*differentials[2][a]+t*next_differentials[2][a];
            // actuator_pressures[a][ind++] = Vec2r(diff_0e3, 0);
            // actuator_pressures[a][ind++] = Vec2r(75e3+diff_75e3, 75e3);
            // actuator_pressures[a][ind++] = Vec2r(150e3+diff_150e3, 150e3);
            // actuator_pressures[a][ind++] = Vec2r(75e3+diff_75e3, 75e3);
            // actuator_pressures[a][ind++] = Vec2r(diff_0e3, 0);
            // actuator_pressures[a][ind++] = Vec2r(diff_0e3, 0);

            actuator_pressures[a][ind++] = Vec2r(0, diff_0e3);
            actuator_pressures[a][ind++] = Vec2r(75e3, 75e3+diff_75e3);
            actuator_pressures[a][ind++] = Vec2r(150e3, 150e3+diff_150e3);
            actuator_pressures[a][ind++] = Vec2r(75e3, 75e3+diff_75e3);
            actuator_pressures[a][ind++] = Vec2r(0, diff_0e3);
            actuator_pressures[a][ind++] = Vec2r(0, diff_0e3);
            // actuator_pressures[a][ind++] = Vec2r(0e3, 0e3);
            // actuator_pressures[a][ind++] = Vec2r(60e3, 60e3);
            // actuator_pressures[a][ind++] = Vec2r(150e3, 150e3);
            // actuator_pressures[a][ind++] = Vec2r(60e3, 60e3);
            // actuator_pressures[a][ind++] = Vec2r(0e3, 0e3);

            ind += (gap-6);

            if (cycle_num > transition_cycles[differential_index])
            {
                differential_index++;
                std::cout << "new diff index: " << differential_index << std::endl;
            }
        }       
        // if (a <= num_actuators/2)
        // {
        //     for (int i = 0; i < num_cycles*num_steps_per_cycle; i++)
        //     {
        //         actuator_pressures[a][i] = Vec2r(0e3, 20e3);
        //     }
        //     while (ind+4 < num_cycles*num_steps_per_cycle)
        //     {
        //         actuator_pressures[a][ind++] = Vec2r(0e3, 20e3);
        //         actuator_pressures[a][ind++] = Vec2r(60e3, 80e3);
        //         actuator_pressures[a][ind++] = Vec2r(120e3, 140e3);
        //         actuator_pressures[a][ind++] = Vec2r(60e3, 80e3);
        //         actuator_pressures[a][ind++] = Vec2r(0e3, 20e3);
        //         // actuator_pressures[a][ind++] = Vec2r(0e3, 0e3);
        //         // actuator_pressures[a][ind++] = Vec2r(60e3, 60e3);
        //         // actuator_pressures[a][ind++] = Vec2r(150e3, 150e3);
        //         // actuator_pressures[a][ind++] = Vec2r(60e3, 60e3);
        //         // actuator_pressures[a][ind++] = Vec2r(0e3, 0e3);

        //         ind += (gap-5);
        //     }
        // }
        // else
        // {
        //     // for (int i = 0; i < num_cycles*num_steps_per_cycle; i++)
        //     // {
        //     //     actuator_pressures[a][i] = Vec2r(120e3, 120e3);
        //     // }
        //     while (ind+4 < num_cycles*num_steps_per_cycle)
        //     {
        //         actuator_pressures[a][ind++] = Vec2r(0e3, 0e3);
        //         actuator_pressures[a][ind++] = Vec2r(80e3, 80e3);
        //         actuator_pressures[a][ind++] = Vec2r(140e3, 140e3);
        //         actuator_pressures[a][ind++] = Vec2r(80e3, 80e3);
        //         actuator_pressures[a][ind++] = Vec2r(0e3, 0e3);

        //         ind += (gap-5);
        //     }
        // }
    }

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