#ifndef __PERISTALTIC_BENDING_ROBOT_HPP
#define __PERISTALTIC_BENDING_ROBOT_HPP

#include "common.hpp"
#include "math.hpp"
#include "CrossSection.hpp"
#include "PeristalticRobotState.hpp"

#include <vector>
#include <cassert>
#include <fstream>


template<int NumNodes_>
class PeristalticBendingRobot
{
public:
    constexpr static int NumNodes = NumNodes_;

    using State = PeristalticRobot_State<NumNodes_>;

    // typedef for gradient of tip position w.r.t. the state
    using PositionGradientType = Eigen::Matrix<Real, 3, State::NumStates>;
    using PositionAndOrientationGradientType = Eigen::Matrix<Real, 6, State::NumStates>;
    // typedef for gradient of energy w.r.t. the state
    using EnergyGradientType = Eigen::Vector<Real, State::NumStates>;   

public:
    // constructor accepts any type of cross section
    PeristalticBendingRobot(Real length, const EllipseCrossSection& rod_cross_section, Real E, Real nu,
        int num_actuators, Real actuator_length, const EllipseCrossSection& actuator_cross_section)
        : _length(length),  _rod_cross_section(rod_cross_section), 
         _num_actuators(num_actuators), _actuator_length(actuator_length), _actuator_cross_section(actuator_cross_section), _state(),
         _E(E), _nu(nu)
    {   

        // calculate (x,y) coords in the cross section of the actuator
        _actuator_x = 0.5*rod_cross_section.rx();
        std::cout << "actuator x: " << _actuator_x << std::endl;
        // make sure we have an odd number of nodes
        static_assert(NumNodes_ % 2 == 1);

        _center_node = NumNodes_ / 2;

        assert(num_actuators*actuator_length < length);

        // figure out the how many segments each actuator should take up
        Real h = length / (NumNodes_ - 1);
        int segments_per_actuator = std::round(actuator_length / h);
        assert(segments_per_actuator >= 2);
        
        int num_leftover_segments = (NumNodes_-1) - segments_per_actuator * num_actuators;
        assert(num_leftover_segments >= 2*num_actuators);
        int num_cap_segments = 1;
        int num_center_segments = (num_leftover_segments - num_cap_segments*2) / (num_actuators-1);

        std::cout << "Num cap segments: " << num_cap_segments << " Num center segments: " << num_center_segments << std::endl;

        _actuator_intervals.push_back(std::make_pair(num_cap_segments, num_cap_segments+segments_per_actuator));
        for (int i = 1; i < num_actuators; i++)
        {
            int prev_end = _actuator_intervals.back().second;
            int start = (i!=num_actuators-1) ? prev_end + num_center_segments : prev_end + (num_leftover_segments - num_cap_segments*2 - num_center_segments*(num_actuators-2));
            _actuator_intervals.push_back(std::make_pair(start, start+segments_per_actuator));
        }

        // TODO: figure out node spacing based on actuator intervals
        _node_locations.resize(NumNodes_);
        // int first_actuator_start = _actuator_intervals.front().first;
        // for (int i = 0; i < first_actuator_start+1; i++)
        // {
        //     _node_locations[i] = i*actuator_spacing / first_actuator_start;
        // }
        // for (int a = 0; a < _num_actuators; a++)
        // {
        //     for (int i = 0; i < num_nodes_per_actuator)
        //     {
        //         _node_locations[a]
        //     }
        // }
        // int last_actuator_end = _actuator_intervals.back().second;
        for (int i = 0; i < NumNodes_; i++)
        {
            _node_locations[i] = i * h;
        }

        // print out info
        for (int i = 0; i < num_actuators; i++)
        {
            std::cout << "Actuator " << i << " node interval: [" << _actuator_intervals[i].first << ", " << _actuator_intervals[i].second << "]" <<
            "; Length interval: [" << _node_locations[_actuator_intervals[i].first] << ", " << _node_locations[_actuator_intervals[i].second] <<
            "]" << std::endl;
        }


        // initialize material properties from E and nu
        _M = _E * (1-_nu) / ( (1+_nu) * (1-2*_nu) );
        _lam = _E * _nu / ( (1+_nu) * (1-2*_nu) );
        _G = _E / (2 * (1+_nu));
        _K << _M, _lam, _lam, 0, 0, 0,
              _lam, _M, _lam, 0, 0, 0,
              _lam, _lam, _M, 0, 0, 0,
              0, 0, 0, _G, 0, 0,
              0, 0, 0, 0, _G, 0,
              0, 0, 0, 0, 0, _G;
    }
    // getters for rod properties
    Real length() const { return _length; }
    Real E() const { return _E; }
    Real nu() const { return _nu; }
    const CrossSection* crossSection() const { return &_rod_cross_section; }

    int numActuators() const { return _num_actuators; }

    // getter/setters for the rod state
    const State& state() const { return _state; }
    void setState(const State& new_state) { _state = new_state; }
    void setState(const typename State::StateVecType& new_state_vec) { _state.state_vec = new_state_vec; }

    /** Computes the gradient of the tip position w.r.t. the rod state. */
    // TipPositionGradientType tipPositionGradient() const;

    int actuatorNode(int actuator_index) const
    {
        return (_actuator_intervals[actuator_index].first + _actuator_intervals[actuator_index].second)/2;
    }

    Vec3r actuatorPosition(int actuator_index) const;

    Vec3r actuatorPosition(int actuator_index,
        const Vec3r& center_orientation,
        const typename State::StrainVarVecType& v1,
        const typename State::StrainVarVecType& v2,
        const typename State::StrainVarVecType& v3,
        const typename State::StrainVarVecType& u1,
        const typename State::StrainVarVecType& u2,
        const typename State::StrainVarVecType& u3) const;

    PositionGradientType actuatorPositionGradient(int actuator_index) const;

    Vec6r actuatorPositionAndOrientation(int actuator_index) const;

    Vec6r actuatorPositionAndOrientation(int actuator_index,
        const Vec3r& center_orientation,
        const typename State::StrainVarVecType& v1,
        const typename State::StrainVarVecType& v2,
        const typename State::StrainVarVecType& v3,
        const typename State::StrainVarVecType& u1,
        const typename State::StrainVarVecType& u2,
        const typename State::StrainVarVecType& u3) const;

    PositionAndOrientationGradientType actuatorPositionAndOrientationGradient(int actuator_index) const;

    std::vector<Vec3r> nodePositions() const;

    std::vector<Vec3r> nodePositions(
        const Vec3r& center_position,
        const Vec3r& center_orientation,
        const typename State::StrainVarVecType& v1,
        const typename State::StrainVarVecType& v2,
        const typename State::StrainVarVecType& v3,
        const typename State::StrainVarVecType& u1,
        const typename State::StrainVarVecType& u2,
        const typename State::StrainVarVecType& u3) const;

    std::vector<PositionGradientType> nodePositionGradients() const;

    void printNodePositions() const;

    /** Computes the total energy (i.e., strain energy - f*x) used by LBFGS.
     * The minimization of this energy over the rod state yields the equilibrium rod state
     * resulting from the applied tip force.
      */
    Real minimizationEnergy(const std::vector<Vec2r>& actuator_pressures) const;

    /** Computes the gradient of the minimization energy w.r.t. the rod state.
     * Used by LBFGS to minimize the energy.
     */
    EnergyGradientType minimizationEnergyGradient(const std::vector<Vec2r>& actuator_pressures) const;

    /** Writes the peristaltic robot to file. */
    void writeToFile(const std::string& filename) const
    {
        std::ofstream file(filename);
        if (file.is_open())
        {
            file << toString();
        }
    }

    /** Returns a string representing the peristaltic robot. */
    std::string toString() const
    {
        std::stringstream ss;
        ss << NumNodes_ << "\n" << _length << "\n" << _E << "\n" << _nu << "\n" <<
                crossSection()->type() << "\n" << crossSection()->rx() << "\n" << crossSection()->ry() << "\n" <<
                _state.state_vec;
        
        return ss.str();
    }

protected:
    Real _length;   // the length of the rod (z-axis dimension)
    EllipseCrossSection _rod_cross_section;   // the cross section of the rod

    int _num_actuators; // number of pneumatic actuators
    Real _actuator_length;  // the length of each pneumatic actuator
    EllipseCrossSection _actuator_cross_section; // the cross section of the pneumatic actuators

    std::vector<std::pair<int,int>> _actuator_intervals;    // node indices (start, end) of the beginning and end of each actuator

    std::vector<Real> _node_locations;  // the location (in terms of undeformed length along the rod) of each node

    int _center_node;

    Real _actuator_x;

    State _state;

    // material properties
    Real _E;
    Real _nu;
    Real _M;
    Real _G;
    Real _lam;
    Mat6r _K;
};
#include "PeristalticBendingRobot.impl.hpp"

/** Functor used in the LBFGS optimization of the minimiziation energy.
 * Given x (the state), the () operator computes f(x) and the gradient.
 */
template <int NumNodes_>
class PeristalticBendingRobot_OptimizationFunctor
{
public:
    PeristalticBendingRobot_OptimizationFunctor(PeristalticBendingRobot<NumNodes_>* robot, const std::vector<Vec2r>& actuation_pressures)
        : _robot(robot), _actuation_pressures(actuation_pressures)
    {}

    Real operator() (const VecXr& x, VecXr& grad)
    {
        // convert dynamic VecXr to static StateVecType vector and set the rod's state
        using StateVecType = typename PeristalticBendingRobot<NumNodes_>::State::StateVecType;
        StateVecType state = x.head<PeristalticBendingRobot<NumNodes_>::State::NumStates>();
        _robot->setState(state);

        // compute gradient and energy
        grad = _robot->minimizationEnergyGradient(_actuation_pressures);
        Real energy = _robot->minimizationEnergy(_actuation_pressures);
        // std::cout << "Energy: " << energy << std::endl;

        return energy;
    }

private:
    PeristalticBendingRobot<NumNodes_>* _robot;
    const std::vector<Vec2r>& _actuation_pressures;
};

#include "../alglib-cpp/src/ap.h"
template <int N>
struct PeristalticBendingRobot_Optimization
{
    struct UserInfo
    {
        PeristalticBendingRobot<N>* robot;
        std::vector<Vec2r> actuation_pressures;
        std::vector<Vec6r> actuation_positions;
    };

    static void ground_func(const alglib::real_1d_array& x, alglib::real_1d_array& fi, alglib::real_2d_array& jac, void* ptr)
    {
        UserInfo* info = static_cast<UserInfo*>(ptr);
        PeristalticBendingRobot<N>* robot = info->robot;
        
        using StateVecType = typename PeristalticBendingRobot<N>::State::StateVecType;
        const StateVecType state = Eigen::Map<const StateVecType>(x.getcontent());
        robot->setState(state);

        typename PeristalticBendingRobot<N>::EnergyGradientType grad = robot->minimizationEnergyGradient(info->actuation_pressures);
        Real energy = robot->minimizationEnergy(info->actuation_pressures);

        // set optimization function and its gradient
        fi[0] = energy;

        int num_states = PeristalticBendingRobot<N>::State::NumStates;
        for (int i = 0; i < num_states; i++)
            jac[0][i] = grad[i];

        // get actuator position constraint functions and their gradients w.r.t state
        // Real radius = robot->crossSection()->ry();
        // for (int a = 0; a < robot->numActuators(); a++)
        // {
        //     Vec3r pos = robot->actuatorPosition(a);
        //     Vec3r pos_diff = (pos - info->actuation_positions[a]);
        //     fi[a*3 + 1] = pos_diff[0];
        //     fi[a*3 + 2] = pos_diff[1];
        //     fi[a*3 + 3] = pos_diff[2];
        //     typename PeristalticBendingRobot<N>::PositionGradientType pos_grad = robot->actuatorPositionGradient(a);
        //     for (int i = 0; i < num_states; i++)
        //     {
        //         jac[a*3 + 1][i] = pos_grad(0,i);
        //         jac[a*3 + 2][i] = pos_grad(1,i);
        //         jac[a*3 + 3][i] = pos_grad(2,i);
        //     }
        // }

        Real radius = robot->crossSection()->ry();
        for (int a = 0; a < robot->numActuators(); a++)
        {
            Vec6r pos = robot->actuatorPositionAndOrientation(a);
            Vec6r pos_diff = (pos - info->actuation_positions[a]);
            fi[a*6 + 1] = pos_diff[0];
            fi[a*6 + 2] = pos_diff[1];
            fi[a*6 + 3] = pos_diff[2];
            fi[a*6 + 4] = pos_diff[3];
            fi[a*6 + 5] = pos_diff[4];
            fi[a*6 + 6] = pos_diff[5];
            typename PeristalticBendingRobot<N>::PositionAndOrientationGradientType pos_grad 
                = robot->actuatorPositionAndOrientationGradient(a);

            for (int i = 0; i < num_states; i++)
            {
                jac[a*6 + 1][i] = pos_grad(0,i);
                jac[a*6 + 2][i] = pos_grad(1,i);
                jac[a*6 + 3][i] = pos_grad(2,i);
                jac[a*6 + 4][i] = pos_grad(3,i);
                jac[a*6 + 5][i] = pos_grad(4,i);
                jac[a*6 + 6][i] = pos_grad(5,i);
            }
        }

        // get node above-ground constraint functions and their gradients w.r.t state
        // std::vector<Vec3r> node_positions = robot->nodePositions();
        // std::vector<typename PeristalticBendingRobot<N>::PositionGradientType> node_position_gradients = robot->nodePositionGradients();
        // Real radius = robot->crossSection()->ry();
        // for (int i = 0; i < N; i++)
        // {
        //     int func_ind = 3*robot->numActuators()+1 + i;
        //     Real z_diff = (node_positions[i][2] - state[PeristalticBendingRobot<N>::State::bStart+i]*radius - 0);
        //     fi[func_ind] = z_diff;

        //     for (int j = 0; j < num_states; j++)
        //     {
        //         jac[func_ind][j] = node_position_gradients[i](2,j);
        //     }
        //     jac[func_ind][PeristalticBendingRobot<N>::State::bStart+i] = -radius;
        // }
    }
};

#endif // __PERISTALTIC_BENDING_ROBOT_HPP