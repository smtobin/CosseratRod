#ifndef __PERISTALTIC_ROBOT_HPP
#define __PERISTALTIC_ROBOT_HPP

#include "common.hpp"
#include "math.hpp"
#include "CrossSection.hpp"
#include "PeristalticRobotState.hpp"

#include <vector>
#include <cassert>

template<int NumNodes_>
class PeristalticRobot_OptimizationFunctor;

template<int NumNodes_>
class PeristalticRobot
{
public:
    constexpr static int NumNodes = NumNodes_;
    constexpr static Real OptTol = 0;

    using State = PeristalticRobot_State<NumNodes_>;
    using OptimizationFunctor = PeristalticRobot_OptimizationFunctor<NumNodes_>;

    // typedef for gradient of tip position w.r.t. the state
    using TipPositionGradientType = Eigen::Matrix<Real, 3, State::NumStates>;
    // typedef for gradient of energy w.r.t. the state
    using EnergyGradientType = Eigen::Vector<Real, State::NumStates>;   

public:
    // constructor accepts any type of cross section
    PeristalticRobot(Real length, const EllipseCrossSection& rod_cross_section, Real E, Real nu,
        int num_actuators, Real actuator_length, const EllipseCrossSection& actuator_cross_section)
        : _length(length),  _rod_cross_section(rod_cross_section), 
         _num_actuators(num_actuators), _actuator_length(actuator_length), _actuator_cross_section(actuator_cross_section), _state(),
         _E(E), _nu(nu)
    {   
        assert(num_actuators*actuator_length < length);

        // figure out the how many nodes each actuator should take up
        Real h = length / (NumNodes_ - 1);
        int num_nodes_per_actuator = std::round(actuator_length / h);
        assert(num_nodes_per_actuator > 1);
        
        Real actuator_spacing = (length - num_actuators*actuator_length) / (num_actuators+1);
        int num_leftover_nodes = NumNodes_ - (num_nodes_per_actuator+1) * num_actuators;
        assert(num_leftover_nodes >= num_actuators+1);
        int num_cap_nodes = num_leftover_nodes / (num_actuators+1);
        int num_center_nodes = (num_leftover_nodes - num_cap_nodes*2) / (num_actuators-1);

        std::cout << "Num cap nodes: " << num_cap_nodes << " Num center nodes: " << num_center_nodes << std::endl;

        _actuator_intervals.push_back(std::make_pair(num_cap_nodes, num_cap_nodes+num_nodes_per_actuator));
        for (int i = 1; i < num_actuators; i++)
        {
            int prev_end = _actuator_intervals.back().second;
            int start = (i==num_actuators-1) ? prev_end + num_center_nodes+1 : prev_end + (num_leftover_nodes - num_cap_nodes*2 - num_center_nodes*(num_actuators-2))+1;
            _actuator_intervals.push_back(std::make_pair(start, start+num_nodes_per_actuator));
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

    // getter/setters for the rod state
    const State& state() const { return _state; }
    void setState(const State& new_state) { _state = new_state; }
    void setState(const typename State::StateVecType& new_state_vec) { _state.state_vec = new_state_vec; }

    /** Computes the tip position given:
     * @param h - the rest distance between adjacent nodes
     * @param v1, v2, v3 - the shear/extension strain values
     * @param u1, u2, u3 - the bending/torsion strain values
     * 
     * This is useful as a static method for numerically computing the tip position gradient, where we make a lot of small
     * changes to the state variables and want to see how the tip position changes.
     */
    // static Vec3r tipPosition(Real h,
    //                          const typename State::StrainVarVecType& v1,
    //                          const typename State::StrainVarVecType& v2,
    //                          const typename State::StrainVarVecType& v3,
    //                          const typename State::StrainVarVecType& u1,
    //                          const typename State::StrainVarVecType& u2,
    //                          const typename State::StrainVarVecType& u3);

    /** Computes the tip position of the rod in its current state. */
    // Vec3r tipPosition() const;

    /** Computes the gradient of the tip position w.r.t. the rod state. */
    // TipPositionGradientType tipPositionGradient() const;

    /** Computes the total energy (i.e., strain energy - f*x) used by LBFGS.
     * The minimization of this energy over the rod state yields the equilibrium rod state
     * resulting from the applied tip force.
      */
    Real minimizationEnergy(const std::vector<Real>& actuator_pressures) const;

    /** Computes the gradient of the minimization energy w.r.t. the rod state.
     * Used by LBFGS to minimize the energy.
     */
    EnergyGradientType minimizationEnergyGradient(const std::vector<Real>& actuator_pressures) const;

protected:
    Real _length;   // the length of the rod (z-axis dimension)
    EllipseCrossSection _rod_cross_section;   // the cross section of the rod

    int _num_actuators; // number of pneumatic actuators
    Real _actuator_length;  // the length of each pneumatic actuator
    EllipseCrossSection _actuator_cross_section; // the cross section of the pneumatic actuators

    std::vector<std::pair<int,int>> _actuator_intervals;    // node indices (start, end) of the beginning and end of each actuator

    std::vector<Real> _node_locations;  // the location (in terms of undeformed length along the rod) of each node

    State _state;

    // material properties
    Real _E;
    Real _nu;
    Real _M;
    Real _G;
    Real _lam;
    Mat6r _K;
};
#include "PeristalticRobot.impl.hpp"

/** Functor used in the LBFGS optimization of the minimiziation energy.
 * Given x (the state), the () operator computes f(x) and the gradient.
 */
template <int NumNodes_>
class PeristalticRobot_OptimizationFunctor
{
public:
    PeristalticRobot_OptimizationFunctor(PeristalticRobot<NumNodes_>* robot, const std::vector<Real>& actuation_pressures)
        : _robot(robot), _actuation_pressures(actuation_pressures)
    {}

    Real operator() (const VecXr& x, VecXr& grad)
    {
        // convert dynamic VecXr to static StateVecType vector and set the rod's state
        using StateVecType = typename PeristalticRobot<NumNodes_>::State::StateVecType;
        StateVecType state = x.head<PeristalticRobot<NumNodes_>::State::NumStates>();
        _robot->setState(state);

        // compute gradient and energy
        grad = _robot->minimizationEnergyGradient(_actuation_pressures);
        Real energy = _robot->minimizationEnergy(_actuation_pressures);
        // std::cout << "Energy: " << energy << std::endl;

        return energy;
    }

private:
    PeristalticRobot<NumNodes_>* _robot;
    const std::vector<Real>& _actuation_pressures;
};

#endif // __PERISTALTIC_ROBOT_HPP