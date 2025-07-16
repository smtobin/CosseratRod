#ifndef __COSSERAT_ROD_HPP
#define __COSSERAT_ROD_HPP

#include "common.hpp"
#include "math.hpp"
#include "CrossSection.hpp"

#include <memory>

template <int NumNodes_>
class CosseratRod
{
public:
    constexpr static int NumNodes = NumNodes_;

    /** A struct representing the state variables of the Cosserat rod.
     * This includes the curvatures u, the shear/stretch strains v,
     * and the cross-sectional deformation parameters a, b, and c.
     * 
     * Ultimately just a wrapper around a state vector, with some utilities for extracting
     * and working with different parts of the state easily.
     * 
     */
    struct State
    {
        constexpr static int NumStates = 6*(NumNodes-1);   // number of states in the state vector
        using StateVecType = Eigen::Vector<Real, NumStates>;            // typedef for the entire state vector
        using StrainVarVecType = Eigen::Vector<Real, NumNodes-1>;       // typedef for v1,v2,v3,u1,u2,u3 vectors (they each have N-1 entries)

        // start indices for each of the different variables in the state vector
        // each variable is stored contiguously
        constexpr static int v1Start = 0;
        constexpr static int v2Start = NumNodes-1;
        constexpr static int v3Start = 2*(NumNodes-1);
        constexpr static int u1Start = 3*(NumNodes-1);
        constexpr static int u2Start = 4*(NumNodes-1);
        constexpr static int u3Start = 5*(NumNodes-1);

        // the vector that holds the entire state of the rod
        StateVecType state_vec;

        // constructor initializes the state to be a straight rod with no deformation energy
        State()
        {
            state_vec = StateVecType::Zero();

            set_v3(StrainVarVecType::Ones());
        }

        // adding two states is just adding the two state vectors
        State operator+(const State& other)
        {
            State new_state;
            new_state.state_vec = state_vec + other.state_vec;
            return new_state;
        }

        // subtracting two states is just subtracting one state vector from the other
        State operator-(const State& other)
        {
            State new_state;
            new_state.state_vec = state_vec - other.state_vec;
            return new_state;
        }

        // getters for each state variable
        StrainVarVecType v1() const { return state_vec( Eigen::seqN(v1Start, NumNodes-1) ); }
        StrainVarVecType v2() const { return state_vec( Eigen::seqN(v2Start, NumNodes-1) ); }
        StrainVarVecType v3() const { return state_vec( Eigen::seqN(v3Start, NumNodes-1) ); }
        StrainVarVecType u1() const { return state_vec( Eigen::seqN(u1Start, NumNodes-1) ); }
        StrainVarVecType u2() const { return state_vec( Eigen::seqN(u2Start, NumNodes-1) ); }
        StrainVarVecType u3() const { return state_vec( Eigen::seqN(u3Start, NumNodes-1) ); }
    
        // setters for each state variable
        void set_v1(const StrainVarVecType& new_v1) { state_vec(Eigen::seqN(v1Start, NumNodes-1) ) = new_v1; }
        void set_v2(const StrainVarVecType& new_v2) { state_vec(Eigen::seqN(v2Start, NumNodes-1) ) = new_v2; }
        void set_v3(const StrainVarVecType& new_v3) { state_vec(Eigen::seqN(v3Start, NumNodes-1) ) = new_v3; }
        void set_u1(const StrainVarVecType& new_u1) { state_vec(Eigen::seqN(u1Start, NumNodes-1) ) = new_u1; }
        void set_u2(const StrainVarVecType& new_u2) { state_vec(Eigen::seqN(u2Start, NumNodes-1) ) = new_u2; }
        void set_u3(const StrainVarVecType& new_u3) { state_vec(Eigen::seqN(u3Start, NumNodes-1) ) = new_u3; }
    };

    using TipPositionGradientType = Eigen::Matrix<Real, 3, State::NumStates>;   // typedef for gradient of tip position w.r.t. the state
    using EnergyGradientType = Eigen::Vector<Real, State::NumStates>;   // typedef for gradient of energy w.r.t. the state

public:
    // constructor accepts any type of cross section
    template<typename CrossSectionType_>
    CosseratRod(Real length, const CrossSectionType_& cross_section,
        Real E, Real nu)
        : _length(length), _state(), _E(E), _nu(nu)
    {
        // make a copy of the cross section
        _cross_section = std::make_unique<CrossSectionType_>(cross_section);
        
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

    // getter/setters for the rod state
    const State& state() { return _state; }
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
    static Vec3r tipPosition(Real h,
                             const typename State::StrainVarVecType& v1,
                             const typename State::StrainVarVecType& v2,
                             const typename State::StrainVarVecType& v3,
                             const typename State::StrainVarVecType& u1,
                             const typename State::StrainVarVecType& u2,
                             const typename State::StrainVarVecType& u3);

    /** Computes the tip position of the rod in its current state. */
    Vec3r tipPosition();

    /** Computes the total energy (i.e., strain energy - f*x) used by LBFGS.
     * The minimization of this energy over the rod state yields the equilibrium rod state
     * resulting from the applied tip force.
      */
    Real minimizationEnergy(const Vec3r& applied_tip_force);

    /** Computes the gradient of the tip position w.r.t. the rod state. */
    TipPositionGradientType tipPositionGradient();

    /** Computes the gradient of the minimization energy w.r.t. the rod state.
     * Used by LBFGS to minimize the energy.
     */
    EnergyGradientType minimizationEnergyGradient(const Vec3r& applied_tip_force);

    protected:
    Real _length;
    std::unique_ptr<CrossSection> _cross_section;
    State _state;

    // material properties
    Real _E;
    Real _nu;
    Real _M;
    Real _G;
    Real _lam;
    Mat6r _K;
};
#include "CosseratRod.impl.hpp"

/** Functor used in the LBFGS optimization of the minimiziation energy.
 * Given x (the state), the () operator computes f(x) and the gradient.
 */
template <int NumNodes_>
class CosseratRodOptimizationFunctor
{
public:
    CosseratRodOptimizationFunctor(CosseratRod<NumNodes_>* rod, Vec3r applied_tip_force)
        : _rod(rod), _applied_tip_force(applied_tip_force)
    {}

    Real operator() (const VecXr& x, VecXr& grad)
    {
        // convert dynamic VecXr to static StateVecType vector and set the rod's state
        typename CosseratRod<NumNodes_>::State::StateVecType state = x.head<CosseratRod<NumNodes_>::State::NumStates>();
        _rod->setState(state);

        // compute gradient and energy
        grad = _rod->minimizationEnergyGradient(_applied_tip_force);
        Real energy = _rod->minimizationEnergy(_applied_tip_force);
        // std::cout << "Energy: " << energy << std::endl;

        return energy;
    }

private:
    CosseratRod<NumNodes_>* _rod;
    Vec3r _applied_tip_force;
};

#endif // __COSSERAT_HPP