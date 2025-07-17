#ifndef __COSSERAT_ROD_BASE_HPP
#define __COSSERAT_ROD_BASE_HPP

#include "math.hpp"
#include "common.hpp"
#include "CrossSection.hpp"

#include <memory>


template <int NumNodes_, typename State>
class CosseratRod_Base
{
public:
    constexpr static int NumNodes = NumNodes_;
    // number of "rod strain variables", i.e. the combined number of variables in (v1,v2,v3,u1,u2,u3)
    constexpr static int NumRodStrainVariables = 6*(NumNodes_-1);

    // typedef for vector of a single rod strain variable
    using RodStrainVariableVecType = Eigen::Vector<Real, NumNodes_-1>;
    // typedef for vector containing all the rod strain variables
    using AllRodStrainVariablesVecType = Eigen::Vector<Real, NumRodStrainVariables>;

    // typedef for gradient of tip position w.r.t. the state
    using TipPositionGradientType = Eigen::Matrix<Real, 3, State::NumStates>;
    // typedef for gradient of energy w.r.t. the state
    using EnergyGradientType = Eigen::Vector<Real, State::NumStates>;   

    // constructor accepts any type of cross section
    template<typename CrossSectionType_>
    CosseratRod_Base(Real length, const CrossSectionType_& cross_section, Real E, Real nu)
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
    Vec3r tipPosition() const;

    /** Computes the gradient of the tip position w.r.t. the rod state. */
    TipPositionGradientType tipPositionGradient() const;

    /** Computes the total energy (i.e., strain energy - f*x) used by LBFGS.
     * The minimization of this energy over the rod state yields the equilibrium rod state
     * resulting from the applied tip force.
      */
    virtual Real minimizationEnergy(const Vec3r& applied_tip_force) const = 0;

    /** Computes the gradient of the minimization energy w.r.t. the rod state.
     * Used by LBFGS to minimize the energy.
     */
    virtual EnergyGradientType minimizationEnergyGradient(const Vec3r& applied_tip_force) const = 0;

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

#include "CosseratRodBase.impl.hpp"

#endif // __COSSERAT_ROD_BASE_HPP