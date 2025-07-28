#ifndef __COSSERAT_ROD_LINEARIZED_HPP
#define __COSSERAT_ROD_LINEARIZED_HPP

#include "common.hpp"
#include "math.hpp"
#include "CosseratRodBase.hpp"
#include "CosseratRodStates.hpp"

template <int NumNodes_>
class CosseratRodWithCrossSectionalDeformationLinearized_OptimizationFunctor;

template <int NumNodes_>
class CosseratRodWithCrossSectionalDeformationLinearized : public CosseratRod_Base<NumNodes_, CosseratRodWithCrossSectionalDeformation_State<NumNodes_>>
{
public:
    constexpr static int NumNodes = NumNodes_;
    constexpr static Real OptTol = 0;

    using State = CosseratRodWithCrossSectionalDeformation_State<NumNodes_>;
    using Base = CosseratRod_Base<NumNodes_, State>;
    using OptimizationFunctor = CosseratRodWithCrossSectionalDeformationLinearized_OptimizationFunctor<NumNodes_>;
    using TipPositionGradientType = typename Base::TipPositionGradientType;
    using EnergyGradientType = typename Base::EnergyGradientType;

public:
    // constructor accepts any type of cross section
    template<typename CrossSectionType_>
    CosseratRodWithCrossSectionalDeformationLinearized(Real length, const CrossSectionType_& cross_section, Real E, Real nu, bool constrain_base=true, bool constrain_tip=false)
        : Base(length, cross_section, E, nu, constrain_base, constrain_tip)
    {   
    }

    /** Computes the total energy (i.e., strain energy - f*x) used by LBFGS.
     * The minimization of this energy over the rod state yields the equilibrium rod state
     * resulting from the applied tip force.
      */
    virtual Real minimizationEnergy(const Vec3r& applied_tip_force) const override;

    /** Computes the gradient of the minimization energy w.r.t. the rod state.
     * Used by LBFGS to minimize the energy.
     */
    virtual EnergyGradientType minimizationEnergyGradient(const Vec3r& applied_tip_force) const override;
};
#include "CosseratRodWithCrossSectionalDeformationLinearized.impl.hpp"

/** Functor used in the LBFGS optimization of the minimiziation energy.
 * Given x (the state), the () operator computes f(x) and the gradient.
 */
template <int NumNodes_>
class CosseratRodWithCrossSectionalDeformationLinearized_OptimizationFunctor
{
public:
    CosseratRodWithCrossSectionalDeformationLinearized_OptimizationFunctor(CosseratRodWithCrossSectionalDeformationLinearized<NumNodes_>* rod, Vec3r applied_tip_force)
        : _rod(rod), _applied_tip_force(applied_tip_force)
    {}

    Real operator() (const VecXr& x, VecXr& grad)
    {
        // convert dynamic VecXr to static StateVecType vector and set the rod's state
        typename CosseratRodWithCrossSectionalDeformationLinearized<NumNodes_>::State::StateVecType state = x.head<CosseratRodWithCrossSectionalDeformationLinearized<NumNodes_>::State::NumStates>();
        _rod->setState(state);

        // compute gradient and energy
        grad = _rod->minimizationEnergyGradient(_applied_tip_force);
        Real energy = _rod->minimizationEnergy(_applied_tip_force);
        // std::cout << "Energy: " << energy << std::endl;

        return energy;
    }

private:
    CosseratRodWithCrossSectionalDeformationLinearized<NumNodes_>* _rod;
    Vec3r _applied_tip_force;
};

#endif // __COSSERAT_ROD_LINEARIZED_HPP