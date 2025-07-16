#ifndef __COSSERAT_HPP
#define __COSSERAT_HPP

#include "common.hpp"
#include "CrossSection.hpp"

#include <memory>

template <int NumNodes_>
class CosseratRod
{
public:
    constexpr static int NumNodes = NumNodes_;

    struct State
    {
        constexpr static int NumStates = 3*NumNodes + 6*(NumNodes-1);
        using StateVecType = Eigen::Vector<Real, NumStates>;
        using CrossSectionVarVecType = Eigen::Vector<Real, NumNodes>;
        using StrainVarVecType = Eigen::Vector<Real, NumNodes-1>;

        constexpr static int aStart = 0;
        constexpr static int bStart = NumNodes;
        constexpr static int cStart = 2*NumNodes;
        constexpr static int v1Start = 3*NumNodes;
        constexpr static int v2Start = 4*NumNodes-1;
        constexpr static int v3Start = 5*NumNodes-2;
        constexpr static int u1Start = 6*NumNodes-3;
        constexpr static int u2Start = 7*NumNodes-4;
        constexpr static int u3Start = 8*NumNodes-5;

        StateVecType state_vec;


        State()
        {
            state_vec = StateVecType::Zero();

            set_a(CrossSectionVarVecType::Ones());
            set_b(CrossSectionVarVecType::Ones());
            set_v3(StrainVarVecType::Ones());
        }

        State operator+(const State& other)
        {
            State new_state;
            new_state.state_vec = state_vec + other.state_vec;
            return new_state;
        }

        State operator-(const State& other)
        {
            State new_state;
            new_state.state_vec = state_vec - other.state_vec;
            return new_state;
        }

        CrossSectionVarVecType a() const { return state_vec( Eigen::seqN(aStart,NumNodes) ); } 
        CrossSectionVarVecType b() const { return state_vec( Eigen::seqN(bStart, NumNodes) ); }
        CrossSectionVarVecType c() const { return state_vec( Eigen::seqN(cStart, NumNodes) ); }
        StrainVarVecType v1() const { return state_vec( Eigen::seqN(v1Start, NumNodes-1) ); }
        StrainVarVecType v2() const { return state_vec( Eigen::seqN(v2Start, NumNodes-1) ); }
        StrainVarVecType v3() const { return state_vec( Eigen::seqN(v3Start, NumNodes-1) ); }
        StrainVarVecType u1() const { return state_vec( Eigen::seqN(u1Start, NumNodes-1) ); }
        StrainVarVecType u2() const { return state_vec( Eigen::seqN(u2Start, NumNodes-1) ); }
        StrainVarVecType u3() const { return state_vec( Eigen::seqN(u3Start, NumNodes-1) ); }
    
        void set_a(const CrossSectionVarVecType& new_a) { state_vec( Eigen::seqN(aStart,NumNodes) ) = new_a; }
        void set_b(const CrossSectionVarVecType& new_b) { state_vec( Eigen::seqN(bStart,NumNodes) ) = new_b; }
        void set_c(const CrossSectionVarVecType& new_c) { state_vec( Eigen::seqN(cStart,NumNodes) ) = new_c; }
        void set_v1(const StrainVarVecType& new_v1) { state_vec(Eigen::seqN(v1Start, NumNodes-1) ) = new_v1; }
        void set_v2(const StrainVarVecType& new_v2) { state_vec(Eigen::seqN(v2Start, NumNodes-1) ) = new_v2; }
        void set_v3(const StrainVarVecType& new_v3) { state_vec(Eigen::seqN(v3Start, NumNodes-1) ) = new_v3; }
        void set_u1(const StrainVarVecType& new_u1) { state_vec(Eigen::seqN(u1Start, NumNodes-1) ) = new_u1; }
        void set_u2(const StrainVarVecType& new_u2) { state_vec(Eigen::seqN(u2Start, NumNodes-1) ) = new_u2; }
        void set_u3(const StrainVarVecType& new_u3) { state_vec(Eigen::seqN(u3Start, NumNodes-1) ) = new_u3; }
    };

public:
    template<typename CrossSectionType_>
    CosseratRod(Real length, const CrossSectionType_& cross_section,
        Real E, Real nu)
        : _length(length), _state(), _E(E), _nu(nu)
    {
        _cross_section = std::make_unique<CrossSectionType_>(cross_section);
        
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

    const State& state() { return _state; }
    void setState(const State& new_state) { _state = new_state; }

    static Real totalEnergy(CosseratRod<NumNodes_>&, const Vec3r& applied_tip_force);

    static Vec3r tipPosition(Real h,
                             const typename State::StrainVarVecType& v1,
                             const typename State::StrainVarVecType& v2,
                             const typename State::StrainVarVecType& v3,
                             const typename State::StrainVarVecType& u1,
                             const typename State::StrainVarVecType& u2,
                             const typename State::StrainVarVecType& u3);

    static Eigen::Matrix<Real, 3, State::NumStates> gradTipPosition(const CosseratRod<NumNodes_>& rod);

    static Eigen::Vector<Real, State::NumStates> gradEnergy(const CosseratRod<NumNodes_>& rod, const Vec3r& applied_tip_force);

    protected:
    void _updateNodeRelTransforms();

    protected:
    Real _length;
    std::unique_ptr<CrossSection> _cross_section;
    State _state;

    // relative transforms between nodes
    std::array<Mat4r, NumNodes_-1> _node_rel_transforms;

    // material properties
    Real _E;
    Real _nu;
    Real _M;
    Real _G;
    Real _lam;
    Mat6r _K;
};

#endif // __COSSERAT_HPP