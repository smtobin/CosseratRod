#ifndef __PERISTALTIC_ROBOT_STATE_HPP
#define __PERISTALTIC_ROBOT_STATE_HPP

#include "common.hpp"

/** A struct representing the state variables of the Cosserat rod.
 * This includes the curvatures u, the shear/stretch strains v,
 * and the cross-sectional deformation parameters a, b, and c.
 * 
 * Ultimately just a wrapper around a state vector, with some utilities for extracting
 * and working with different parts of the state easily.
 * 
 */
template <int NumNodes_>
struct PeristalticRobot_State
{
    constexpr static int NumNodes = NumNodes_;
    constexpr static int NumStates = 3*NumNodes + 6*(NumNodes-1);   // number of states in the state vector
    using StateVecType = Eigen::Vector<Real, NumStates>;            // typedef for the entire state vector
    using CrossSectionVarVecType = Eigen::Vector<Real, NumNodes>;   // typedef for a,b,c vectors (they each have N entries)
    using StrainVarVecType = Eigen::Vector<Real, NumNodes-1>;       // typedef for v1,v2,v3,u1,u2,u3 vectors (they each have N-1 entries)

    // start indices for each of the different variables in the state vector
    // each variable is stored contiguously
    constexpr static int aStart = 0;
    constexpr static int bStart = NumNodes;
    constexpr static int cStart = 2*NumNodes;
    constexpr static int v1Start = 3*NumNodes;
    constexpr static int v2Start = 4*NumNodes-1;
    constexpr static int v3Start = 5*NumNodes-2;
    constexpr static int u1Start = 6*NumNodes-3;
    constexpr static int u2Start = 7*NumNodes-4;
    constexpr static int u3Start = 8*NumNodes-5;

    // the vector that holds the entire state of the rod
    StateVecType state_vec;

    // constructor initializes the state to be a straight rod with no deformation energy
    PeristalticRobot_State()
    {
        state_vec = StateVecType::Zero();

        set_a(CrossSectionVarVecType::Ones());
        set_b(CrossSectionVarVecType::Ones());
        set_v3(StrainVarVecType::Ones());
    }

    // adding two states is just adding the two state vectors
    PeristalticRobot_State operator+(const PeristalticRobot_State& other)
    {
        PeristalticRobot_State new_state;
        new_state.state_vec = state_vec + other.state_vec;
        return new_state;
    }

    // subtracting two states is just subtracting one state vector from the other
    PeristalticRobot_State operator-(const PeristalticRobot_State& other)
    {
        PeristalticRobot_State new_state;
        new_state.state_vec = state_vec - other.state_vec;
        return new_state;
    }

    // getters for each state variable
    CrossSectionVarVecType a() const { return state_vec( Eigen::seqN(aStart,NumNodes) ); } 
    CrossSectionVarVecType b() const { return state_vec( Eigen::seqN(bStart, NumNodes) ); }
    CrossSectionVarVecType c() const { return state_vec( Eigen::seqN(cStart, NumNodes) ); }
    StrainVarVecType v1() const { return state_vec( Eigen::seqN(v1Start, NumNodes-1) ); }
    StrainVarVecType v2() const { return state_vec( Eigen::seqN(v2Start, NumNodes-1) ); }
    StrainVarVecType v3() const { return state_vec( Eigen::seqN(v3Start, NumNodes-1) ); }
    StrainVarVecType u1() const { return state_vec( Eigen::seqN(u1Start, NumNodes-1) ); }
    StrainVarVecType u2() const { return state_vec( Eigen::seqN(u2Start, NumNodes-1) ); }
    StrainVarVecType u3() const { return state_vec( Eigen::seqN(u3Start, NumNodes-1) ); }

    // setters for each state variable
    void set_a(const CrossSectionVarVecType& new_a) { state_vec( Eigen::seqN(aStart,NumNodes) ) = new_a; }
    void set_b(const CrossSectionVarVecType& new_b) { state_vec( Eigen::seqN(bStart,NumNodes) ) = new_b; }
    void set_c(const CrossSectionVarVecType& new_c) { state_vec( Eigen::seqN(cStart,NumNodes) ) = new_c; }
    void set_v1(const StrainVarVecType& new_v1) { state_vec(Eigen::seqN(v1Start, NumNodes-1) ) = new_v1; }
    void set_v2(const StrainVarVecType& new_v2) { state_vec(Eigen::seqN(v2Start, NumNodes-1) ) = new_v2; }
    void set_v3(const StrainVarVecType& new_v3) { state_vec(Eigen::seqN(v3Start, NumNodes-1) ) = new_v3; }
    void set_u1(const StrainVarVecType& new_u1) { state_vec(Eigen::seqN(u1Start, NumNodes-1) ) = new_u1; }
    void set_u2(const StrainVarVecType& new_u2) { state_vec(Eigen::seqN(u2Start, NumNodes-1) ) = new_u2; }
    void set_u3(const StrainVarVecType& new_u3) { state_vec(Eigen::seqN(u3Start, NumNodes-1) ) = new_u3; }

    friend std::ostream& operator<<(std::ostream& stream, const PeristalticRobot_State& state)
    {
        stream << "a: " << state.a().transpose() << "\nb: " << state.b().transpose() << "\nc: " << state.c().transpose() <<
         "\nv1: " << state.v1().transpose() << "\nv2: " << state.v2().transpose() << "\nv3: " << state.v3().transpose() << 
            "\nu1: " << state.u1().transpose() << "\nu2: " << state.u2().transpose() << "\nu3: " << state.u3().transpose();
        return stream;
    }
};

#endif // __PERISTALTIC_ROBOT_STATE_HPP