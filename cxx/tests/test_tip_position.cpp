#include "common.hpp"
#include "math.hpp"
#include "CosseratRodWithCrossSectionalDeformation.hpp"

#define N 11

int main()
{

    EllipseCrossSection circle_cross_section(0.5, 0.5);

    Real length = 3.0;
    Real E = 3e6;
    Real nu = 0.45;
    CosseratRodWithCrossSectionalDeformation<N> rod(length, circle_cross_section, E, nu);
    Real h = length / (N-1);

    // set some initial state
    using State = CosseratRodWithCrossSectionalDeformation<N>::State;
    State state = rod.state();

    typename State::StrainVarVecType v1, v2, v3, u1, u2, u3;
    for (int i = 0; i < N-1; i++)
    {
        // u1[i] = i/100.0;
        // u2[i] = 0.0;
        // u3[i] = i/50.0;

        // v1[i] = i/100.0;
        // v2[i] = i/55.0;
        // v3[i] = 1 + i/40.0;
        u1[i] = 0.0;
        u2[i] = 0.0;
        u3[i] = 0.0;

        v1[i] = 0.0;
        v2[i] = 0.0;
        v3[i] = 1.0;
    }

    state.set_u1(u1);
    state.set_u2(u2);
    state.set_u3(u3);

    state.set_v1(v1);
    state.set_v2(v2);
    state.set_v3(v3);

    rod.setState(state);

    Vec3r orig_tip_pos = CosseratRodWithCrossSectionalDeformation<N>::tipPosition(h, v1, v2, v3, u1, u2, u3);
    CosseratRodWithCrossSectionalDeformation<N>::TipPositionGradientType grad = rod.tipPositionGradient();

    // small change in state
    State delta;
    typename State::StrainVarVecType dv1, dv2, dv3, du1, du2, du3;
    for (int i = 0; i < N-1; i++)
    {
        dv1[i] = -0.003 * i/20.0;
        dv2[i] = 0.0005 * i/15.0;
        dv3[i] = -0.002 * i/40.0;
        du1[i] = -0.0006 * i/15.0;
        du2[i] = 0.0008 * i/20.0;
        du3[i] = 0.001 * i/25.0;
    }

    delta.set_u1(du1);
    delta.set_u2(du2);
    delta.set_u3(du3);
    delta.set_v1(dv1);
    delta.set_v2(dv2);
    delta.set_v3(dv3);

    // new state
    State new_state = state + delta;
    typename State::StrainVarVecType nv1, nv2, nv3, nu1, nu2, nu3;
    nv1 = new_state.v1(); nv2 = new_state.v2(); nv3 = new_state.v3();
    nu1 = new_state.u1(); nu2 = new_state.u2(); nu3 = new_state.u3();
    Vec3r new_tip_pos = CosseratRodWithCrossSectionalDeformation<N>::tipPosition(h, nv1, nv2, nv3, nu1, nu2, nu3);

    // get to new state from gradient
    Vec3r new_tip_pos_grad = orig_tip_pos + grad * delta.state_vec;

    // compare
    std::cout << "Original tip position: (" << orig_tip_pos[0] << ", " << orig_tip_pos[1] << ", " << orig_tip_pos[2] << ")" << std::endl;
    std::cout << "New tip position: (" << new_tip_pos[0] << ", " << new_tip_pos[1] << ", " << new_tip_pos[2] << ")" << std::endl;
    std::cout << "New tip position from grad: (" << new_tip_pos_grad[0] << ", " << new_tip_pos_grad[1] << ", " << new_tip_pos_grad[2] << ")" << std::endl;





    return EXIT_SUCCESS;
}