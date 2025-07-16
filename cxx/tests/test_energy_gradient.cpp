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

    // set some initial state
    using State = CosseratRodWithCrossSectionalDeformation<N>::State;
    State state = rod.state();

    typename State::StrainVarVecType v1, v2, v3, u1, u2, u3;
    for (int i = 0; i < N-1; i++)
    {
        u1[i] = i/100.0;
        u2[i] = 0.0;
        u3[i] = i/50.0;

        v1[i] = i/100.0;
        v2[i] = i/55.0;
        v3[i] = 1 + i/40.0;
        // u1[i] = 0.0;
        // u2[i] = 0.0;
        // u3[i] = 0.0;

        // v1[i] = 0.0;
        // v2[i] = 0.0;
        // v3[i] = 1.0;
    }

    state.set_u1(u1);
    state.set_u2(u2);
    state.set_u3(u3);

    state.set_v1(v1);
    state.set_v2(v2);
    state.set_v3(v3);

    rod.setState(state);

    Vec3r applied_tip_force(1000, 2000, 2500);
    Real orig_energy = rod.minimizationEnergy(applied_tip_force);
    CosseratRodWithCrossSectionalDeformation<N>::EnergyGradientType energy_grad = rod.minimizationEnergyGradient(applied_tip_force);

    // small change in state
    State delta;
    typename State::StrainVarVecType dv1, dv2, dv3, du1, du2, du3;
    typename State::CrossSectionVarVecType da, db, dc;
    for (int i = 0; i < N-1; i++)
    {
        dv1[i] = -0.003 * i/200.0;
        dv2[i] = 0.0005 * i/150.0;
        dv3[i] = -0.002 * i/400.0;
        du1[i] = -0.0006 * i/150.0;
        du2[i] = 0.0008 * i/200.0;
        du3[i] = 0.001 * i/250.0;
    }

    for (int i = 0; i < N; i++)
    {
        da[i] = 0.0001;
        db[i] = -0.0004;
        dc[i] = 0.00005;
    }

    delta.set_a(da);
    delta.set_b(db);
    delta.set_c(dc);
    delta.set_u1(du1);
    delta.set_u2(du2);
    delta.set_u3(du3);
    delta.set_v1(dv1);
    delta.set_v2(dv2);
    delta.set_v3(dv3);

    // new state
    State new_state = state + delta;
    rod.setState(new_state);
    Real new_energy = rod.minimizationEnergy(applied_tip_force);

    // new energy approximation from gradient
    Real new_energy_approx = orig_energy + energy_grad.dot(delta.state_vec);


    // compare
    std::cout << "Original energy: " << orig_energy << std::endl;
    std::cout << "New energy: " << new_energy << std::endl;
    std::cout << "New energy (From gradient): " << new_energy_approx << std::endl; 

    return EXIT_SUCCESS;
}