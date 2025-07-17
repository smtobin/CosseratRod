#ifndef __COSSERAT_ROD_IMPL_HPP
#define __COSSERAT_ROD_IMPL_HPP

template <int N>
Real CosseratRod<N>::minimizationEnergy(const Vec3r& applied_tip_force) const
{
    Real h = this->_length / (N-1);

    Real energy = 0;
    Mat4r T = Mat4r::Identity();

    const typename State::StrainVarVecType v1 = this->_state.v1();
    const typename State::StrainVarVecType v2 = this->_state.v2();
    const typename State::StrainVarVecType v3 = this->_state.v3();
    const typename State::StrainVarVecType u1 = this->_state.u1();
    const typename State::StrainVarVecType u2 = this->_state.u2();
    const typename State::StrainVarVecType u3 = this->_state.u3();

    const Real E = this->_E;
    const Real G = this->_G;
    const Real A = this->_cross_section->A0();
    const Real Ix = this->_cross_section->Ix();
    const Real Iy = this->_cross_section->Iy();
    const Real Iz = this->_cross_section->Iz();
    const Real eta = this->_cross_section->torsionalCorrection();

    // iterate through links (midpoints between the nodes)
    for (int i = 0; i < N-1; i++)
    {
        // compute energy for this segment

        energy += 0.5 * h * (
            G*A*v1[i]*v1[i] + G*A*v2[i]*v2[i] + E*A*(v3[i]-1)*(v3[i]-1) +
            E*Ix*u1[i]*u1[i] + E*Iy*u2[i]*u2[i] +
            G*eta*Iz*u3[i]*u3[i]
        );

        // compute the transform from the base to node i
        T = T * Math::Exp_se3( h*Vec6r(u1[i], u2[i], u3[i], v1[i], v2[i], v3[i]));
    }

    // Vec3r x_dir = T.block<3,1>(0,0);
    // Vec3r y_dir = T.block<3,1>(0,1);
    Vec3r tip_pos = T.block<3,1>(0,3);

    // std::cout << "Tip position: (" << tip_pos[0] << ", " << tip_pos[1] << ", " << tip_pos[2] << ")" <<  std::endl;

    energy -= applied_tip_force.dot(tip_pos);

    return energy;
}

template <int N>
typename CosseratRod<N>::EnergyGradientType CosseratRod<N>::minimizationEnergyGradient(const Vec3r& applied_tip_force) const
{
    Real h = this->_length / (N-1);

    TipPositionGradientType tip_pos_grad = Base::tipPositionGradient();

    EnergyGradientType energy_grad = EnergyGradientType::Zero();

    const typename State::StrainVarVecType v1 = this->_state.v1();
    const typename State::StrainVarVecType v2 = this->_state.v2();
    const typename State::StrainVarVecType v3 = this->_state.v3();
    const typename State::StrainVarVecType u1 = this->_state.u1();
    const typename State::StrainVarVecType u2 = this->_state.u2();
    const typename State::StrainVarVecType u3 = this->_state.u3();

    const Real E = this->_E;
    const Real G = this->_G;
    const Real A = this->_cross_section->A0();
    const Real Ix = this->_cross_section->Ix();
    const Real Iy = this->_cross_section->Iy();
    const Real Iz = this->_cross_section->Iz();
    const Real eta = this->_cross_section->torsionalCorrection();

    // gradients of U'
    for (int i = 0; i < N-1; i++)
    {
        energy_grad[State::v1Start + i] = h * G * A * v1[i];
        energy_grad[State::v2Start + i] = h * G * A * v2[i];
        energy_grad[State::v3Start + i] = h * E * A * (v3[i] - 1);
        energy_grad[State::u1Start + i] = h * E * Ix * u1[i];
        energy_grad[State::u2Start + i] = h * E * Iy * u2[i];
        energy_grad[State::u3Start + i] = h * G * eta * Iz * u3[i];
    }

    // subtract gradient w.r.t tip position
    energy_grad -= applied_tip_force.transpose() * tip_pos_grad;

    return energy_grad;
}

#endif // __COSSERAT_IMPL_HPP