#ifndef __COSSERAT_ROD_IMPL_HPP
#define __COSSERAT_ROD_IMPL_HPP

template <int N>
Vec3r CosseratRod<N>::tipPosition(  Real h,
                                    const typename State::StrainVarVecType& v1,
                                    const typename State::StrainVarVecType& v2,
                                    const typename State::StrainVarVecType& v3,
                                    const typename State::StrainVarVecType& u1,
                                    const typename State::StrainVarVecType& u2,
                                    const typename State::StrainVarVecType& u3)
{

    Mat4r T = Mat4r::Identity();
    for (int i = 0; i < N-1; i++)
    {
        T = T * Math::Exp_se3( h*Vec6r(u1[i], u2[i], u3[i], v1[i], v2[i], v3[i]));
    }

    return T.block<3,1>(0,3);
}

template<int N>
Vec3r CosseratRod<N>::tipPosition()
{
    Real h = _length / (N-1);
    const typename State::StrainVarVecType v1 = _state.v1();
    const typename State::StrainVarVecType v2 = _state.v2();
    const typename State::StrainVarVecType v3 = _state.v3();
    const typename State::StrainVarVecType u1 = _state.u1();
    const typename State::StrainVarVecType u2 = _state.u2();
    const typename State::StrainVarVecType u3 = _state.u3();

    return tipPosition(h, v1, v2, v3, u1, u2, u3);

}

template <int N>
Real CosseratRod<N>::minimizationEnergy(const Vec3r& applied_tip_force)
{
    Real h = _length / (N-1);

    Real energy = 0;
    Mat4r T = Mat4r::Identity();

    const typename State::StrainVarVecType v1 = _state.v1();
    const typename State::StrainVarVecType v2 = _state.v2();
    const typename State::StrainVarVecType v3 = _state.v3();
    const typename State::StrainVarVecType u1 = _state.u1();
    const typename State::StrainVarVecType u2 = _state.u2();
    const typename State::StrainVarVecType u3 = _state.u3();

    // iterate through links (midpoints between the nodes)
    for (int i = 0; i < N-1; i++)
    {
        // compute energy for this segment

        energy += 0.5 * h * (
            _G*_cross_section->A0()*v1[i]*v1[i] + _G*_cross_section->A0()*v2[i]*v2[i] + _E*_cross_section->A0()*(v3[i]-1)*(v3[i]-1) +
            _E*_cross_section->Ix()*u1[i]*u1[i] + _E*_cross_section->Iy()*u2[i]*u2[i] +
            _G*_cross_section->torsionalCorrection()*_cross_section->Iz()*u3[i]*u3[i]
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
typename CosseratRod<N>::TipPositionGradientType CosseratRod<N>::tipPositionGradient()
{
    Real h = _length / (N-1);

    Eigen::Matrix<Real, 3, State::NumStates> grad = Eigen::Matrix<Real, 3, State::NumStates>::Zero();

    // gradient of tip position w.r.t a,b,c is 0

    typename State::StrainVarVecType v1 = _state.v1();
    typename State::StrainVarVecType v2 = _state.v2();
    typename State::StrainVarVecType v3 = _state.v3();
    typename State::StrainVarVecType u1 = _state.u1();
    typename State::StrainVarVecType u2 = _state.u2();
    typename State::StrainVarVecType u3 = _state.u3();

    Vec3r orig_tip_position = tipPosition(h, v1, v2, v3, u1, u2, u3);

    const Real v_delta = 1e-5;
    const Real u_delta = 1e-6;
    for (int i = 0; i < N-1; i++)
    {
        // vary v1
        v1[i] += v_delta;
        Vec3r v1_tip = tipPosition(h, v1, v2, v3, u1, u2, u3);
        grad.col(State::v1Start + i) = (v1_tip - orig_tip_position) / v_delta;
        v1[i] -= v_delta;

        // vary v2
        v2[i] += v_delta;
        Vec3r v2_tip = tipPosition(h, v1, v2, v3, u1, u2, u3);
        grad.col(State::v2Start + i) = (v2_tip - orig_tip_position) / v_delta;
        v2[i] -= v_delta;

        // vary v3
        v3[i] += v_delta;
        Vec3r v3_tip = tipPosition(h, v1, v2, v3, u1, u2, u3);
        grad.col(State::v3Start + i) = (v3_tip - orig_tip_position) / v_delta;
        v3[i] -= v_delta;

        // vary u1
        u1[i] += u_delta;
        Vec3r u1_tip = tipPosition(h, v1, v2, v3, u1, u2, u3);
        grad.col(State::u1Start + i) = (u1_tip - orig_tip_position) / u_delta;
        u1[i] -= u_delta;

        // vary u2
        u2[i] += u_delta;
        Vec3r u2_tip = tipPosition(h, v1, v2, v3, u1, u2, u3);
        grad.col(State::u2Start + i) = (u2_tip - orig_tip_position) / u_delta;
        u2[i] -= u_delta;

        // vary u3
        u3[i] += u_delta;
        Vec3r u3_tip = tipPosition(h, v1, v2, v3, u1, u2, u3);
        grad.col(State::u3Start + i) = (u3_tip - orig_tip_position) / u_delta;
        u3[i] -= u_delta;

    }

    return grad;
}

template <int N>
typename CosseratRod<N>::EnergyGradientType CosseratRod<N>::minimizationEnergyGradient(const Vec3r& applied_tip_force)
{
    Real h = _length / (N-1);

    TipPositionGradientType tip_pos_grad = tipPositionGradient();

    EnergyGradientType energy_grad = Eigen::Vector<Real, State::NumStates>::Zero();

    const typename State::StrainVarVecType v1 = _state.v1();
    const typename State::StrainVarVecType v2 = _state.v2();
    const typename State::StrainVarVecType v3 = _state.v3();
    const typename State::StrainVarVecType u1 = _state.u1();
    const typename State::StrainVarVecType u2 = _state.u2();
    const typename State::StrainVarVecType u3 = _state.u3();

    Real E = _E;
    Real G = _G;
    Real A = _cross_section->A0();
    Real Ix = _cross_section->Ix();
    Real Iy = _cross_section->Iy();
    Real Iz = _cross_section->Iz();
    Real eta = _cross_section->torsionalCorrection();

    // gradients of U'
    const Mat3r M_mat = _K.block<3,3>(0,0);
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