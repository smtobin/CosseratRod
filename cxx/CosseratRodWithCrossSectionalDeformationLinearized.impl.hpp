#ifndef __COSSERAT_ROD_LINEARIZED_IMPL_HPP
#define __COSSERAT_ROD_LINEARIZED_IMPL_HPP

template <int N>
Real CosseratRodWithCrossSectionalDeformationLinearized<N>::minimizationEnergy(const Vec3r& applied_tip_force) const
{
    Real h = this->_length / (N-1);

    Real energy = 0;
    Mat4r T = Mat4r::Identity();

    const typename State::CrossSectionVarVecType a = this->_state.a();
    const typename State::CrossSectionVarVecType b = this->_state.b();
    const typename State::CrossSectionVarVecType c = this->_state.c();
    const typename State::StrainVarVecType v1 = this->_state.v1();
    const typename State::StrainVarVecType v2 = this->_state.v2();
    const typename State::StrainVarVecType v3 = this->_state.v3();
    const typename State::StrainVarVecType u1 = this->_state.u1();
    const typename State::StrainVarVecType u2 = this->_state.u2();
    const typename State::StrainVarVecType u3 = this->_state.u3();

    const Real E = this->_bending_correction ? this->_E : this->_M;
    const Real G = this->_G;
    const Real A = this->_cross_section->A0();
    const Real Ix = this->_cross_section->Ix();
    const Real Iy = this->_cross_section->Iy();
    const Real Iz = this->_cross_section->Iz();
    const Real eta = this->_cross_section->torsionalCorrection();

    const Mat3r M_mat = this->_K.template block<3,3>(0,0);
    // iterate through links (midpoints between the nodes)
    for (int i = 0; i < N-1; i++)
    {
        // cross-sectional parameters at midpoint between nodes
        const Real a_mid = 0.5*(a[i] + a[i+1]);
        const Real b_mid = 0.5*(b[i] + b[i+1]);
        const Real c_mid = 0.5*(c[i] + c[i+1]);

        // get approximate arc-length derivatives of a, b, c
        const Real a_prime = (a[i+1] - a[i]) / h;
        const Real b_prime = (b[i+1] - b[i]) / h;
        const Real c_prime = (c[i+1] - c[i]) / h;

        // compute energy for this segment
        Vec3r strain_vec(a_mid-1, b_mid-1, v3[i]-1);
        Real shear_energy = (strain_vec.transpose() * M_mat * strain_vec);

        energy += 0.5 * h * (
            G*A*v1[i]*v1[i] + G*A*v2[i]*v2[i] + 4*G*A*c_mid*c_mid +
            E*Ix*u1[i]*u1[i] + E*Iy*u2[i]*u2[i] + eta*G*Iz*u3[i]*u3[i] +
            A * shear_energy +
            G*Iy*a_prime*a_prime + G*Ix*b_prime*b_prime + eta*G*Iz*c_prime*c_prime +
            2*G*c_prime*u3[i]*(Iy - Ix)
        );

        // std::cout << "Accumulated energy: " << energy << std::endl;

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
typename CosseratRodWithCrossSectionalDeformationLinearized<N>::EnergyGradientType CosseratRodWithCrossSectionalDeformationLinearized<N>::minimizationEnergyGradient(const Vec3r& applied_tip_force) const
{
    Real h = this->_length / (N-1);

    TipPositionGradientType tip_pos_grad = Base::tipPositionGradient();

    EnergyGradientType energy_grad = EnergyGradientType::Zero();


    const typename State::CrossSectionVarVecType a = this->_state.a();
    const typename State::CrossSectionVarVecType b = this->_state.b();
    const typename State::CrossSectionVarVecType c = this->_state.c();
    const typename State::StrainVarVecType v1 = this->_state.v1();
    const typename State::StrainVarVecType v2 = this->_state.v2();
    const typename State::StrainVarVecType v3 = this->_state.v3();
    const typename State::StrainVarVecType u1 = this->_state.u1();
    const typename State::StrainVarVecType u2 = this->_state.u2();
    const typename State::StrainVarVecType u3 = this->_state.u3();

    const Real E = this->_bending_correction ? this->_E : this->_M;
    const Real G = this->_G;
    const Real A = this->_cross_section->A0();
    const Real Ix = this->_cross_section->Ix();
    const Real Iy = this->_cross_section->Iy();
    const Real Iz = this->_cross_section->Iz();
    const Real eta = this->_cross_section->torsionalCorrection();

    // gradients of U'
    const Mat3r M_mat = this->_K.template block<3,3>(0,0);
    for (int i = 0; i < N-1; i++)
    {
        // cross-sectional parameters at midpoint between nodes
        const Real a_mid = 0.5*(a[i] + a[i+1]);
        const Real b_mid = 0.5*(b[i] + b[i+1]);
        const Real c_mid = 0.5*(c[i] + c[i+1]);

        // get approximate arc-length derivatives of a, b, c
        const Real a_prime = (a[i+1] - a[i]) / h;
        const Real b_prime = (b[i+1] - b[i]) / h;
        const Real c_prime = (c[i+1] - c[i]) / h;

        Vec3r u_vec(u1[i], u2[i], u3[i]);
        Vec3r strain_vec(a_mid-1, b_mid-1, v3[i]-1);

        energy_grad[State::v1Start + i] = h * G * A * v1[i];
        energy_grad[State::v2Start + i] = h * G * A * v2[i];
        energy_grad[State::v3Start + i] = h * A * Vec3r(0,0,1).transpose() * M_mat * strain_vec;
        energy_grad[State::u1Start + i] = h * E * Ix * u1[i];
        energy_grad[State::u2Start + i] = h * E * Iy * u2[i];
        energy_grad[State::u3Start + i] = h * (eta*G*Iz*u3[i] + G*c_prime*( Iy - Ix ) );
        


        Real da_dai = 0.5;
        Real db_dbi = 0.5;
        Real dc_dci = 0.5;

        Real dap_dai = -1.0/h;  Real dap_daiplus1 = 1.0/h;
        Real dbp_dbi = -1.0/h;  Real dbp_dbiplus1 = 1.0/h;
        Real dcp_dci = -1.0/h;  Real dcp_dciplus1 = 1.0/h;

        // w.r.t a_i and a_(i+1)
        energy_grad[State::aStart + i] += 0.5*h * (
            2*A * Vec3r(da_dai,0,0).transpose() * M_mat * strain_vec +
            2*G*Iy*a_prime*dap_dai
        );
        energy_grad[State::aStart + i+1] += 0.5*h * (
            2*A * Vec3r(da_dai,0,0).transpose() * M_mat * strain_vec +
            2*G*Iy*a_prime*dap_daiplus1
        );

        // w.r.t b_i and b_(i+1)
        energy_grad[State::bStart + i] += 0.5*h * (
            2*A* Vec3r(0,db_dbi,0).transpose() * M_mat * strain_vec +
            2*G*Ix*b_prime*dbp_dbi
        );
        energy_grad[State::bStart + i+1] += 0.5*h * (
            2*A* Vec3r(0,db_dbi,0).transpose() * M_mat * strain_vec +
            2*G*Ix*b_prime*dbp_dbiplus1
        );

        // w.r.t c_i and c_(i+1)
        energy_grad[State::cStart + i] += 0.5*h * (
            8*G*A*c_mid*dc_dci + 
            2*eta*G*Iz*c_prime*dcp_dci +
            2*G*u3[i]*(Iy - Ix)*dcp_dci
        );
        energy_grad[State::cStart + i+1] += 0.5*h * (
            8*G*A*c_mid*dc_dci + 
            2*eta*G*Iz*c_prime*dcp_dciplus1 +
            2*G*u3[i]*(Iy - Ix)*dcp_dciplus1
        );
    }

    if (this->_constrain_base)
    {
        energy_grad[State::aStart] = 0;
        energy_grad[State::bStart] = 0;
        energy_grad[State::cStart] = 0;
    }
    
    if (this->_constrain_tip)
    {
        energy_grad[State::aStart+N-1] = 0;
        energy_grad[State::bStart+N-1] = 0;
        energy_grad[State::cStart+N-1] = 0;
    }
    

    // subtract gradient w.r.t tip position
    energy_grad -= applied_tip_force.transpose() * tip_pos_grad;

    return energy_grad;
}

#endif // __COSSERAT_ROD_LINEARIZED_IMPL_HPP