#ifndef __PERISTALTIC_ROBOT_IMPL_HPP
#define __PERISTALTIC_ROBOT_IMPL_HPP

template<int N>
Vec3r PeristalticRobot<N>::actuatorPosition(int actuator_index) const
{
    const typename State::StrainVarVecType v1 = _state.v1();
    const typename State::StrainVarVecType v2 = _state.v2();
    const typename State::StrainVarVecType v3 = _state.v3();
    const typename State::StrainVarVecType u1 = _state.u1();
    const typename State::StrainVarVecType u2 = _state.u2();
    const typename State::StrainVarVecType u3 = _state.u3();

    return actuatorPosition(actuator_index, v1, v2, v3, u1, u2, u3);
}

template<int N>
Vec3r PeristalticRobot<N>::actuatorPosition(int actuator_index,
        const typename State::StrainVarVecType& v1,
        const typename State::StrainVarVecType& v2,
        const typename State::StrainVarVecType& v3,
        const typename State::StrainVarVecType& u1,
        const typename State::StrainVarVecType& u2,
        const typename State::StrainVarVecType& u3) const
{
    int actuator_node = (_actuator_intervals[actuator_index].first + _actuator_intervals[actuator_index].second)/2;
    // std::cout << "Actuator " << actuator_index << " node index: " << actuator_node << std::endl;
    Vec3r center_position = _state.p();

    if (actuator_node == _center_node)
        return center_position;
    
    Mat4r T = Mat4r::Identity();
    T.block<3,1>(0,3) = center_position;

    // integrate forwards from the center
    if (actuator_node > _center_node)
    {
        for (int i = _center_node; i < actuator_node; i++)
        {
            Real h = _node_locations[i+1] - _node_locations[i];
            T = T * Math::Exp_se3( h*Vec6r(u1[i], u2[i], u3[i], v1[i], v2[i], v3[i]));
        }
    }
    // integrate backwards from the center
    else
    {
        for (int i = _center_node; i > actuator_node; i--)
        {
            Real h = _node_locations[i-1] - _node_locations[i]; // h is negative since we are going backwards
            T = T * Math::Exp_se3( h*Vec6r(u1[i-1], u2[i-1], u3[i-1], v1[i-1], v2[i-1], v3[i-1]));
        }
    }

    Vec3r pos = T.block<3,1>(0,3);

    // std::cout << "Actuator " << actuator_index << " position: " << pos.transpose() << std::endl;
    return pos;
}

template<int N>
typename PeristalticRobot<N>::PositionGradientType PeristalticRobot<N>::actuatorPositionGradient(int actuator_index) const
{

    PositionGradientType grad = PositionGradientType::Zero();

    // gradient of actuator position w.r.t center position is just identity
    grad.template block<3,3>(0, State::pStart) = Mat3r::Identity();

    // gradient of tip position w.r.t a,b,c is 0

    typename State::StrainVarVecType v1 = _state.v1();
    typename State::StrainVarVecType v2 = _state.v2();
    typename State::StrainVarVecType v3 = _state.v3();
    typename State::StrainVarVecType u1 = _state.u1();
    typename State::StrainVarVecType u2 = _state.u2();
    typename State::StrainVarVecType u3 = _state.u3();

    Vec3r orig_pos = actuatorPosition(actuator_index, v1, v2, v3, u1, u2, u3);

    /** TODO: don't need to do every single u and v, just the ones between the center and the actuator */
    const Real v_delta = 1e-5;
    const Real u_delta = 1e-6;
    for (int i = 0; i < N-1; i++)
    {
        // vary v1
        v1[i] += v_delta;
        Vec3r v1_tip = actuatorPosition(actuator_index, v1, v2, v3, u1, u2, u3);
        grad.col(State::v1Start + i) = (v1_tip - orig_pos) / v_delta;
        v1[i] -= v_delta;

        // vary v2
        v2[i] += v_delta;
        Vec3r v2_tip = actuatorPosition(actuator_index, v1, v2, v3, u1, u2, u3);
        grad.col(State::v2Start + i) = (v2_tip - orig_pos) / v_delta;
        v2[i] -= v_delta;

        // vary v3
        v3[i] += v_delta;
        Vec3r v3_tip = actuatorPosition(actuator_index, v1, v2, v3, u1, u2, u3);
        grad.col(State::v3Start + i) = (v3_tip - orig_pos) / v_delta;
        v3[i] -= v_delta;

        // vary u1
        u1[i] += u_delta;
        Vec3r u1_tip = actuatorPosition(actuator_index, v1, v2, v3, u1, u2, u3);
        grad.col(State::u1Start + i) = (u1_tip - orig_pos) / u_delta;
        u1[i] -= u_delta;

        // vary u2
        u2[i] += u_delta;
        Vec3r u2_tip = actuatorPosition(actuator_index, v1, v2, v3, u1, u2, u3);
        grad.col(State::u2Start + i) = (u2_tip - orig_pos) / u_delta;
        u2[i] -= u_delta;

        // vary u3
        u3[i] += u_delta;
        Vec3r u3_tip = actuatorPosition(actuator_index, v1, v2, v3, u1, u2, u3);
        grad.col(State::u3Start + i) = (u3_tip - orig_pos) / u_delta;
        u3[i] -= u_delta;

    }

    return grad;
}

template<int N>
void PeristalticRobot<N>::printNodePositions() const
{
    Mat4r T = Mat4r::Identity();

    const typename State::StrainVarVecType v1 = _state.v1();
    const typename State::StrainVarVecType v2 = _state.v2();
    const typename State::StrainVarVecType v3 = _state.v3();
    const typename State::StrainVarVecType u1 = _state.u1();
    const typename State::StrainVarVecType u2 = _state.u2();
    const typename State::StrainVarVecType u3 = _state.u3();

    std::cout << "Node 0 location: " << T.block<3,1>(0,3).transpose() << std::endl;
    for (int i = 0; i < N-1; i++)
    {
        Real h = _node_locations[i+1] - _node_locations[i];
        T = T * Math::Exp_se3( h*Vec6r(u1[i], u2[i], u3[i], v1[i], v2[i], v3[i]));
        std::cout << "Node " << i+1 << " location: " << T.block<3,1>(0,3).transpose() << std::endl;
    }
}

// template <int N>
// Real PeristalticRobot<N>::actuatorVolume(int actuator_index,
//         const typename State::CrossSectionVarVecType& a,
//         const typename State::CrossSectionVarVecType& b,
//         const typename State::CrossSectionVarVecType& c,
//         const typename State::StrainVarVecType& v1,
//         const typename State::StrainVarVecType& v2,
//         const typename State::StrainVarVecType& v3,
//         const typename State::StrainVarVecType& u1,
//         const typename State::StrainVarVecType& u2,
//         const typename State::StrainVarVecType& u3) const
// {
//     Real actuator_volume = 0;
//     const Real A_act = _actuator_cross_section.A0();
//     for (int i = 0; i < N-1; i++)
//     {
//         Real h = _node_locations[i+1] - _node_locations[i];

//         // cross-sectional parameters at midpoint between nodes
//         const Real a_mid = 0.5*(a[i] + a[i+1]);
//         const Real b_mid = 0.5*(b[i] + b[i+1]);
//         const Real c_mid = 0.5*(c[i] + c[i+1]);

//         // check if we need to increment the actuator index
//         if (i >= _actuator_intervals[actuator_index].second)
//             break;

//         // check and see if we are at a segment that is spanned by a pneumatic actuator
//         std::pair<int,int> actuator_interval = _actuator_intervals[actuator_index];
//         if (i >= actuator_interval.first && i < actuator_interval.second)
//         {
//             // calculate current volume of pneumatic actuator
//             actuator_volume += h * (a_mid*b_mid - c_mid*c_mid) * A_act;
//         }
//         else
//         {
//             // if not move on to the next node
//             continue;
//         }
//     }

//     return actuator_volume;
// }

template <int N>
Real PeristalticRobot<N>::minimizationEnergy(const std::vector<Real>& actuation_pressures) const
{
    Real energy = 0;
    Mat4r T = Mat4r::Identity();

    const typename State::CrossSectionVarVecType a = _state.a();
    const typename State::CrossSectionVarVecType b = _state.b();
    const typename State::CrossSectionVarVecType c = _state.c();
    const typename State::StrainVarVecType v1 = _state.v1();
    const typename State::StrainVarVecType v2 = _state.v2();
    const typename State::StrainVarVecType v3 = _state.v3();
    const typename State::StrainVarVecType u1 = _state.u1();
    const typename State::StrainVarVecType u2 = _state.u2();
    const typename State::StrainVarVecType u3 = _state.u3();

    const Real E = _E;
    const Real G = _G;
    const Real A_rod = _rod_cross_section.A0();
    const Real Ix_rod = _rod_cross_section.Ix();
    const Real Iy_rod = _rod_cross_section.Iy();
    const Real Iz_rod = _rod_cross_section.Iz();
    const Real eta = _rod_cross_section.torsionalCorrection();

    const Real A_act = _actuator_cross_section.A0();
    const Real Ix_act = _actuator_cross_section.Ix();
    const Real Iy_act = _actuator_cross_section.Iy();
    const Real Iz_act = _actuator_cross_section.Iz();

    Mat3r M_mat = _K.template block<3,3>(0,0);

    std::vector<Real> current_actuator_volumes(_num_actuators);
    std::vector<Real> rest_actuator_volumes(_num_actuators);
    int current_actuator_index = 0;

    // iterate through links (midpoints between the nodes)
    for (int i = 0; i < N-1; i++)
    {
        Real h = _node_locations[i+1] - _node_locations[i];

        // cross-sectional parameters at midpoint between nodes
        const Real a_mid = 0.5*(a[i] + a[i+1]);
        const Real b_mid = 0.5*(b[i] + b[i+1]);
        const Real c_mid = 0.5*(c[i] + c[i+1]);

        // get approximate arc-length derivatives of a, b, c
        const Real a_prime = (a[i+1] - a[i]) / h;
        const Real b_prime = (b[i+1] - b[i]) / h;
        const Real c_prime = (c[i+1] - c[i]) / h;

        // check if we need to increment the actuator index
        if (i >= _actuator_intervals[current_actuator_index].second)
            current_actuator_index++;

        // check and see if we are at a segment that is spanned by a pneumatic actuator
        std::pair<int,int> cur_actuator_interval = _actuator_intervals[current_actuator_index];
        Real A, Ix, Iy, Iz;
        if (i >= cur_actuator_interval.first && i < cur_actuator_interval.second)
        {
            // std::cout << "i = " << i << ", spanned by actuator " << current_actuator_index << std::endl;
            // if we are, set the cross-sectional properties accordingly
            A = A_rod - A_act;
            Ix = Ix_rod - Ix_act;
            Iy = Iy_rod - Iy_act;
            Iz = Iz_rod - Iz_act;

            // calculate current volume of pneumatic actuator
            current_actuator_volumes[current_actuator_index] += h * v3[i] * (a_mid*b_mid - c_mid*c_mid) * A_act;
            rest_actuator_volumes[current_actuator_index] += h * A_act;
        }
        else
        {
            // if not, then the cross-sectional properties are just that of the normal rod
            A = A_rod;
            Ix = Ix_rod;
            Iy = Iy_rod;
            Iz = Iz_rod;
        }

        // compute energy for this segment

        Mat3r K_mat = Mat3r::Zero();
        K_mat(0,0) = E * (Ix * b_mid * b_mid + Iy * c_mid * c_mid);
        K_mat(1,1) = E * (Iy * a_mid * a_mid + Ix * c_mid * c_mid);
        K_mat(0,1) = -E * (Iy * a_mid * c_mid + Ix * b_mid * c_mid);
        K_mat(1,0) = K_mat(0,1);
        K_mat(2,2) = eta*G*Iz*c_mid*c_mid +
            G * Iy*a_mid*a_mid +
            G * Ix*b_mid*b_mid;

        Vec3r u_vec(u1[i], u2[i], u3[i]);
        Vec3r strain_vec(a_mid-1, b_mid-1, v3[i]-1);

        Real curvature_energy = (u_vec.transpose() * K_mat * u_vec);
        Real shear_energy = (strain_vec.transpose() * M_mat * strain_vec);

        energy += 0.5 * h * (
            G*A*v1[i]*v1[i] + G*A*v2[i]*v2[i] + 4*G*A*c_mid*c_mid +
            curvature_energy +
            A * shear_energy +
            G*Iy*a_prime*a_prime + G*Ix*b_prime*b_prime + eta*G*Iz*c_prime*c_prime +
            2*G*u3[i]*( (a_mid*c_prime - a_prime*c_mid)*Iy - (b_mid*c_prime - b_prime*c_mid)*Ix)
        );

        // std::cout << "Accumulated energy: " << energy << std::endl;

        // compute the transform from the base to node i
        T = T * Math::Exp_se3( h*Vec6r(u1[i], u2[i], u3[i], v1[i], v2[i], v3[i]));
    }

    // Vec3r x_dir = T.block<3,1>(0,0);
    // Vec3r y_dir = T.block<3,1>(0,1);
    Vec3r tip_pos = T.block<3,1>(0,3);

    // std::cout << "Tip position: (" << tip_pos[0] << ", " << tip_pos[1] << ", " << tip_pos[2] << ")" <<  std::endl;
    
    // add contributions from chamber compression
    for (int i = 0; i < _num_actuators; i++)
    {
        Real dV = current_actuator_volumes[i] - rest_actuator_volumes[i];
        energy -= dV * actuation_pressures[i];
    }

    return energy;
}

template <int N>
typename PeristalticRobot<N>::EnergyGradientType PeristalticRobot<N>::minimizationEnergyGradient(const std::vector<Real>& actuation_pressures) const
{
    Real h = _length / (N-1);

    EnergyGradientType energy_grad = EnergyGradientType::Zero();


    const typename State::CrossSectionVarVecType a = _state.a();
    const typename State::CrossSectionVarVecType b = _state.b();
    const typename State::CrossSectionVarVecType c = _state.c();
    const typename State::StrainVarVecType v1 = _state.v1();
    const typename State::StrainVarVecType v2 = _state.v2();
    const typename State::StrainVarVecType v3 = _state.v3();
    const typename State::StrainVarVecType u1 = _state.u1();
    const typename State::StrainVarVecType u2 = _state.u2();
    const typename State::StrainVarVecType u3 = _state.u3();

    const Real E = _E;
    const Real G = _G;
    const Real A_rod = _rod_cross_section.A0();
    const Real Ix_rod = _rod_cross_section.Ix();
    const Real Iy_rod = _rod_cross_section.Iy();
    const Real Iz_rod = _rod_cross_section.Iz();
    const Real eta = _rod_cross_section.torsionalCorrection();

    const Real A_act = _actuator_cross_section.A0();
    const Real Ix_act = _actuator_cross_section.Ix();
    const Real Iy_act = _actuator_cross_section.Iy();
    const Real Iz_act = _actuator_cross_section.Iz();

    // gradients of U'
    int current_actuator_index = 0;
    const Mat3r M_mat = _K.template block<3,3>(0,0);
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

        Real da_dai = 0.5;
        Real db_dbi = 0.5;
        Real dc_dci = 0.5;

        Real dap_dai = -1.0/h;  Real dap_daiplus1 = 1.0/h;
        Real dbp_dbi = -1.0/h;  Real dbp_dbiplus1 = 1.0/h;
        Real dcp_dci = -1.0/h;  Real dcp_dciplus1 = 1.0/h;

        // check if we need to increment the actuator index
        if (i >= _actuator_intervals[current_actuator_index].second)
            current_actuator_index++;

        // check and see if we are at a segment that is spanned by a pneumatic actuator
        std::pair<int,int> cur_actuator_interval = _actuator_intervals[current_actuator_index];
        Real A, Ix, Iy, Iz;
        if (i >= cur_actuator_interval.first && i < cur_actuator_interval.second)
        {
            // if we are, set the cross-sectional properties accordingly
            A = A_rod - A_act;
            Ix = Ix_rod - Ix_act;
            Iy = Iy_rod - Iy_act;
            Iz = Iz_rod - Iz_act;

            // calculate gradients with respect to energy due to volume change of pneumatic actuators
            energy_grad[State::aStart + i] += h * v3[i] * da_dai*b_mid * A_act * -actuation_pressures[current_actuator_index];
            energy_grad[State::aStart + i+1] += h * v3[i] * da_dai*b_mid * A_act * -actuation_pressures[current_actuator_index];
            energy_grad[State::bStart + i] += h * v3[i] * db_dbi*a_mid * A_act * -actuation_pressures[current_actuator_index];
            energy_grad[State::bStart + i+1] += h * v3[i] * db_dbi*a_mid * A_act * -actuation_pressures[current_actuator_index];
            energy_grad[State::cStart + i] += h * v3[i] * -2*c_mid*dc_dci * A_act * -actuation_pressures[current_actuator_index];
            energy_grad[State::cStart + i+1] += h * v3[i] * -2*c_mid*dc_dci * A_act * -actuation_pressures[current_actuator_index];
            energy_grad[State::v3Start + i] += h * (a_mid*b_mid - c_mid*c_mid) * A_act * -actuation_pressures[current_actuator_index];
        }
        else
        {
            // if not, then the cross-sectional properties are just that of the normal rod
            A = A_rod;
            Ix = Ix_rod;
            Iy = Iy_rod;
            Iz = Iz_rod;
        }

        Mat3r K_mat = Mat3r::Zero();
        K_mat(0,0) = E * (Ix * b_mid * b_mid + Iy * c_mid * c_mid);
        K_mat(1,1) = E * (Iy * a_mid * a_mid + Ix * c_mid * c_mid);
        K_mat(0,1) = -E * (Iy * a_mid * c_mid + Ix * b_mid * c_mid);
        K_mat(1,0) = K_mat(0,1);
        K_mat(2,2) = G * Iy * (a_mid * a_mid + c_mid * c_mid) +
            G * Ix * (b_mid * b_mid + c_mid * c_mid);

        Vec3r u_vec(u1[i], u2[i], u3[i]);
        Vec3r strain_vec(a_mid-1, b_mid-1, v3[i]-1);

        energy_grad[State::v1Start + i] += h * G * A * v1[i];
        energy_grad[State::v2Start + i] += h * G * A * v2[i];
        energy_grad[State::v3Start + i] += h * A * Vec3r(0,0,1).transpose() * M_mat * strain_vec;
        energy_grad[State::u1Start + i] += h * Vec3r(1,0,0).transpose() * K_mat * u_vec;
        energy_grad[State::u2Start + i] += h * Vec3r(0,1,0).transpose() * K_mat * u_vec;
        energy_grad[State::u3Start + i] += h * Vec3r(0,0,1).transpose() * K_mat * u_vec +
            2*G*( (a_mid*c_prime - a_prime*c_mid)*Iy - (b_mid*c_prime - b_prime*c_mid)*Ix );
        

        Mat3r dKmat_da;
        dKmat_da << 0, -E*Iy*c_mid*da_dai, 0,
                    -E*Iy*c_mid*da_dai, 2*E*Iy*a_mid*da_dai, 0,
                    0, 0, 2*eta*G*Iy*a_mid*da_dai;
        Real uvec_dKmat_da = u_vec.transpose() * dKmat_da * u_vec;

        Mat3r dKmat_db;
        dKmat_db << 2*E*Ix*b_mid*db_dbi, -E*Ix*c_mid*db_dbi, 0,
                    -E*Ix*c_mid*db_dbi, 0, 0,
                    0, 0, 2*eta*G*Ix*b_mid*db_dbi;
        Real uvec_dKmat_db = u_vec.transpose() * dKmat_db * u_vec;

        Mat3r dKmat_dc;
        dKmat_dc << 2*E*Iy*c_mid*dc_dci, -E*(Iy*a_mid + Ix*b_mid)*dc_dci, 0,
                    -E*(Iy*a_mid + Ix*b_mid)*dc_dci*dc_dci, 2*E*Ix*c_mid*dc_dci, 0,
                    0, 0, 2*eta*G*Iz*c_mid*dc_dci;
        Real uvec_dKmat_dc = u_vec.transpose() * dKmat_dc * u_vec;

        // w.r.t a_i and a_(i+1)
        energy_grad[State::aStart + i] += 0.5*h * (
            2*A * Vec3r(da_dai,0,0).transpose() * M_mat * strain_vec +
            uvec_dKmat_da +
            2*G*Iy*a_prime*dap_dai +
            2*G*u3[i]*Iy*(c_prime*da_dai - c_mid*dap_dai)
        );
        energy_grad[State::aStart + i+1] += 0.5*h * (
            2*A * Vec3r(da_dai,0,0).transpose() * M_mat * strain_vec +
            uvec_dKmat_da +
            2*G*Iy*a_prime*dap_daiplus1 +
            2*G*u3[i]*Iy*(c_prime*da_dai - c_mid*dap_daiplus1)
        );

        // w.r.t b_i and b_(i+1)
        energy_grad[State::bStart + i] += 0.5*h * (
            2*A* Vec3r(0,db_dbi,0).transpose() * M_mat * strain_vec +
            uvec_dKmat_db +
            2*G*Ix*b_prime*dbp_dbi +
            2*G*u3[i]*Ix*(c_prime*db_dbi - c_mid*dbp_dbi)
        );
        energy_grad[State::bStart + i+1] += 0.5*h * (
            2*A* Vec3r(0,db_dbi,0).transpose() * M_mat * strain_vec +
            uvec_dKmat_db +
            2*G*Ix*b_prime*dbp_dbiplus1 +
            2*G*u3[i]*Ix*(c_prime*db_dbi - c_mid*dbp_dbiplus1)
        );

        // w.r.t c_i and c_(i+1)
        energy_grad[State::cStart + i] += 0.5*h * (
            8*G*A*c_mid*dc_dci + 
            uvec_dKmat_dc +
            2*eta*G*Iz*c_prime*dcp_dci +
            2*G*u3[i]*( (a_mid*dcp_dci - a_prime*dc_dci)*Iy - (b_mid*dcp_dci - b_prime*dc_dci)*Ix )
        );
        energy_grad[State::cStart + i+1] += 0.5*h * (
            8*G*A*c_mid*dc_dci + 
            uvec_dKmat_dc +
            2*eta*G*Iz*c_prime*dcp_dciplus1 +
            2*G*u3[i]*( (a_mid*dcp_dciplus1 - a_prime*dc_dci)*Iy - (b_mid*dcp_dciplus1 - b_prime*dc_dci)*Ix )
        );
    }

    // energy_grad[State::aStart] = 0;
    // energy_grad[State::bStart] = 0;
    // energy_grad[State::cStart] = 0;

    // energy_grad[State::aStart+N-1] = 0;
    // energy_grad[State::bStart+N-1] = 0;
    // energy_grad[State::cStart+N-1] = 0;
    

    return energy_grad;
}

#endif // __PERISTLATIC_ROBOT_IMPL_HPP