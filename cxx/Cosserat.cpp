#include "Cosserat.hpp"
#include "math.hpp"

template <int N>
Real CosseratRod<N>::totalEnergy(CosseratRod<N>& rod, const Vec3r& applied_tip_force)
{
    rod._updateNodeRelTransforms();

    Real h = rod._length / (N-1);

    Real energy = 0;
    Mat4r T = Mat4r::Identity();

    const typename State::CrossSectionVarVecType a = rod._state.a();
    const typename State::CrossSectionVarVecType b = rod._state.b();
    const typename State::CrossSectionVarVecType c = rod._state.c();
    const typename State::StrainVarVecType v1 = rod._state.v1();
    const typename State::StrainVarVecType v2 = rod._state.v2();
    const typename State::StrainVarVecType v3 = rod._state.v3();
    const typename State::StrainVarVecType u1 = rod._state.u1();
    const typename State::StrainVarVecType u2 = rod._state.u2();
    const typename State::StrainVarVecType u3 = rod._state.u3();

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

        Mat3r K_mat = Mat3r::Zero();
        K_mat(0,0) = rod._E * (rod._cross_section->Ix() * b_mid * b_mid + rod._cross_section->Iy() * c_mid * c_mid);
        K_mat(1,1) = rod._E * (rod._cross_section->Iy() * a_mid * a_mid + rod._cross_section->Ix() * c_mid * c_mid);
        K_mat(0,1) = -rod._E * (rod._cross_section->Iy() * a_mid * c_mid + rod._cross_section->Ix() * b_mid * c_mid);
        K_mat(1,0) = K_mat(0,1);
        K_mat(2,2) = rod._cross_section->torsionalCorrection()*rod._G*rod._cross_section->Iz()*c_mid*c_mid +
            rod._G * rod._cross_section->Iy()*a_mid*a_mid +
            rod._G * rod._cross_section->Ix()*b_mid*b_mid;

        Vec3r u_vec(u1[i], u2[i], u3[i]);
        Vec3r strain_vec(a_mid-1, b_mid-1, v3[i]-1);
        Mat3r M_mat = rod._K.block<3,3>(0,0);

        Real curvature_energy = (u_vec.transpose() * K_mat * u_vec);
        Real shear_energy = (strain_vec.transpose() * M_mat * strain_vec);

        energy += 0.5 * h * (
            rod._G*rod._cross_section->A0()*v1[i]*v1[i] + rod._G*rod._cross_section->A0()*v2[i]*v2[i] + 4*rod._G*rod._cross_section->A0()*c_mid*c_mid +
            curvature_energy +
            rod._cross_section->A0() * shear_energy +
            rod._G*rod._cross_section->Iy()*a_prime*a_prime + rod._G*rod._cross_section->Ix()*b_prime*b_prime + rod._cross_section->torsionalCorrection()*rod._G*rod._cross_section->Iz()*c_prime*c_prime +
            2*rod._G*u3[i]*( (a_mid*c_prime - a_prime*c_mid)*rod._cross_section->Iy() - (b_mid*c_prime - b_prime*c_mid)*rod._cross_section->Ix())
        );

        std::cout << "Accumulated energy: " << energy << std::endl;

        // compute the transform from the base to node i
        T = T * rod._node_rel_transforms[i];
    }

    // Vec3r x_dir = T.block<3,1>(0,0);
    // Vec3r y_dir = T.block<3,1>(0,1);
    Vec3r tip_pos = T.block<3,1>(0,3);

    std::cout << "Tip position: (" << tip_pos[0] << ", " << tip_pos[1] << ", " << tip_pos[2] << ")" <<  std::endl;

    energy -= applied_tip_force.dot(tip_pos);

    return energy;
}

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

template <int N>
Eigen::Matrix<Real, 3, CosseratRod<N>::State::NumStates> CosseratRod<N>::gradTipPosition(const CosseratRod<N>& rod)
{
    Real h = rod._length / (N-1);

    Eigen::Matrix<Real, 3, State::NumStates> grad = Eigen::Matrix<Real, 3, State::NumStates>::Zero();

    // gradient of tip position w.r.t a,b,c is 0

    typename State::StrainVarVecType v1 = rod._state.v1();
    typename State::StrainVarVecType v2 = rod._state.v2();
    typename State::StrainVarVecType v3 = rod._state.v3();
    typename State::StrainVarVecType u1 = rod._state.u1();
    typename State::StrainVarVecType u2 = rod._state.u2();
    typename State::StrainVarVecType u3 = rod._state.u3();

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
Eigen::Vector<Real, CosseratRod<N>::State::NumStates> CosseratRod<N>::gradEnergy(const CosseratRod<N>& rod, const Vec3r& applied_tip_force)
{
    Real h = rod._length / (N-1);

    Eigen::Matrix<Real, 3, State::NumStates> tip_pos_grad = gradTipPosition(rod);

    Eigen::Vector<Real, State::NumStates> energy_grad = Eigen::Vector<Real, State::NumStates>::Zero();


    const typename State::CrossSectionVarVecType a = rod._state.a();
    const typename State::CrossSectionVarVecType b = rod._state.b();
    const typename State::CrossSectionVarVecType c = rod._state.c();
    const typename State::StrainVarVecType v1 = rod._state.v1();
    const typename State::StrainVarVecType v2 = rod._state.v2();
    const typename State::StrainVarVecType v3 = rod._state.v3();
    const typename State::StrainVarVecType u1 = rod._state.u1();
    const typename State::StrainVarVecType u2 = rod._state.u2();
    const typename State::StrainVarVecType u3 = rod._state.u3();

    Real E = rod._E;
    Real G = rod._G;
    Real A = rod._cross_section->A0();
    Real Ix = rod._cross_section->Ix();
    Real Iy = rod._cross_section->Iy();
    Real Iz = rod._cross_section->Iz();
    Real eta = rod._cross_section->torsionalCorrection();

    // gradients of U'
    const Mat3r M_mat = rod._K.block<3,3>(0,0);
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

        Mat3r K_mat = Mat3r::Zero();
        K_mat(0,0) = rod._E * (rod._cross_section->Ix() * b_mid * b_mid + rod._cross_section->Iy() * c_mid * c_mid);
        K_mat(1,1) = rod._E * (rod._cross_section->Iy() * a_mid * a_mid + rod._cross_section->Ix() * c_mid * c_mid);
        K_mat(0,1) = -rod._E * (rod._cross_section->Iy() * a_mid * c_mid + rod._cross_section->Ix() * b_mid * c_mid);
        K_mat(1,0) = K_mat(0,1);
        K_mat(2,2) = rod._G * rod._cross_section->Iy() * (a_mid * a_mid + c_mid * c_mid) +
            rod._G * rod._cross_section->Ix() * (b_mid * b_mid + c_mid * c_mid);

        Vec3r u_vec(u1[i], u2[i], u3[i]);
        Vec3r strain_vec(a_mid-1, b_mid-1, v3[i]-1);

        energy_grad[State::v1Start + i] = h * rod._G * rod._cross_section->A0() * v1[i];
        energy_grad[State::v2Start + i] = h * rod._G * rod._cross_section->A0() * v2[i];
        energy_grad[State::v3Start + i] = h * rod._cross_section->A0() * Vec3r(0,0,1).transpose() * M_mat * strain_vec;
        energy_grad[State::u1Start + i] = h * Vec3r(1,0,0).transpose() * K_mat * u_vec;
        energy_grad[State::u2Start + i] = h * Vec3r(0,1,0).transpose() * K_mat * u_vec;
        energy_grad[State::u3Start + i] = h * Vec3r(0,0,1).transpose() * K_mat * u_vec +
            2*rod._G*( (a_mid*c_prime - a_prime*c_mid)*rod._cross_section->Iy() - (b_mid*c_prime - b_prime*c_mid)*rod._cross_section->Ix());
        


        Real da_dai = 0.5;  Real da_daiplus1 = 0.5;
        Real db_dbi = 0.5;  Real db_dbiplus1 = 0.5;
        Real dc_dci = 0.5;  Real dc_dciplus1 = 0.5;

        Real dap_dai = -1.0/h;  Real dap_daiplus1 = 1.0/h;
        Real dbp_dbi = -1.0/h;  Real dbp_dbiplus1 = 1.0/h;
        Real dcp_dci = -1.0/h;  Real dcp_dciplus1 = 1.0/h;

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

    // subtract gradient w.r.t tip position
    energy_grad -= applied_tip_force.transpose() * tip_pos_grad;

    return energy_grad;
}

template<int N>
void CosseratRod<N>::_updateNodeRelTransforms()
{
    Real h = _length / (N-1);

    const typename State::StrainVarVecType v1 = _state.v1();
    const typename State::StrainVarVecType v2 = _state.v2();
    const typename State::StrainVarVecType v3 = _state.v3();
    const typename State::StrainVarVecType u1 = _state.u1();
    const typename State::StrainVarVecType u2 = _state.u2();
    const typename State::StrainVarVecType u3 = _state.u3();

    for (int i = 0; i < N-1; i++)
    {
        // compute the transform from the base to node i
        _node_rel_transforms[i] = Math::Exp_se3( h*Vec6r(u1[i], u2[i], u3[i], v1[i], v2[i], v3[i]));
    }
}

template class CosseratRod<11>;