
template <int N>
Real CosseratRodWithLinearModesOfCrossSectionalDeformation<N>::minimizationEnergy(const Vec3r& applied_tip_force) const
{
    Real h = this->_length / (N-1);

    Real energy = 0;
    Mat4r T = Mat4r::Identity();

    const typename State::CrossSectionVarVecType a0 = this->_state.a0();
    const typename State::CrossSectionVarVecType ax = this->_state.ax();
    const typename State::CrossSectionVarVecType ay = this->_state.ay();
    const typename State::CrossSectionVarVecType b0 = this->_state.b0();
    const typename State::CrossSectionVarVecType bx = this->_state.bx();
    const typename State::CrossSectionVarVecType by = this->_state.by();
    const typename State::CrossSectionVarVecType c0 = this->_state.c0();
    const typename State::CrossSectionVarVecType cx = this->_state.cx();
    const typename State::CrossSectionVarVecType cy = this->_state.cy();
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
    const Real Qx = this->_cross_section->Qx();
    const Real Qy = this->_cross_section->Qy();
    const Real Qxy = this->_cross_section->Qxy();
    const Real eta = this->_cross_section->torsionalCorrection();

    Mat6r K = this->_K;
    // TODO: how to apply torsional correction?
    // K(5,5) *= eta;  // apply torsional correction
    for (int i = 0; i < N-1; i++)
    {
        // cross-sectional parameters at midpoint between nodes
        const Real a0_mid = 0.5*(a0[i] + a0[i+1]);
        const Real ax_mid = 0.5*(ax[i] + ax[i+1]);
        const Real ay_mid = 0.5*(ay[i] + ay[i+1]);
        const Real b0_mid = 0.5*(b0[i] + b0[i+1]);
        const Real bx_mid = 0.5*(bx[i] + bx[i+1]);
        const Real by_mid = 0.5*(by[i] + by[i+1]);
        const Real c0_mid = 0.5*(c0[i] + c0[i+1]);
        const Real cx_mid = 0.5*(cx[i] + cx[i+1]);
        const Real cy_mid = 0.5*(cy[i] + cy[i+1]);

        // get approximate arc-length derivatives of a, b, c
        const Real a0_prime = (a0[i+1] - a0[i]) / h;
        const Real ax_prime = (ax[i+1] - ax[i]) / h;
        const Real ay_prime = (ay[i+1] - ay[i]) / h;
        const Real b0_prime = (b0[i+1] - b0[i]) / h;
        const Real bx_prime = (bx[i+1] - bx[i]) / h;
        const Real by_prime = (by[i+1] - by[i]) / h;
        const Real c0_prime = (c0[i+1] - c0[i]) / h;
        const Real cx_prime = (cx[i+1] - cx[i]) / h;
        const Real cy_prime = (cy[i+1] - cy[i]) / h;

        Vec6r e0(
            a0_mid - 1,
            b0_mid - 1,
            v3[i] - 1,
            v2[i],
            v1[i],
            2*c0_mid
        );

        Vec6r ex(
            2*ax_mid,
            bx_mid + cy_mid,
            c0_mid*u1[i] - a0_mid*u2[i],
            c0_prime + a0_mid*u3[i],
            a0_prime - c0_mid*u3[i],
            3*cx_mid + ay_mid
        );

        Vec6r ey(
            cx_mid + ay_mid,
            2*by_mid,
            b0_mid*u1[i] - c0_mid*u2[i],
            b0_prime + c0_mid*u3[i],
            c0_prime - b0_mid*u3[i],
            bx_mid + 3*cy_mid
        );

        Vec6r ex2(
            0,
            0,
            cx_mid*u1[i] - ax_mid*u2[i],
            cx_prime + ax_mid*u3[i],
            ax_prime - cx_mid*u3[i],
            0
        );

        Vec6r ey2(
            0,
            0,
            by_mid*u1[i] - cy_mid*u2[i],
            by_prime + cy_mid*u3[i],
            cy_prime - by_mid*u3[i],
            0
        );

        Vec6r exy(
            0,
            0,
            cy_mid*u1[i] - ay_mid*u2[i] + bx_mid*u1[i] - cx_mid*u2[i],
            bx_prime + cy_prime + ay_mid*u3[i] + cx_mid*u3[i],
            ay_prime + cx_prime - bx_mid*u3[i] - cy_mid*u3[i],
            0
        );

        energy += 0.5 * h * (
            (e0.transpose() * K * e0 * A) +
            (ex.transpose() * K * ex * Iy) +
            (ey.transpose() * K * ey * Ix) +
            (2*e0.transpose() * K * ex2 * Iy) +
            (2*e0.transpose() * K * ey2 * Ix) +
            (2*ex2.transpose() * K * ey2 * Qxy) +
            (exy.transpose() * K * exy * Qxy) +
            (ex2.transpose() * K * ex2 * Qy) +
            (ey2.transpose() * K * ey2 * Qx)
        )(0);

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
typename CosseratRodWithLinearModesOfCrossSectionalDeformation<N>::EnergyGradientType 
CosseratRodWithLinearModesOfCrossSectionalDeformation<N>::minimizationEnergyGradient(const Vec3r& applied_tip_force) const
{
    Real h = this->_length / (N-1);

    TipPositionGradientType tip_pos_grad = Base::tipPositionGradient();

    EnergyGradientType energy_grad = EnergyGradientType::Zero();

    const typename State::CrossSectionVarVecType a0 = this->_state.a0();
    const typename State::CrossSectionVarVecType ax = this->_state.ax();
    const typename State::CrossSectionVarVecType ay = this->_state.ay();
    const typename State::CrossSectionVarVecType b0 = this->_state.b0();
    const typename State::CrossSectionVarVecType bx = this->_state.bx();
    const typename State::CrossSectionVarVecType by = this->_state.by();
    const typename State::CrossSectionVarVecType c0 = this->_state.c0();
    const typename State::CrossSectionVarVecType cx = this->_state.cx();
    const typename State::CrossSectionVarVecType cy = this->_state.cy();
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
    const Real Qx = this->_cross_section->Qx();
    const Real Qy = this->_cross_section->Qy();
    const Real Qxy = this->_cross_section->Qxy();
    const Real eta = this->_cross_section->torsionalCorrection();

    Mat6r K = this->_K;
    // TODO: how to apply torsional correction?
    // K(5,5) *= eta;  // apply torsional correction

    for (int i = 0; i < N-1; i++)
    {
        // cross-sectional parameters at midpoint between nodes
        const Real a0_mid = 0.5*(a0[i] + a0[i+1]);
        const Real ax_mid = 0.5*(ax[i] + ax[i+1]);
        const Real ay_mid = 0.5*(ay[i] + ay[i+1]);
        const Real b0_mid = 0.5*(b0[i] + b0[i+1]);
        const Real bx_mid = 0.5*(bx[i] + bx[i+1]);
        const Real by_mid = 0.5*(by[i] + by[i+1]);
        const Real c0_mid = 0.5*(c0[i] + c0[i+1]);
        const Real cx_mid = 0.5*(cx[i] + cx[i+1]);
        const Real cy_mid = 0.5*(cy[i] + cy[i+1]);

        // get approximate arc-length derivatives of a, b, c
        const Real a0_prime = (a0[i+1] - a0[i]) / h;
        const Real ax_prime = (ax[i+1] - ax[i]) / h;
        const Real ay_prime = (ay[i+1] - ay[i]) / h;
        const Real b0_prime = (b0[i+1] - b0[i]) / h;
        const Real bx_prime = (bx[i+1] - bx[i]) / h;
        const Real by_prime = (by[i+1] - by[i]) / h;
        const Real c0_prime = (c0[i+1] - c0[i]) / h;
        const Real cx_prime = (cx[i+1] - cx[i]) / h;
        const Real cy_prime = (cy[i+1] - cy[i]) / h;

        Real da_dai = 0.5;
        Real db_dbi = 0.5;
        Real dc_dci = 0.5;

        Real dprime_di = -1.0/h;
        Real dprime_diplus1 = 1.0/h;

        Vec6r e0(
            a0_mid - 1,
            b0_mid - 1,
            v3[i] - 1,
            v2[i],
            v1[i],
            2*c0_mid
        );

        Vec6r ex(
            2*ax_mid,
            bx_mid + cy_mid,
            c0_mid*u1[i] - a0_mid*u2[i],
            c0_prime + a0_mid*u3[i],
            a0_prime - c0_mid*u3[i],
            3*cx_mid + ay_mid
        );

        Vec6r ey(
            cx_mid + ay_mid,
            2*by_mid,
            b0_mid*u1[i] - c0_mid*u2[i],
            b0_prime + c0_mid*u3[i],
            c0_prime - b0_mid*u3[i],
            bx_mid + 3*cy_mid
        );

        Vec6r ex2(
            0,
            0,
            cx_mid*u1[i] - ax_mid*u2[i],
            cx_prime + ax_mid*u3[i],
            ax_prime - cx_mid*u3[i],
            0
        );

        Vec6r ey2(
            0,
            0,
            by_mid*u1[i] - cy_mid*u2[i],
            by_prime + cy_mid*u3[i],
            cy_prime - by_mid*u3[i],
            0
        );

        Vec6r exy(
            0,
            0,
            cy_mid*u1[i] - ay_mid*u2[i] + bx_mid*u1[i] - cx_mid*u2[i],
            bx_prime + cy_prime + ay_mid*u3[i] + cx_mid*u3[i],
            ay_prime + cx_prime - bx_mid*u3[i] - cy_mid*u3[i],
            0
        );

        // gradient w.r.t. v1[i]
        Vec6r de0_dv1i(0, 0, 0, 0, 1, 0);
        energy_grad[State::v1Start+i] = 0.5 * h * (
            2*de0_dv1i.transpose() * K * e0 * A +
            2*de0_dv1i.transpose() * K * ex2 * Iy +
            2*de0_dv1i.transpose() * K * ey2 * Ix
        )(0);

        // gradient w.r.t v2[i]
        Vec6r de0_dv2i(0, 0, 0, 1, 0, 0);
        energy_grad[State::v2Start+i] = 0.5 * h * (
            2*de0_dv2i.transpose() * K * e0 * A +
            2*de0_dv2i.transpose() * K * ex2 * Iy +
            2*de0_dv2i.transpose() * K * ey2 * Ix
        )(0);

        // gradient w.r.t v3[i]
        Vec6r de0_dv3i(0, 0, 1, 0, 0, 0);
        energy_grad[State::v3Start+i] = 0.5 * h * (
            2*de0_dv3i.transpose() * K * e0 * A +
            2*de0_dv3i.transpose() * K * ex2 * Iy +
            2*de0_dv3i.transpose() * K * ey2 * Ix
        )(0);

        // gradient w.r.t. u1[i]
        Vec6r dex_du1i(0, 0, c0_mid, 0, 0, 0);
        Vec6r dey_du1i(0, 0, b0_mid, 0, 0, 0);
        Vec6r dex2_du1i(0, 0, cx_mid, 0, 0, 0);
        Vec6r dey2_du1i(0, 0, by_mid, 0, 0, 0);
        Vec6r dexy_du1i(0, 0, cy_mid+bx_mid, 0, 0, 0);
        energy_grad[State::u1Start+i] = 0.5 * h * (
            2*dex_du1i.transpose() * K * ex * Iy +
            2*dey_du1i.transpose() * K * ey * Ix +
            2*e0.transpose() * K * dex2_du1i * Iy +
            2*e0.transpose() * K * dey2_du1i * Ix +
            2*dex2_du1i.transpose() * K * ey2 * Qxy +
            2*ex2.transpose() * K * dey2_du1i * Qxy +
            2*dexy_du1i.transpose() * K * exy * Qxy +
            2*dex2_du1i.transpose() * K * ex2 * Qy +
            2*dey2_du1i.transpose() * K * ey2 * Qx 
        )(0);

        // gradient w.r.t. u2[i]
        Vec6r dex_du2i(0, 0, -a0_mid, 0, 0, 0);
        Vec6r dey_du2i(0, 0, -c0_mid, 0, 0, 0);
        Vec6r dex2_du2i(0, 0, -ax_mid, 0, 0, 0);
        Vec6r dey2_du2i(0, 0, -cy_mid, 0, 0, 0);
        Vec6r dexy_du2i(0, 0, -ay_mid-cx_mid, 0, 0, 0);
        energy_grad[State::u2Start+i] = 0.5 * h * (
            2*dex_du2i.transpose() * K * ex * Iy +
            2*dey_du2i.transpose() * K * ey * Ix +
            2*e0.transpose() * K * dex2_du2i * Iy +
            2*e0.transpose() * K * dey2_du2i * Ix +
            2*dex2_du2i.transpose() * K * ey2 * Qxy +
            2*ex2.transpose() * K * dey2_du2i * Qxy +
            2*dexy_du2i.transpose() * K * exy * Qxy +
            2*dex2_du2i.transpose() * K * ex2 * Qy +
            2*dey2_du2i.transpose() * K * ey2 * Qx 
        )(0);
        
        // gradient w.r.t. u3[i]
        Vec6r dex_du3i(0, 0, 0, a0_mid, -c0_mid, 0);
        Vec6r dey_du3i(0, 0, 0, c0_mid, -b0_mid, 0);
        Vec6r dex2_du3i(0, 0, 0, ax_mid, -cx_mid, 0);
        Vec6r dey2_du3i(0, 0, 0, cy_mid, -by_mid, 0);
        Vec6r dexy_du3i(0, 0, 0, ay_mid+cx_mid, -bx_mid-cy_mid, 0);
        energy_grad[State::u3Start+i] = 0.5 * h * (
            2*dex_du3i.transpose() * K * ex * Iy +
            2*dey_du3i.transpose() * K * ey * Ix +
            2*e0.transpose() * K * dex2_du3i * Iy +
            2*e0.transpose() * K * dey2_du3i * Ix +
            2*dex2_du3i.transpose() * K * ey2 * Qxy +
            2*ex2.transpose() * K * dey2_du3i * Qxy +
            2*dexy_du3i.transpose() * K * exy * Qxy +
            2*dex2_du3i.transpose() * K * ex2 * Qy +
            2*dey2_du3i.transpose() * K * ey2 * Qx 
        )(0);

        // gradient w.r.t. a0[i]
        Vec6r de0_da0i(da_dai, 0, 0, 0, 0, 0);
        Vec6r dex_da0i(0, 0, -u2[i]*da_dai, u3[i]*da_dai, dprime_di, 0);
        energy_grad[State::a0Start+i] += 0.5 * h * (
            2*de0_da0i.transpose() * K * e0 * A + 
            2*dex_da0i.transpose() * K * ex * Iy +
            2*de0_da0i.transpose() * K * ex2 * Iy +
            2*de0_da0i.transpose() * K * ey2 * Ix
        )(0);
        // gradient w.r.t. a0[i+1]
        Vec6r de0_da0iplus1(da_dai, 0, 0, 0, 0, 0);
        Vec6r dex_da0iplus1(0, 0, -u2[i]*da_dai, u3[i]*da_dai, dprime_diplus1, 0);
        energy_grad[State::a0Start+i+1] += 0.5 * h * (
            2*de0_da0iplus1.transpose() * K * e0 * A + 
            2*dex_da0iplus1.transpose() * K * ex * Iy +
            2*de0_da0iplus1.transpose() * K * ex2 * Iy +
            2*de0_da0iplus1.transpose() * K * ey2 * Ix
        )(0);

        // gradient w.r.t. ax[i]
        Vec6r dex_daxi(2*da_dai, 0, 0, 0, 0, 0);
        Vec6r dex2_daxi(0, 0, -u2[i]*da_dai, u3[i]*da_dai, 0, dprime_di);
        energy_grad[State::axStart+i] += 0.5 * h * (
            2*dex_daxi.transpose() * K * ex * Iy +
            2*e0.transpose() * K * dex2_daxi * Iy +
            2*dex2_daxi.transpose() * K * ey2 * Qxy +
            2*dex2_daxi.transpose() * K * ex2 * Qy
        )(0);
        // gradient w.r.t. ax[i+1]
        Vec6r dex_daxiplus1(2*da_dai, 0, 0, 0, 0, 0);
        Vec6r dex2_daxiplus1(0, 0, -u2[i]*da_dai, u3[i]*da_dai, 0, dprime_diplus1);
        energy_grad[State::axStart+i+1] += 0.5 * h * (
            2*dex_daxiplus1.transpose() * K * ex * Iy +
            2*e0.transpose() * K * dex2_daxiplus1 * Iy +
            2*dex2_daxiplus1.transpose() * K * ey2 * Qxy +
            2*dex2_daxiplus1.transpose() * K * ex2 * Qy
        )(0);

        // gradient w.r.t. ay[i]
        Vec6r dex_dayi(0, 0, 0, 0, 0, da_dai);
        Vec6r dey_dayi(da_dai, 0, 0, 0, 0, 0);
        Vec6r dexy_dayi(0, 0, -u2[i]*da_dai, u3[i]*da_dai, dprime_di, 0);
        energy_grad[State::ayStart+i] += 0.5 * h * (
            2*dex_dayi.transpose() * K * ex * Iy +
            2*dey_dayi.transpose() * K * ey * Ix +
            2*dexy_dayi.transpose() * K * exy * Qxy
        )(0);

        // gradient w.r.t. ay[i+1]
        Vec6r dex_dayiplus1 = dex_dayi;
        Vec6r dey_dayiplus1 = dey_dayi;
        Vec6r dexy_dayiplus1(0, 0, -u2[i]*da_dai, u3[i]*da_dai, dprime_diplus1, 0);
        energy_grad[State::ayStart+i+1] += 0.5 * h * (
            2*dex_dayiplus1.transpose() * K * ex * Iy +
            2*dey_dayiplus1.transpose() * K * ey * Ix +
            2*dexy_dayiplus1.transpose() * K * exy * Qxy
        )(0);

        // gradient w.r.t. b0[i]
        Vec6r de0_db0i(0, db_dbi, 0, 0, 0, 0);
        Vec6r dey_db0i(0, 0, u1[i]*db_dbi, dprime_di, -u3[i]*db_dbi, 0);
        energy_grad[State::b0Start+i] += 0.5 * h * (
            2*de0_db0i.transpose() * K * e0 * A +
            2*dey_db0i.transpose() * K * ey * Ix +
            2*de0_db0i.transpose() * K * ex2 * Iy + 
            2*de0_db0i.transpose() * K * ey2 * Ix
        )(0);

        // gradient w.r.t. b0[i+1]
        Vec6r de0_db0iplus1(0, db_dbi, 0, 0, 0, 0);
        Vec6r dey_db0iplus1(0, 0, u1[i]*db_dbi, dprime_diplus1, -u3[i]*db_dbi, 0);
        energy_grad[State::b0Start+i+1] += 0.5 * h * (
            2*de0_db0iplus1.transpose() * K * e0 * A +
            2*dey_db0iplus1.transpose() * K * ey * Ix +
            2*de0_db0iplus1.transpose() * K * ex2 * Iy + 
            2*de0_db0iplus1.transpose() * K * ey2 * Ix
        )(0);

        // gradient w.r.t. bx[i]
        Vec6r dex_dbxi(0, db_dbi, 0, 0, 0, 0);
        Vec6r dey_dbxi(0, 0, 0, 0, 0, db_dbi);
        Vec6r dexy_dbxi(0, 0, u1[i]*db_dbi, dprime_di, -u3[i]*db_dbi, 0);
        energy_grad[State::bxStart+i] += 0.5 * h * (
            2*dex_dbxi.transpose() * K * ex * Iy +
            2*dey_dbxi.transpose() * K * ey * Ix + 
            2*dexy_dbxi.transpose() * K * exy * Qxy
        )(0);

        // gradient w.r.t. bx[i+1]
        Vec6r dex_dbxiplus1(0, db_dbi, 0, 0, 0, 0);
        Vec6r dey_dbxiplus1(0, 0, 0, 0, 0, db_dbi);
        Vec6r dexy_dbxiplus1(0, 0, u1[i]*db_dbi, dprime_diplus1, -u3[i]*db_dbi, 0);
        energy_grad[State::bxStart+i+1] += 0.5 * h * (
            2*dex_dbxiplus1.transpose() * K * ex * Iy +
            2*dey_dbxiplus1.transpose() * K * ey * Ix + 
            2*dexy_dbxiplus1.transpose() * K * exy * Qxy
        )(0);

        // gradient w.r.t. by[i]
        Vec6r dey_dbyi(0, 2*db_dbi, 0, 0, 0, 0);
        Vec6r dey2_dbyi(0, 0, u1[i]*db_dbi, dprime_di, -u3[i]*db_dbi, 0);
        energy_grad[State::byStart+i] += 0.5 * h * (
            2*dey_dbyi.transpose() * K * ey * Ix +
            2*e0.transpose() * K * dey2_dbyi * Ix + 
            2*ex2.transpose() * K * dey2_dbyi * Qxy + 
            2*dey2_dbyi.transpose() * K * ey2 * Qx
        )(0);

        // gradient w.r.t. by[i+1]
        Vec6r dey_dbyiplus1(0, 2*db_dbi, 0, 0, 0, 0);
        Vec6r dey2_dbyiplus1(0, 0, u1[i]*db_dbi, dprime_diplus1, -u3[i]*db_dbi, 0);
        energy_grad[State::byStart+i+1] += 0.5 * h * (
            2*dey_dbyiplus1.transpose() * K * ey * Ix +
            2*e0.transpose() * K * dey2_dbyiplus1 * Ix + 
            2*ex2.transpose() * K * dey2_dbyiplus1 * Qxy + 
            2*dey2_dbyiplus1.transpose() * K * ey2 * Qx
        )(0);

        // gradient w.r.t. c0[i]
        Vec6r de0_dc0i(0, 0, 0, 0, 0, 2*dc_dci);
        Vec6r dex_dc0i(0, 0, u1[i]*dc_dci, dprime_di, -u3[i]*dc_dci, 0);
        Vec6r dey_dc0i(0, 0, -u2[i]*dc_dci, u3[i]*dc_dci, dprime_di, 0);
        energy_grad[State::c0Start+i] += 0.5 * h * (
            2*de0_dc0i.transpose() * K * e0 * A +
            2*dex_dc0i.transpose() * K * ex * Iy + 
            2*dey_dc0i.transpose() * K * ey * Ix + 
            2*de0_dc0i.transpose() * K * ex2 * Iy +
            2*de0_dc0i.transpose() * K * ey2 * Ix
        )(0);

        // gradient w.r.t. c0[i+1]
        Vec6r de0_dc0iplus1(0, 0, 0, 0, 0, 2*dc_dci);
        Vec6r dex_dc0iplus1(0, 0, u1[i]*dc_dci, dprime_diplus1, -u3[i]*dc_dci, 0);
        Vec6r dey_dc0iplus1(0, 0, -u2[i]*dc_dci, u3[i]*dc_dci, dprime_diplus1, 0);
        energy_grad[State::c0Start+i+1] += 0.5 * h * (
            2*de0_dc0iplus1.transpose() * K * e0 * A +
            2*dex_dc0iplus1.transpose() * K * ex * Iy + 
            2*dey_dc0iplus1.transpose() * K * ey * Ix + 
            2*de0_dc0iplus1.transpose() * K * ex2 * Iy +
            2*de0_dc0iplus1.transpose() * K * ey2 * Ix
        )(0);

        // gradient w.r.t. cx[i]
        Vec6r dex_dcxi(0, 0, 0, 0, 0, 3*dc_dci);
        Vec6r dey_dcxi(dc_dci, 0, 0, 0, 0, 0);
        Vec6r dex2_dcxi(0, 0, u1[i]*dc_dci, dprime_di, -u3[i]*dc_dci, 0);
        Vec6r dexy_dcxi(0, 0, -u2[i]*dc_dci, u3[i]*dc_dci, dprime_di, 0);
        energy_grad[State::cxStart+i] += 0.5 * h * (
            2*dex_dcxi.transpose() * K * ex * Iy +
            2*dey_dcxi.transpose() * K * ey * Ix + 
            2*e0.transpose() * K * dex2_dcxi * Iy +
            2*dex2_dcxi.transpose() * K * ey2 * Qxy +
            2*dex2_dcxi.transpose() * K * ex2 * Qy + 
            2*dexy_dcxi.transpose() * K * exy * Qxy
        )(0);

        // gradient w.r.t cx[i+1]
        Vec6r dex_dcxiplus1(0, 0, 0, 0, 0, 3*dc_dci);
        Vec6r dey_dcxiplus1(dc_dci, 0, 0, 0, 0, 0);
        Vec6r dex2_dcxiplus1(0, 0, u1[i]*dc_dci, dprime_diplus1, -u3[i]*dc_dci, 0);
        Vec6r dexy_dcxiplus1(0, 0, -u2[i]*dc_dci, u3[i]*dc_dci, dprime_diplus1, 0);
        energy_grad[State::cxStart+i+1] += 0.5 * h * (
            2*dex_dcxiplus1.transpose() * K * ex * Iy +
            2*dey_dcxiplus1.transpose() * K * ey * Ix + 
            2*e0.transpose() * K * dex2_dcxiplus1 * Iy +
            2*dex2_dcxiplus1.transpose() * K * ey2 * Qxy +
            2*dex2_dcxiplus1.transpose() * K * ex2 * Qy + 
            2*dexy_dcxiplus1.transpose() * K * exy * Qxy
        )(0);

        // gradient w.r.t cy[i]
        Vec6r dex_dcyi(0, dc_dci, 0, 0, 0, 0);
        Vec6r dey_dcyi(0, 0, 0, 0, 0, 3*dc_dci);
        Vec6r dey2_dcyi(0, 0, -u2[i]*dc_dci, u3[i]*dc_dci, dprime_di, 0);
        Vec6r dexy_dcyi(0, 0, u1[i]*dc_dci, dprime_di, -u3[i]*dc_dci, 0);
        energy_grad[State::cyStart+i] += 0.5 * h * (
            2*dex_dcyi.transpose() * K * ex * Iy + 
            2*dey_dcyi.transpose() * K * ey * Ix +
            2*e0.transpose() * K * dey2_dcyi * Ix + 
            2*ex2.transpose() * K * dey2_dcyi * Qxy +
            2*dey2_dcyi.transpose() * K * ey2 * Qx +
            2*dexy_dcyi.transpose() * K * exy * Qxy
        )(0);

        // gradient w.r.t cy[i+1]
        Vec6r dex_dcyiplus1(0, dc_dci, 0, 0, 0, 0);
        Vec6r dey_dcyiplus1(0, 0, 0, 0, 0, 3*dc_dci);
        Vec6r dey2_dcyiplus1(0, 0, -u2[i]*dc_dci, u3[i]*dc_dci, dprime_diplus1, 0);
        Vec6r dexy_dcyiplus1(0, 0, u1[i]*dc_dci, dprime_diplus1, -u3[i]*dc_dci, 0);
        energy_grad[State::cyStart+i+1] += 0.5 * h * (
            2*dex_dcyiplus1.transpose() * K * ex * Iy + 
            2*dey_dcyiplus1.transpose() * K * ey * Ix +
            2*e0.transpose() * K * dey2_dcyiplus1 * Ix + 
            2*ex2.transpose() * K * dey2_dcyiplus1 * Qxy +
            2*dey2_dcyiplus1.transpose() * K * ey2 * Qx +
            2*dexy_dcyiplus1.transpose() * K * exy * Qxy
        )(0);

    }

    if (this->_constrain_base)
    {
        energy_grad[State::a0Start] = 0;
        energy_grad[State::b0Start] = 0;
        energy_grad[State::c0Start] = 0;
        energy_grad[State::axStart] = 0;
        energy_grad[State::bxStart] = 0;
        energy_grad[State::cxStart] = 0;
        energy_grad[State::ayStart] = 0;
        energy_grad[State::byStart] = 0;
        energy_grad[State::cyStart] = 0;
    }

    if (this->_constrain_tip)
    {
        energy_grad[State::a0Start+N-1] = 0;
        energy_grad[State::b0Start+N-1] = 0;
        energy_grad[State::c0Start+N-1] = 0;
        energy_grad[State::axStart+N-1] = 0;
        energy_grad[State::bxStart+N-1] = 0;
        energy_grad[State::cxStart+N-1] = 0;
        energy_grad[State::ayStart+N-1] = 0;
        energy_grad[State::byStart+N-1] = 0;
        energy_grad[State::cyStart+N-1] = 0;
    }
    

    // subtract gradient w.r.t tip position
    energy_grad -= applied_tip_force.transpose() * tip_pos_grad;

    return energy_grad;
}