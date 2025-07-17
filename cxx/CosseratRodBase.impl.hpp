#ifndef __COSSERAT_ROD_BASE_IMPL_HPP
#define __COSSERAT_ROD_BASE_IMPL_HPP

template <int N, typename State>
Vec3r CosseratRod_Base<N, State>::tipPosition(  Real h,
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

template<int N, typename State>
Vec3r CosseratRod_Base<N, State>::tipPosition() const
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

template <int N, typename State>
typename CosseratRod_Base<N, State>::TipPositionGradientType CosseratRod_Base<N, State>::tipPositionGradient() const
{
    Real h = _length / (N-1);

    TipPositionGradientType grad = TipPositionGradientType::Zero();

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


#endif // __COSSERAT_ROD_BASE_IMPL_HPP