#include "common.hpp"
#include "math.hpp"

int main()
{

    Vec6r se3_vec(0.1, 0.3, 0.2, 1.5, 0, 2.0);
    Mat4r T_orig = Math::Exp_se3(se3_vec);

    Vec6r se3_delta(0.001, 0.005, 0.001, 0.0, 0.0, 0.0);
    Mat4r T_delta = Math::Exp_se3(se3_delta);

    Mat4r T_new = T_orig * T_delta;
    // Mat4r T_new = T_delta * T_orig;


    Mat6r jac = Math::ExpMap_Jacobian_SE3(se3_vec);
    Vec6r se3_vec_new = se3_vec + jac * se3_delta;
    Mat4r T_new_jac = Math::Exp_se3(se3_vec_new);

    std::cout << "T orig:\n" << T_orig << std::endl;
    std::cout << "T new:\n" << T_new << std::endl;
    std::cout << "T new jac:\n" << T_new_jac << std::endl;

    std::cout << "================================" << std::endl;

    Vec3r so3_vec(0.1, 0.3, 0.2);
    Mat3r R_orig = Math::Exp_so3(so3_vec);

    Vec3r so3_delta(0.001, 0.005, 0.001);
    Mat3r R_delta = Math::Exp_so3(so3_delta);

    Mat3r R_new = R_orig * R_delta;

    Mat3r jac_so3 = Math::ExpMap_Jacobian_SO3(so3_vec);
    Vec3r so3_vec_new = so3_vec + jac_so3 * so3_delta;
    Mat3r R_new_jac = Math::Exp_so3(so3_vec_new);

    std::cout << "R orig:\n" << R_orig << std::endl;
    std::cout << "R new:\n" << R_new << std::endl;
    std::cout << "R new jac:\n" << R_new_jac << std::endl;

    return EXIT_SUCCESS;
}