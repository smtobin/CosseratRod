#include "../alglib-cpp/src/optimization.h"

void nlcfunc1_jac(const alglib::real_1d_array& x, alglib::real_1d_array& fi, alglib::real_2d_array& jac, void* ptr)
{
    fi[0] = -x[0] + x[1];
    fi[1] = x[0]*x[0] + x[1]*x[1] - 1.0;
    jac[0][0] = -1.0;
    jac[0][1] = 1.0;
    jac[1][0] = 2*x[0];
    jac[1][1] = 2*x[1];
}

int main()
{
    try
    {
        alglib::real_1d_array x0 = "[0,0]";
        alglib::real_1d_array s = "[1,1]";
        double epsx = 0.000001;
        alglib::ae_int_t maxits = 0;
        alglib::minnlcstate state;

        // create optimizer object
        alglib::minnlccreate(2, x0, state);
        alglib::minnlcsetcond(state, epsx, maxits);
        alglib::minnlcsetscale(state, s);
        alglib::minnlcsetalgosqp(state);

        // specify bounds
        alglib::real_1d_array bndl = "[0,0]";
        alglib::real_1d_array bndu = "[+inf,+inf]";
        alglib::real_1d_array nl = "[-inf]";
        alglib::real_1d_array nu = "[0]";

        alglib::minnlcsetbc(state, bndl, bndu);
        alglib::minnlcsetnlc2(state, nl, nu);

        // optimize
        alglib::minnlcreport rep;
        alglib::real_1d_array x1;
        alglib::minnlcoptimize(state, nlcfunc1_jac);
        alglib::minnlcresults(state, x1, rep);

        std::cout << "Final state: " << x1.tostring(2).c_str() << std::endl;
    }
    catch(alglib::ap_error alglib_exception)
    {
        std::cerr << alglib_exception.msg.c_str() << '\n';
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}
