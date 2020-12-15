#define OPTIM_ENABLE_ARMA_WRAPPERS
#include <iostream>
#include "optim.hpp"

#ifdef __cplusplus
extern "C" {
#endif
#include "calc_cnv_res_equilibrium_NaHCO3_CaCl2.h"
#ifdef __cplusplus
}
#endif
#include "eq_exported_constanst.h"

using namespace std;


arma::vec zeros_test_objfn_1(const arma::vec& vals_inp, void* opt_data)
{
    double x_1 = vals_inp(0);
    double x_2 = vals_inp(1);

    arma::vec ret(2);

    ret(0) = std::exp(-std::exp(-(x_1+x_2))) - x_2*(1 + std::pow(x_1,2));
    ret(1) = x_1*std::cos(x_2) + x_2*std::sin(x_1) - 0.5;
    //
    return ret;
}

arma::mat zeros_test_jacob_1(const arma::vec& vals_inp, void* opt_data)
{
    double x_1 = vals_inp(0);
    double x_2 = vals_inp(1);

    arma::mat ret(2,2);

    ret(0,0) = std::exp(-std::exp(-(x_1+x_2))-(x_1+x_2)) - 2*x_1*x_1;
    ret(0,1) = std::exp(-std::exp(-(x_1+x_2))-(x_1+x_2)) - x_1*x_1 - 1.0;
    ret(1,0) = std::cos(x_2) + x_2*std::cos(x_1);
    ret(1,1) = -x_1*std::sin(x_2) + std::cos(x_1);
    //
    return ret;
}


arma::vec residuals_wrapper_eq_sys(const arma::vec& vals_inp, void* opt_data)
{
    arma::vec ret(15);
    double concs[2] = {10e-3, 5e-3};
    double TK = 25.0+273.15;
    calc_cnv_res_equilibrium_NaHCO3_CaCl2(TK, concs, (double*)vals_inp.memptr(), ret.memptr());

    arma::cout << "During solution residuals :\n" << ret << arma::endl;
    // arma::cout << "X POST :\n" << vals_inp << arma::endl;
    // getchar();

    return ret;
}

arma::mat jacobian_wrapper_eq_sys(const arma::vec& vals_inp, void* opt_data)
{
    arma::mat ret_J(15,15);
    double concs[2] = {10e-3, 5e-3};
    double TK = 25.0+273.15;
    calc_jac(TK, (double*)vals_inp.memptr(), ret_J.memptr());

    arma::cout << "During solution JACOBIAN :\n" << ret_J.t() << arma::endl;
    // arma::cout << "X POST :\n" << vals_inp << arma::endl;
    // getchar();

    return ret_J.t();
}


/*
Reference sample codde from optim repository
*/
int main_ref()
{
    //
    // Broyden Derivative-free Method of Li and Fukushima (2000)


    arma::vec x_1 = arma::zeros(2,1);

    bool success_1 = optim::broyden_df(x_1,zeros_test_objfn_1,nullptr);

    if (success_1) {
        std::cout << "broyden_df: test_1 completed successfully." << std::endl;
    } else {
        std::cout << "broyden_df: test_1 completed unsuccessfully." << std::endl;
    }

    arma::cout << "broyden_df: solution to test_1:\n" << x_1 << arma::endl;

    //
    // Derivative-free Method of Li and Fukushima (2000) using the jacobian

    x_1 = arma::zeros(2,1);

    success_1 = optim::broyden_df(x_1,zeros_test_objfn_1,nullptr,zeros_test_jacob_1,nullptr);

    if (success_1) {
        std::cout << "broyden_df w jacobian: test_1 completed successfully." << std::endl;
    } else {
        std::cout << "broyden_df w jacobian: test_1 completed unsuccessfully." << std::endl;
    }

    arma::cout << "broyden_df w jacobian: solution to test_1:\n" << x_1 << arma::endl;

    //
    // standard Broyden method

    x_1 = arma::zeros(2,1);

    success_1 = optim::broyden(x_1,zeros_test_objfn_1,nullptr);

    if (success_1) {
        std::cout << "broyden: test_1 completed successfully." << std::endl;
    } else {
        std::cout << "broyden: test_1 completed unsuccessfully." << std::endl;
    }

    arma::cout << "broyden: solution to test_1:\n" << x_1 << arma::endl;

    //
    // standard Broyden method using the jacobian

    x_1 = arma::zeros(2,1);

    success_1 = optim::broyden(x_1,zeros_test_objfn_1,nullptr,zeros_test_jacob_1,nullptr);

    if (success_1) {
        std::cout << "broyden w jacobian: test_1 completed successfully." << std::endl;
    } else {
        std::cout << "broyden w jacobian: test_1 completed unsuccessfully." << std::endl;
    }

    arma::cout << "broyden w jacobian: solution to test_1:\n" << x_1 << arma::endl;

    return 0;
}



int main()
{
    //
    // Broyden Derivative-free Method of Li and Fukushima (2000)
    const int N = 15;
    // double state_vals[15] = {-4.41410428, -5.986602  , -7.88445742, -2.00194266, -8.30714557,
    double state_vals[15] = {-5.4141042, -5.986602  , -7.88445742, -2.00194266, -8.30714557,
       -2.03389701, -4.23809309, -5.21713241, -7.94810423, -3.68749322,
       -2.34133052, -7.36356081, -3.85258587, -3.51907119, -2.0};


    // arma::vec x_1 = arma::zeros(2,1);
    arma::vec x_1(state_vals, N);
    x_1.fill(-3);

    x_1.print("x_guess: ");

    // bool success_1 = optim::broyden(x_1, residuals_wrapper_eq_sys, nullptr);
    bool success_1 = optim::broyden(x_1, residuals_wrapper_eq_sys, nullptr, jacobian_wrapper_eq_sys, nullptr);

    std::cout << "Success CODE = " << success_1 << std::endl;

    if (success_1) {
        std::cout << "broyden_df: test_1 completed successfully." << std::endl;
    } else {
        std::cout << "broyden_df: test_1 completed unsuccessfully." << std::endl;
    }

    arma::cout << "broyden_df: solution to test_1:\n" << x_1 << arma::endl;
    return 0;
}
