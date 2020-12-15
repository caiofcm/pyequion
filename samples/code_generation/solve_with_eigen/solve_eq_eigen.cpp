#include <iostream>
#include <chrono>
#include <Eigen/Dense>
#include <Eigen/src/Core/util/DisableStupidWarnings.h>

#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

using namespace std;
using namespace Eigen;

#ifdef __cplusplus
extern "C"
{
#endif
#include "calc_cnv_res_equilibrium_NaHCO3_CaCl2.h"
#ifdef __cplusplus
}
#endif
#include "eq_exported_constanst.h"

// Generic functor
template <typename _Scalar, int NX = Eigen::Dynamic, int NY = Eigen::Dynamic>
struct Functor
{
    typedef _Scalar Scalar;
    enum
    {
        InputsAtCompileTime = NX,
        ValuesAtCompileTime = NY
    };
    typedef Eigen::Matrix<Scalar, InputsAtCompileTime, 1> InputType;
    typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, 1> ValueType;
    typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime> JacobianType;

    int m_inputs, m_values;

    Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
    Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

    int inputs() const { return m_inputs; }
    int values() const { return m_values; }
};

struct my_functor : Functor<double>
{
    my_functor(void) : Functor<double>(2, 2) {}
    int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const
    {
        // Implement y = 10*(x0+3)^2 + (x1-5)^2
        fvec(0) = 10.0 * pow(x(0) + 3.0, 2) + pow(x(1) - 5.0, 2);
        fvec(1) = 0;

        return 0;
    }
};

void testStackOverflowSampleOpt()
{

    int8_t N = 15;
    double state_vals[15] = {-4.41410428, -5.986602, -7.88445742, -2.00194266, -8.30714557,
                             -2.03389701, -4.23809309, -5.21713241, -7.94810423, -3.68749322,
                             -2.34133052, -7.36356081, -3.85258587, -3.51907119, -2.001};

    Eigen::VectorXd x;
    x << -4.41410428, -5.986602, -7.88445742, -2.00194266, -8.30714557,
        -2.03389701, -4.23809309, -5.21713241, -7.94810423, -3.68749322,
        -2.34133052, -7.36356081, -3.85258587, -3.51907119, -2.001;
    // x(0) = 2.0;
    // x(1) = 3.0;
    std::cout << "x: " << x << std::endl;

    my_functor functor;
    Eigen::NumericalDiff<my_functor> numDiff(functor);
    Eigen::LevenbergMarquardt<Eigen::NumericalDiff<my_functor>, double> lm(numDiff);
    lm.parameters.maxfev = 2000;
    lm.parameters.xtol = 1.0e-10;
    std::cout << lm.parameters.maxfev << std::endl;

    int ret = lm.minimize(x);
    std::cout << lm.iter << std::endl;
    std::cout << ret << std::endl;

    std::cout << "x that minimizes the function: " << x << std::endl;

    std::cout << "press [ENTER] to continue " << std::endl;
    std::cin.get();
    return;
}

struct hybrd_functor : Functor<double>
{
    hybrd_functor(int N, const VectorXd &concs_in, double TK_in) :
        Functor<double>(N, N),
        concs(concs_in),
        TK(TK_in) {}

    int operator()(const VectorXd &x, VectorXd &fvec) const
    {
        double temp, temp1, temp2;
        const VectorXd::Index n = x.size();

        assert(fvec.size() == n);
        double *x_ptr = (double *)x.data();
        double *fvec_ptr = fvec.data();
        calc_cnv_res_equilibrium_NaHCO3_CaCl2(TK, (double *)concs.data(), x_ptr, fvec_ptr);

        return 0;
    }
    int df(const VectorXd &x, MatrixXd &fjac)
    {
        const VectorXd::Index n = x.size();
        assert(fjac.rows() == n);
        assert(fjac.cols() == n);

        double *x_ptr = (double *)x.data();
        double *J_ptr = fjac.data();
        calc_jac(TK, x_ptr, J_ptr);
        fjac.transposeInPlace();
        return 0;
    }

    private:
        VectorXd concs;
        double TK;
};

std::chrono::duration<double> getElapsedTime(const std::chrono::time_point<std::chrono::system_clock> &start)
{
    std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    return elapsed_seconds;
}

void testHybrj1()
{
    const int n = 15;
    int info;
    VectorXd x(n);
    // x << -1, -5.986602  , -7.88445742, -2.00194266, -8.30714557,
    //     -2.03389701, -4.23809309, -5.21713241, -7.94810423, -3.68749322,
    //     -2.34133052, -7.36356081, -3.85258587, -3.51907119, -2.1;
    VectorXd concs(2);
    concs << 10e-3, 5e-3;
    double TK = 25.0 + 273.15;

    /* the following starting values provide a rough fit. */
    x.setConstant(n, -1.);

    std::cout << "Hybr: w/o Jacobian" << std::endl;
    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();

    hybrd_functor functor(n, concs, TK);
    HybridNonLinearSolver<hybrd_functor> solver(functor);
    info = solver.solveNumericalDiff(x);

    std::cout << x << std::endl;
    std::cout << "Solution Code: " << info << std::endl;
    std::cout << "Elapsed: " << getElapsedTime(start).count() << " s" << std::endl;

    //-----------------------------------------------------------------------
    x.setConstant(n, -1.);
    std::cout << "Hybr: w/ Jacobian" << std::endl;
    start = std::chrono::system_clock::now();

    hybrd_functor functorJac(n, concs, TK);
    HybridNonLinearSolver<hybrd_functor> solverJac(functorJac);
    info = solverJac.hybrj1(x);

    std::cout << x << std::endl;
    std::cout << "Solution Code: " << info << std::endl;
    std::cout << "Elapsed: " << getElapsedTime(start).count() << " s" << std::endl;

    //-----------------------------------------------------------------------
    std::cout << "Calculating Calcite Saturation Index:" << info << std::endl;
    double SI_calcite = calc_phase_SI_Calcite(TK, x.data());
    std::cout << "SI Calcite = " << SI_calcite << std::endl;

    VectorXd gammas(n);
    calc_gammas(TK, x.data(), gammas.data());
    std::cout << "Gammas: \n " << gammas << std::endl;
    double pH = - log10(pow(10, x[IDX_SPECIES["H+"]]) * gammas[IDX_SPECIES["H+"]]);
    std::cout << "pH = " << pH << std::endl;

    return;
}

int main(int argc, char *argv[])
{
    testHybrj1();
    return 0;
}
