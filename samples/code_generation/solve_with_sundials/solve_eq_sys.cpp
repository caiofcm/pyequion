/*
A simple example using the KINSOL library to solve a simple 2d ODE, treating it
as a stiff system.
*/

#include <iostream>
#include "kinsol/kinsol.h"             // access to KINSOL func., consts.
#include <nvector/nvector_serial.h>    // access to serial N_Vector
#include <sunlinsol/sunlinsol_spgmr.h> //access to SPGMR SUNLinearSolver
#include <kinsol/kinsol_spils.h>       // access to KINSpils interface
#include <sundials/sundials_dense.h>   // use generic dense solver in precond
#include <sundials/sundials_types.h>   // defs. of realtype, sunindextype
#include <sundials/sundials_math.h>    // contains the macros ABS, SUNSQR, EXP

#ifdef __cplusplus
extern "C"
{
#endif
#include "calc_cnv_res_equilibrium_NaHCO3_CaCl2.h"
#ifdef __cplusplus
}
#endif
#include "eq_exported_constanst.h"

// This macro gives access to the individual components of the data array of an
// N Vector.
#define NV_Ith_S(v, i) (NV_DATA_S(v)[i])

using namespace std;

static int f(N_Vector u, N_Vector u_dot, void *user_data);
static int jtv(N_Vector v, N_Vector Jv, realtype t, N_Vector u, N_Vector fu,
               void *user_data, N_Vector tmp);
static int check_flag(void *flagvalue, const char *funcname, int opt);

int main()
{
    int flag; // For checking if functions have run properly


    // 1. Initialize parallel or multi-threaded environment, if appropriate.
    // ---------------------------------------------------------------------------
    // ---------------------------------------------------------------------------

    // 2. Defining the length of the problem.
    // ---------------------------------------------------------------------------
    sunindextype N = 15;

    double state_vals[15] = {-4.41410428, -5.986602  , -7.88445742, -2.00194266, -8.30714557,
        -2.03389701, -4.23809309, -5.21713241, -7.94810423, -3.68749322,
        -2.34133052, -7.36356081, -3.85258587, -3.51907119, -2.001};

    double concs[2] = {10e-3, 5e-3};
    double TK = 25.0+273.15;
    double rhs_result[15];

    for (size_t i = 0; i < N; i++) {
        // state_vals[i] = -1.0;
    }


    calc_cnv_res_equilibrium_NaHCO3_CaCl2(TK, concs, state_vals, rhs_result);

    // ---------------------------------------------------------------------------

    // 3. Set vector with initial guess.
    // ---------------------------------------------------------------------------
    N_Vector y0; // Problem vector.
    y0 = N_VNew_Serial(N);
    for (size_t i = 0; i < N; i++) {
        NV_Ith_S(y0, i) = state_vals[i];
    }

    if (check_flag((void *)y0, "N_VNew_Serial", 0))
        return (1);

    N_Vector sc; // Scaling vector.
    sc = N_VNew_Serial(N);
    for (size_t i = 0; i < N; i++) {
        NV_Ith_S(sc, i) = 1.0;
    }

    if (check_flag((void *)sc, "N_VNew_Serial", 0))
        return (1);


    // ---------------------------------------------------------------------------

    // 4. Create KINSOL Object.
    // ---------------------------------------------------------------------------
    void *kin_mem = NULL; // Problem dedicated memory.
    kin_mem = KINCreate();
    if (check_flag((void *)kin_mem, "KINCreate", 0))
        return (1);
    // ---------------------------------------------------------------------------

    // 5. Set Optional Inputs.
    // ---------------------------------------------------------------------------
    // ---------------------------------------------------------------------------

    // 6. Allocate Internal Memory .
    // ---------------------------------------------------------------------------
    flag = KINInit(kin_mem, f, y0);
    if (check_flag(&flag, "KINInit", 1))
        return (1);
    // ---------------------------------------------------------------------------

    // 7. Create Matrix Object.
    // ---------------------------------------------------------------------------
    // ---------------------------------------------------------------------------

    // 8. Create Linear Solver Object.
    // ---------------------------------------------------------------------------
    SUNLinearSolver LS;
    // Here we chose one of the possible linear solver modules. SUNSPMR is an
    // iterative solver that is designed to be compatible with any nvector
    // implementation (serial, threaded, parallel,
    // user-supplied)that supports a minimal subset of operations.
    LS = SUNSPGMR(y0, 0, 0);
    if (check_flag((void *)LS, "SUNSPGMR", 0))
        return (1);
    // ---------------------------------------------------------------------------

    // 9. Set linear solver optional inputs.
    // ---------------------------------------------------------------------------
    // ---------------------------------------------------------------------------
    flag = KINSetFuncNormTol(kin_mem, 1e-15);

    // if (check_flag((void *)LS, "SUNSPGMR", 0))
    //     return (1);

    // 10. Attach linear solver module.
    // ---------------------------------------------------------------------------
    // CVSpilsSetLinearSolver is for iterative linear solvers.
    flag = KINSpilsSetLinearSolver(kin_mem, LS);
    if (check_flag(&flag, "KINSpilsSetLinearSolver", 1))
        return 1;
    // ---------------------------------------------------------------------------

    // 11. Set linear solver interface optional inputs.
    // ---------------------------------------------------------------------------
    // Sets the jacobian-times-vector function.
    // flag = CVSpilsSetJacTimes(kin_mem, NULL, jtv);
    // if(check_flag(&flag, "CVSpilsSetJacTimes", 1)) return(1);
    // ---------------------------------------------------------------------------

    // 12. Solve problem,
    // ---------------------------------------------------------------------------

    /* Call KINSol and print output concentration profile */
    flag = KINSol(kin_mem,        /* KINSol memory block */
                  y0,             /* initial guess on input; solution vector */
                  KIN_NONE, /* global strategy choice */
                  sc,             /* scaling vector for the variable cc */
                  sc);            /* scaling vector for function values fval */
    if (check_flag(&flag, "KINSol", 1))
        return (1);

    // Printing output.
    std::cout << "Final Value of y0 vector: \n";
    N_VPrint_Serial(y0);

    // ---------------------------------------------------------------------------

    // 13. Get optional outputs.
    // ---------------------------------------------------------------------------
    // ---------------------------------------------------------------------------

    // 14. Deallocate memory for solution vector.
    // ---------------------------------------------------------------------------
    N_VDestroy(y0);
    // ---------------------------------------------------------------------------

    // 15. Free solver memory.
    // ---------------------------------------------------------------------------
    KINFree(&kin_mem);
    // ---------------------------------------------------------------------------

    // 16. Free linear solver and matrix memory.
    // ---------------------------------------------------------------------------
    SUNLinSolFree(LS);
    // ---------------------------------------------------------------------------

    // 17. Finalize MPI, if used.
    // ---------------------------------------------------------------------------
    // ---------------------------------------------------------------------------

    // return(0);
}

// Simple function that calculates the differential equation.
static int f(N_Vector u, N_Vector f_val, void *user_data)
{

    double concs[2] = {10e-3, 5e-3};
    double TK = 25.0+273.15;

    realtype *udata = N_VGetArrayPointer(u);     // pointer u vector data
    realtype *fdata = N_VGetArrayPointer(f_val); // pointer to udot vector data

    calc_cnv_res_equilibrium_NaHCO3_CaCl2(TK, concs, udata, fdata);

    std::cout << "Residual =" << std::endl;
    std:cout << fdata[0] << std::endl;
    // // N_VGetArrayPointer returns a pointer to the data in the N_Vector class.
    // realtype *udata = N_VGetArrayPointer(u);     // pointer u vector data
    // realtype *fdata = N_VGetArrayPointer(f_val); // pointer to udot vector data

    // fdata[0] = -101.0 * udata[0] - 100.0 * udata[1];
    // fdata[1] = udata[0];

    return (0);
}

// Jacobian function vector routine.
static int jtv(N_Vector v, N_Vector Jv, realtype t, N_Vector u, N_Vector fu,
               void *user_data, N_Vector tmp)
{
    realtype *udata = N_VGetArrayPointer(u);
    realtype *vdata = N_VGetArrayPointer(v);
    realtype *Jvdata = N_VGetArrayPointer(Jv);
    realtype *fudata = N_VGetArrayPointer(fu);

    Jvdata[0] = -101.0 * vdata[0] + -100.0 * vdata[1];
    Jvdata[1] = vdata[0] + 0 * vdata[1];

    fudata[0] = 0;
    fudata[1] = 0;

    return (0);
}

// check_flag function is from the cvDiurnals_ky.c example from the CVODE
// package.
/* Check function return value...
     opt == 0 means SUNDIALS function allocates memory so check if
              returned NULL pointer
     opt == 1 means SUNDIALS function returns a flag so check if
              flag >= 0
     opt == 2 means function allocates memory so check if returned
              NULL pointer */
static int check_flag(void *flagvalue, const char *funcname, int opt)
{
    int *errflag;

    /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
    if (opt == 0 && flagvalue == NULL)
    {
        fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
                funcname);
        return (1);
    }

    /* Check if flag < 0 */
    else if (opt == 1)
    {
        errflag = (int *)flagvalue;
        if (*errflag < 0)
        {
            fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n",
                    funcname, *errflag);
            return (1);
        }
    }

    /* Check if function returned NULL pointer - no memory allocated */
    else if (opt == 2 && flagvalue == NULL)
    {
        fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
                funcname);
        return (1);
    }

    return (0);
}
