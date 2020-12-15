// RETURN HERE
/*
Use this template and your code printer to create a file called run.c in the working directory.

To compile the code there are several options. The first is gcc (the GNU C Compiler). If you have Linux, Mac, or Windows (w/ mingw installed) you can use the Jupyter notebook ! command to send your command to the terminal. For example:

!gcc run.c -lm -o run
This will compile run.c, link against the C math library with -lm and output, -o, to a file run (Mac/Linux) or run.exe (Windows).
*/

#include <math.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif
#include "calc_cnv_res_equilibrium_NaHCO3_CaCl2.h"
#ifdef __cplusplus
}
#endif
#include "eq_exported_constanst.h"

using namespace std;

int main() {

    // initialize the state vector with some values
    double state_vals[15] = {-4.41410428, -5.986602  , -7.88445742, -2.00194266, -8.30714557,
       -2.03389701, -4.23809309, -5.21713241, -7.94810423, -3.68749322,
       -2.34133052, -7.36356081, -3.85258587, -3.51907119, -2.0};
    // create "empty" 1D arrays to hold the results of the computation
    double rhs_result[15];
    double jac_result[225];
    double concs[2];
    double TC = 25.0+273.15;

    // cal the imported
    calc_cnv_res_equilibrium_NaHCO3_CaCl2(TC, concs, state_vals, rhs_result);

    // call the saturation using the solution x
    double SI_calcite = calc_phase_SI_Calcite(TC, state_vals);

    // print the computed values to the terminal
    // int i;

    // printf("The right hand side of the equations evaluates to:\n");
    // for (i=0; i < 14; i++) {
    //     printf("%lf\n", rhs_result[i]);
    // }

    printf("Index check = %d :\n", IDX_SPECIES['CO2']);

    printf("Calcite saturation = %lf :\n", SI_calcite);


    // printf("\nThe Jacobian evaluates to:\n");
    // for (i=0; i < 196; i++) {
    //     printf("%lf\n", jac_result[i]);
    // }

    return 0;
}
