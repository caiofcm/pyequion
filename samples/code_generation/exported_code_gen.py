# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython


# %%
import pyequion
import sympy


# # %%
sys_eq = pyequion.create_equilibrium(["NaHCO3", "CaCl2"])
pyequion.save_res_to_c_code(
    sys_eq,
    "dummy",
    "calc_cnv_res_equilibrium_NaHCO3_CaCl2",
    include_jac=False,
    # ,fixed_temperature=25.0
)


# %%
# sys_eq = pyequion.create_equilibrium(['NaHCO3', 'CaCl2'])
# pyequion.save_jacobian_of_res_to_c_code(sys_eq, 'dummy', 'calc_cnv_jac_equilibrium_NaHCO3_CaCl2')


# # %%
# get_ipython().run_cell_magic('writefile', 'calc_cnv_res_equilibrium_NaHCO3_CaCl2.pyxbld', 'import numpy\n\n#            module name specified by `%%cython_pyximport` magic\n#            |        just `modname + ".pyx"`\n#            |        |\ndef make_ext(modname, pyxfilename):\n    from setuptools.extension import Extension\n    return Extension(modname,\n                     sources=[pyxfilename, \'calc_cnv_res_equilibrium_NaHCO3_CaCl2.c\'],\n                     include_dirs=[\'.\', numpy.get_include()])')


# # %%
# get_ipython().run_cell_magic('cython_pyximport', 'cy_odes', 'import numpy as np\ncimport numpy as cnp # cimport gives us access to NumPy\'s C API\n\n# here we just replicate the function signature from the header\ncdef extern from "calc_cnv_res_equilibrium_NaHCO3_CaCl2.h":\n    void calc_cnv_res_equilibrium_NaHCO3_CaCl2(double T, double *concs, double *x, double *res)\n\n# here is the "wrapper" signature that conforms to the odeint interface\ndef cy_calc_cnv_res_equilibrium_NaHCO3_CaCl2(double T, cnp.ndarray[cnp.double_t, ndim=1] concs, cnp.ndarray[cnp.double_t, ndim=1] x):\n    # preallocate our output array\n    cdef cnp.ndarray[cnp.double_t, ndim=1] res = np.empty(x.size, dtype=np.double)\n    # now call the C function\n    calc_cnv_res_equilibrium_NaHCO3_CaCl2(<double> T, <double *> concs.data, <double *> x.data, <double *> res.data)\n    # return the result\n    return dY')


# %%
# sol = pyequion.solve_solution({'NaHCO3': 10, 'CaCl2': 5})
# calc_cnv_res_equilibrium_NaHCO3_CaCl2.pyxbld(25.0, sol.x)
