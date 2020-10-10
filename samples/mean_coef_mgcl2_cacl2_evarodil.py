"""Pyequion sample

- Calculating mean activity coefficients for CaCl2 and MgCl2
"""

# %% IMPORTS
import os

# os.environ['NUMBA_DISABLE_JIT'] = '1'
import pyequion
import numpy as np
from matplotlib import pyplot as plt

# from jac_cacl2_evarodil import jac
# import pykinsol
import json
import time

# class NumpyEncoder(json.JSONEncoder):
#     def default(self, obj): #pylint: disable=method-hidden
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return json.JSONEncoder.default(self, obj)

USE_PGF = True

if USE_PGF:
    import matplotlib

    matplotlib.use("pgf")
    matplotlib.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "text.usetex": True,
            "pgf.rcfonts": False,
        }
    )

# %% SETUP
data_exp_raw_collected = np.array(
    [
        [0.01, 0.7047],
        [0.02, 0.6678],
        [0.03, 0.6328],
        [0.05, 0.5806],
        [0.10, 0.5000],
        [0.30, 0.4590],
        [0.50, 0.4472],
        [0.80, 0.4657],
        [1.00, 0.5091],
        [1.30, 0.5534],
        [1.50, 0.6149],
        [1.80, 0.6944],
        [2.00, 0.7939],
        [2.50, 1.0673],
        [3.00, 1.4763],
    ]
)

data_exp_raw_collected_mgcl2 = np.array(
    [
        [0.01, 0.7107],
        [0.02, 0.6754],
        [0.03, 0.6422],
        [0.05, 0.5930],
        [0.10, 0.5202],
        [0.30, 0.4888],
        [0.50, 0.4887],
        [0.80, 0.5271],
        [1.00, 0.5985],
        [1.20, 0.6715],
        [1.50, 0.7740],
        [1.80, 0.9099],
        [2.00, 1.0846],
        [2.50, 1.5891],
        [3.00, 2.3990],
    ]
)

data_exp_raw_collected_NaCl = np.array(
    [
        [0.200, 0.777],
        [0.200, 0.726],
        [0.200, 0.703],
        [0.400, 0.688],
        [0.600, 0.676],
        [0.800, 0.664],
        [1.00, 0.657],
        [1.20, 0.654],
        [1.40, 0.654],
        [1.60, 0.657],
        [1.80, 0.661],
        [2.00, 0.667],
        [2.20, 0.676],
        [2.40, 0.685],
        [2.60, 0.687],
        [2.80, 0.704],
        [3.00, 0.713],
        [3.20, 0.726],
        [3.40, 0.739],
        [3.60, 0.746],
        [3.80, 0.767],
        [4.00, 0.782],
        [4.20, 0.799],
        [4.40, 0.823],
        [4.60, 0.827],
        [4.80, 0.854],
        [5.00, 0.873],
        [5.20, 0.893],
        [5.40, 0.917],
        [5.60, 0.930],
        [5.80, 0.964],
    ]
)

# %% DO NOT KNOW WHY, BUT JIT COMPILATION IS SLOWER FOR solve_solution, but faster for solve_equilibrium
print("CaCl2 - Check JIT")


def main_check_jit():
    pyequion.jit_compile_functions()

    sys_eq = pyequion.create_equilibrium(
        feed_compounds=["CaCl2"], initial_feed_mass_balance=["Cl-"]
    )
    solution = pyequion.solve_solution(
        {"CaCl2": 1 * 1e3},
        sys_eq,
        activity_model_type="debye",
    )
    print("end first call")

    tstart = time.time()
    solution = pyequion.solve_solution(
        {"CaCl2": 1 * 1e3},
        sys_eq,
        activity_model_type="debye",
    )
    print("Elapsed time = ", time.time() - tstart, solution.pH)

    args = (np.array([1.0]), 25 + 273.15, np.nan)
    tstart = time.time()
    solution = pyequion.solve_equilibrium(sys_eq, args=args)
    print("Elapsed time = ", time.time() - tstart, solution.pH)

    quit()

    tstart = time.time()
    cIn_span = np.geomspace(1e-3, 3.0, 1)
    gamma_mean = np.empty_like(cIn_span)
    for i, c in enumerate(cIn_span):
        solution = pyequion.solve_solution(
            {"CaCl2": c * 1e3},
            sys_eq,
            activity_model_type="debye",
        )
        gamma_mean[i] = pyequion.get_mean_activity_coeff(
            solution, "CaCl2"
        )  # pylint: disable=unsupported-assignment-operation
    print("Elapsed time = ", time.time() - tstart)
    print(gamma_mean)


# %%
def conv_c2I_X1A2(c):
    return 0.5 * (c * 4 + 1 * 2 * c)


def conv_c2I_X1A1(c):
    return 0.5 * (c + c)


def main_cacl2_meancoeff():

    sys_eq = pyequion.create_equilibrium(
        feed_compounds=["CaCl2"], initial_feed_mass_balance=["Cl-"]
    )

    cIn_span = np.geomspace(1e-3, 3.0, 61)
    methods = ["debye", "bromley", "sit", "pitzer"]

    solutions = {
        key: [
            pyequion.solve_solution(
                {"CaCl2": c * 1e3}, sys_eq, activity_model_type=key
            )
            for c in cIn_span
        ]
        for key in methods
    }
    gamma_means = {
        key: np.array(
            [
                pyequion.get_mean_activity_coeff(sol, "CaCl2")
                for sol in solutions[key]
            ]
        )
        for key in methods
    }

    I = {key: np.array([sol.I for sol in solutions[key]]) for key in methods}

    I_exp_cnv = conv_c2I_X1A2(data_exp_raw_collected[:, 0])

    plot_opts = {
        "debye": "-k",
        "bromley": ":k",
        "sit": "-.k",
        "pitzer": "--k",
    }
    plt.figure()
    plt.plot(I_exp_cnv, data_exp_raw_collected[:, 1], "sk", label="exp")
    for key in methods:
        plt.plot(I[key], gamma_means[key], plot_opts[key], label=key)
    plt.legend()

    plt.show()

    return


def cacl2_mgcl2_nacl_mean_activity_coeffs():

    # CaCl2
    sys_eq = pyequion.create_equilibrium(
        feed_compounds=["CaCl2"], initial_feed_mass_balance=["Cl-"]
    )

    cIn_span = np.geomspace(1e-3, 3.0, 51)
    methods = ["debye", "bromley", "sit", "pitzer"]

    solutions = {
        key: [
            pyequion.solve_solution(
                {"CaCl2": c * 1e3}, sys_eq, activity_model_type=key
            )
            for c in cIn_span
        ]
        for key in methods
    }
    gamma_means_CaCl2 = {
        key: np.array(
            [
                pyequion.get_mean_activity_coeff(sol, "CaCl2")
                for sol in solutions[key]
            ]
        )
        for key in methods
    }
    I_CaCl2 = {
        key: np.array([sol.I for sol in solutions[key]]) for key in methods
    }

    # MgCl2
    tag = "MgCl2"
    sys_eq = pyequion.create_equilibrium(
        feed_compounds=[tag], initial_feed_mass_balance=["Cl-"]
    )

    solutions = {
        key: [
            pyequion.solve_solution(
                {tag: c * 1e3}, sys_eq, activity_model_type=key
            )
            for c in cIn_span
        ]
        for key in methods
    }
    gamma_means_MgCl2 = {
        key: np.array(
            [
                pyequion.get_mean_activity_coeff(sol, tag)
                for sol in solutions[key]
            ]
        )
        for key in methods
    }
    I_MgCl2 = {
        key: np.array([sol.I for sol in solutions[key]]) for key in methods
    }

    # NaCl
    cNaCl_span = np.geomspace(1e-3, 6.0, 51)
    tag = "NaCl"
    sys_eq = pyequion.create_equilibrium(
        feed_compounds=[tag], initial_feed_mass_balance=["Cl-"]
    )

    solutions = {
        key: [
            pyequion.solve_solution(
                {tag: c * 1e3}, sys_eq, activity_model_type=key
            )
            for c in cNaCl_span
        ]
        for key in methods
    }
    gamma_means_NaCl = {
        key: np.array(
            [
                pyequion.get_mean_activity_coeff(sol, tag)
                for sol in solutions[key]
            ]
        )
        for key in methods
    }
    I_NaCl = {
        key: np.array([sol.I for sol in solutions[key]]) for key in methods
    }

    I_exp_cnvCaCl2 = conv_c2I_X1A2(data_exp_raw_collected[:, 0])
    I_exp_cnvMgCl2 = conv_c2I_X1A2(data_exp_raw_collected_mgcl2[:, 0])
    I_exp_cnvNaCl = conv_c2I_X1A1(data_exp_raw_collected_NaCl[:, 0])

    fs = 15
    plot_opts = {
        "debye": "-k",
        "bromley": ":k",
        "sit": "-.k",
        "pitzer": "--k",
    }
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 3, 1)
    plt.plot(I_exp_cnvCaCl2, data_exp_raw_collected[:, 1], "sk", label="exp")
    for key in methods:
        plt.plot(
            I_CaCl2[key], gamma_means_CaCl2[key], plot_opts[key], label=key
        )
    plt.xlabel("I [M]", fontsize=12)
    plt.title("$\\gamma_{CaCl_2}$", fontsize=fs)
    # plt.ylabel('$\gamma_{\pm}$')
    plt.legend(fontsize=12)
    plt.subplot(1, 3, 2)
    plt.plot(
        I_exp_cnvMgCl2, data_exp_raw_collected_mgcl2[:, 1], "sk", label="exp"
    )
    plt.xlabel("I [M]", fontsize=fs)
    plt.title("$\\gamma_{MgCl_2}$", fontsize=fs)
    # plt.ylabel('$\gamma_{\pm}$')
    for key in methods:
        plt.plot(
            I_MgCl2[key], gamma_means_MgCl2[key], plot_opts[key], label=key
        )
    plt.legend(fontsize=12)
    plt.subplot(1, 3, 3)
    plt.plot(
        I_exp_cnvNaCl, data_exp_raw_collected_NaCl[:, 1], "sk", label="exp"
    )
    plt.xlabel("I [M]", fontsize=fs)
    plt.title("$\\gamma_{NaCl}$", fontsize=fs)
    # plt.ylabel('$\gamma_{\pm}$')
    for key in methods:
        plt.plot(I_NaCl[key], gamma_means_NaCl[key], plot_opts[key], label=key)
    lgd = plt.legend(fontsize=12)

    if not USE_PGF:
        plt.show()
    else:
        basedir = os.path.dirname(__file__)
        plt.savefig(
            os.path.join(basedir, "single-salt-meancoeffs.pgf"),
            bbox_extra_artists=(lgd,),
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(basedir, "single-salt-meancoeffs.pdf"),
            bbox_extra_artists=(lgd,),
            bbox_inches="tight",
        )

    kw_save = dict(
        CaCl2={
            "exp": {"x": I_exp_cnvCaCl2, "y": data_exp_raw_collected[:, 1]},
            "I_calcs": I_CaCl2,
            "gamma_means": gamma_means_CaCl2,
        },
        MgCl2={
            "exp": {
                "x": I_exp_cnvMgCl2,
                "y": data_exp_raw_collected_mgcl2[:, 1],
            },
            "I_calcs": I_MgCl2,
            "gamma_means": gamma_means_MgCl2,
        },
        NaCl={
            "exp": {
                "x": I_exp_cnvNaCl,
                "y": data_exp_raw_collected_NaCl[:, 1],
            },
            "I_calcs": I_NaCl,
            "gamma_means": gamma_means_NaCl,
        },
    )
    np.savez("single-salt-meancoeffs.npz", **kw_save)
    return


if __name__ == "__main__":
    # main_cacl2_meancoeff()
    cacl2_mgcl2_nacl_mean_activity_coeffs()
# %%
