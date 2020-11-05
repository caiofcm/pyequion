import os
import pyequion
# import daetools_cm
from matplotlib import pyplot as plt

sys_reactions_simpler = [
    {
        "H2O": -1.0,
        "OH-": 1.0,
        "H+": 1.0,
        "log_K25": "",
        "log_K_coefs": [
            293.29227,
            0.1360833,
            -10576.913,
            -123.73158,
            0.0,
            -6.996455e-05,
        ],
        "deltah": "",
        "type": "rev",
    },
    {
        "CO3--": -1.0,
        "H+": -1.0,
        "HCO3-": 1.0,
        "log_K25": 10.329,
        "log_K_coefs": [107.8871, 0.03252849, -5151.79, -38.92561, 563713.9],
        "deltah": -3.561,
        "type": "rev",
    },
    {
        "HCO3-": 1.0,
        "H+": 1.0,
        "CO2": -1.0,
        "H2O": -1.0,
        "log_K25": "",
        "deltah": -5.738,
        "log_K_coefs": [-356.3094, -0.06091964, 21834.37, 126.8339, -1684915],
        "type": "rev",
    },
    {
        "Ca++": -1.0,
        "CO3--": -1.0,
        "CaCO3": 1.0,
        "log_K25": 3.224,
        "log_K_coefs": [-1228.732, -0.29944, 35512.75, 485.818],
        "deltah": 3.545,
        "type": "rev",
    },
    {
        "Ca++": -1.0,
        "HCO3-": -1.0,
        "CaHCO3+": 1.0,
        "log_K25": "",
        "log_K_coefs": [1209.120, 0.31294, -34765.05, -478.782],
        "deltah": -0.871,
        "type": "rev",
    },
    {
        "Na+": -1.0,
        "HCO3-": -1.0,
        "NaHCO3": 1.0,
        "log_K25": -0.25,
        "log_K_coefs": "",
        "deltah": -1.0,
        "type": "rev",
    },
    {
        "Na+": -2.0,
        "CO3--": -1.0,
        "Na2CO3": 1.0,
        "log_K25": 0.672,
        "log_K_coefs": "",
        "type": "rev",
    },
    # {
    #     "Ca++": -1.0,
    #     "H2O": -1.0,
    #     "CaOH+": 1.0,
    #     "H+": 1.0,
    #     "log_K25": -12.78,
    #     "log_K_coefs": "",
    #     "deltah": "",
    #     "type": "rev"
    # },
]


def main_create_nahco3_cacl2_closed_sys():
    sys_eq = get_caco3_nahco3_equilibrium()

    pyequion.display_reactions(sys_eq)

    SAVE_RES = True
    if SAVE_RES:
        pyequion.save_res_to_file(
            sys_eq, "./res_nahco3_cacl2_T_2.py", "res", numbafy=False
        )
    return sys_eq


def get_caco3_nahco3_equilibrium():
    feed_compounds = ["Na+", "HCO3-", "Ca++", "Cl-"]
    fixed_elements = ["Cl-"]
    # element_mass_balance = ['Na', 'Ca']
    # closing_equation_type = pyequion.ClosingEquationType.CARBON_TOTAL

    sys_eq = pyequion.create_equilibrium(
        feed_compounds,
        fixed_elements=fixed_elements,
        # possible_aqueous_reactions_in=sys_reactions_simpler,
    )
    return sys_eq


def gen_plot_for_report():
    file_csv = "out_dyn_calcite_precip_2.csv"
    basedir = os.path.dirname(__file__)
    f_path = os.path.join(basedir, file_csv)
    df_out = daetools_cm.load_csv_daeplotter_many_vars(f_path)
    print(df_out.info())
    recolumns = ["pH", "S", "massCrystConc", "L10"]
    df_out.columns = recolumns

    # plt.figure()
    fig, axs = plt.subplots(
        4, sharex=True, sharey=False, gridspec_kw={"hspace": 0}
    )
    axs[0].plot(df_out.index, df_out["pH"], "k", lw=2)
    axs[0].set_ylabel("pH", fontsize=12)
    axs[1].plot(df_out.index, df_out["S"], "k", lw=2)
    axs[1].set_ylabel("S", fontsize=12)
    mCrystg_mL = df_out["massCrystConc"] * 1e6  # kg/kgW -> g/mL
    axs[2].plot(df_out.index, mCrystg_mL, "k", lw=2)
    axs[2].set_ylabel("$C_{cryst}$ [g/mL]", fontsize=12)
    L10_um = df_out["L10"] * 1e6  # m -> um
    axs[3].plot(df_out.index, L10_um, "k", lw=2)
    axs[3].set_ylabel("$L_{10}$ [$\\mu$m]", fontsize=12)
    axs[3].set_xlabel("time [s]")

    df_out.to_csv("data-figure-5.csv")
    plt.savefig(
        os.path.join(basedir, "dyn_sim_with_pyequion.pdf"),
        bbox_extra_artists=None,
        bbox_inches="tight",
    )

    plt.show()

    return


def for_report_code():

    sys_eq = pyequion.create_equilibrium(
        ["Na+", "HCO3-", "Ca++", "Cl-"],
        fixed_elements=["Cl-"],
        possible_aqueous_reactions_in=sys_reactions_simpler,
    )
    pyequion.save_res_to_file(sys_eq, "./eq_nahco3_caco3.py", "res")
    return


if __name__ == "__main__":
    # main_create_nahco3_cacl2_closed_sys()
    gen_plot_for_report()
