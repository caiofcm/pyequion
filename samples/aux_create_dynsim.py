import pyequion


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
            -6.996455e-05
        ],
        "deltah": "",
        "type": "rev"
    },
    {
        "CO3--": -1.0,
        "H+": -1.0,
        "HCO3-": 1.0,
        "log_K25": 10.329,
        "log_K_coefs": [
            107.8871,
            0.03252849,
            -5151.79,
            -38.92561,
            563713.9
        ],
        "deltah": -3.561,
        "type": "rev"
    },
    {
        "HCO3-": 1.0,
        "H+": 1.0,
        "CO2": -1.0,
        "H2O": -1.0,
        "log_K25": "",
        "deltah": -5.738,
        "log_K_coefs": [
            -356.3094,
            -0.06091964,
            21834.37,
            126.8339,
            -1684915
        ],
        "type": "rev"
    },
    {
        "Ca++": -1.0,
        "CO3--": -1.0,
        "CaCO3": 1.0,
        "log_K25": 3.224,
        "log_K_coefs": [
            -1228.732,
            -0.29944,
            35512.75,
            485.818
        ],
        "deltah": 3.545,
        "type": "rev"
    },
    {
        "Ca++": -1.0,
        "HCO3-": -1.0,
        "CaHCO3+": 1.0,
        "log_K25": "",
        "log_K_coefs": [
            1209.120,
            0.31294,
            -34765.05,
            -478.782
        ],
        "deltah": -0.871,
        "type": "rev"
    },
    {
        "Na+": -1.0,
        "HCO3-": -1.0,
        "NaHCO3": 1.0,
        "log_K25": -0.25,
        "log_K_coefs": "",
        "deltah": -1.0,
        "type": "rev"
    },
    {
        "Na+": -2.0,
        "CO3--": -1.0,
        "Na2CO3": 1.0,
        "log_K25": 0.672,
        "log_K_coefs": "",
        "type": "rev"
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

    SAVE_RES = False
    if SAVE_RES:
        pyequion.save_res_to_file(sys_eq, './res_nahco3_cacl2_reduced_T_2.py', 'res', numbafy=False)
    return sys_eq

def get_caco3_nahco3_equilibrium():
    feed_compounds = ['Na+', 'HCO3-', 'Ca++', 'Cl-']
    fixed_elements = ['Cl-']
    # element_mass_balance = ['Na', 'Ca']
    # closing_equation_type = pyequion.ClosingEquationType.CARBON_TOTAL

    sys_eq = pyequion.create_equilibrium(feed_compounds,
        fixed_elements=fixed_elements,
        possible_aqueous_reactions_in=sys_reactions_simpler,
    )
    return sys_eq


if __name__ == "__main__":
    main_create_nahco3_cacl2_closed_sys()
