import pyequion

def main_create_nahco3_cacl2_closed_sys():
    feed_compounds = ['Na+', 'HCO3-', 'Ca++', 'Cl-']
    fixed_elements = ['Cl-']
    # element_mass_balance = ['Na', 'Ca']
    closing_equation_type = pyequion.ClosingEquationType.CARBON_TOTAL

    sys_eq = pyequion.create_equilibrium(feed_compounds,
        # closing_equation_type,
        # element_mass_balance,
        fixed_elements=fixed_elements,
    )

    pyequion.display_reactions(sys_eq)

    SAVE_RES = False
    if SAVE_RES:
        pyequion.save_res_to_file(sys_eq, './res_nahco3_cacl2_reduced_T_2.py', 'res', numbafy=False)
    return sys_eq


if __name__ == "__main__":
    main_create_nahco3_cacl2_closed_sys()
