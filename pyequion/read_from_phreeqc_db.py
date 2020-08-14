# import json
import commentjson as json

import numpy as np
import pandas as pd

# fname = 'phreeqc.dat'

def read_phreeqc_as_df(fname):
    my_cols = [str(i) for i in range(24)]
    df = pd.read_csv(fname,
                encoding = "ISO-8859-1", header=None,
                sep=r"\t|,| |", names=my_cols,
                skip_blank_lines=True, comment='#',
                engine='python')
    return df

def conditions_is_element_in_datfile(item):
    if item.strip() in ['+', '-', '=']:
        return False
    elif is_number(item):
        return False
    else:
        return True

def conditions_is_element_or_coef_in_datfile(item):
    if item.strip() in ['+', '-', '=']:
        return False
    else:
        return True

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def get_qt_of_a_sign(ele, sign):
    splitted = ele.split(sign)
    if len(splitted) > 1:
        if is_number(splitted[1]):
            psigns = int(splitted[1])
        else:
            psigns = 1
    else:
        return 0
    return psigns

def convert_to_qt_sign(ele):
    splitted = ele.split('-')
    if len(splitted) > 1:
        minus = get_qt_of_a_sign(ele, '-')
        minus_str = '-' * minus
        ele_ret = splitted[0] + minus_str
        return ele_ret
    splitted = ele.split('+')
    if len(splitted) > 1:
        plus = get_qt_of_a_sign(ele, '+')
        plus_str = '+' * plus
        ele_ret = splitted[0] + plus_str
        return ele_ret
    return ele

def convert_ele_list_as_db(ele_list_raw):
    vals_ele = [v for v in ele_list_raw if conditions_is_element_in_datfile(v)]
    r = {}
    for ele in vals_ele:
        if is_number(ele[0]): #first char is number
            coef = float(ele[0])
            ele = ele[1:]
        else: #check if past item has coef, if not is 1
            idx = ele_list_raw.index(ele)
            if idx == 0:
                coef = 1.0
            else:
                if is_number(ele_list_raw[idx-1]):
                    coef = float(ele_list_raw[idx-1])
                else:
                    coef = 1.0

        # Fixing signs
        ele_new = convert_to_qt_sign(ele)
        r[ele_new] = coef
    return r

def create_db_reaction_entries_solutions(df):
    idx_start = df[df.iloc[:,0] == 'SOLUTION_SPECIES'].index[0] + 1
    idx_final = df[df.iloc[:,0] == 'PHASES'].index[0]
    df_sol = df.iloc[idx_start:idx_final, :]

    reactions_list = []
    row_elements = df_sol[df_sol['0'].str[0] != '-']
    for i in range(0, row_elements.shape[0]):
        row = row_elements.iloc[i, :]
        raw_reac = row.dropna()
        di = list(raw_reac.to_dict().values())
        eles_only = [v for v in di if conditions_is_element_in_datfile(v)]
        there_is_duplicated = len(eles_only) != len(set(eles_only))
        if there_is_duplicated:
            continue
        idx_equal = di.index('=')
        react_raw = di[0:idx_equal]
        prods_raw = di[idx_equal+1:]

        r_reactants = convert_ele_list_as_db(react_raw)
        r_products = convert_ele_list_as_db(prods_raw)

        r_reactants = {k:-v  for k,v in r_reactants.items()}

        if 'e-' in di:
            type_reac = 'electronic'
        else:
            type_reac = 'rev'

        # Get Coefficient Values
        try:
            idx_next =  row_elements.iloc[i+1, :].name
        except IndexError:
            idx_next = df_sol.iloc[-1,:].name

        df_ele = df_sol.loc[row.name:idx_next, :]
        logk25 = extract_field_phreeqc_db(df_ele, '-log_k')
        # dw = extract_field_phreeqc_db(df_ele, '-dw')
        deltah = extract_field_phreeqc_db(df_ele, '-delta_h')
        # Vm = extract_field_phreeqc_db(df_ele, '-Vm')

        logK_coefs = extract_field_phreeqc_db(df_ele, '-analytic')

        r_reaction = dict(
            **r_reactants, **r_products,
            log_K25=logk25,
            log_K_coefs=logK_coefs,
            deltah=deltah,
            # dw=dw,
            # Vm=Vm,
            type=type_reac
        )

        reactions_list.append(r_reaction)
    return reactions_list

def create_db_reaction_entries_phases(df):
    idx_start = df[df.iloc[:,0] == 'PHASES'].index[0] + 1
    idx_final = df[df.iloc[:,0] == 'EXCHANGE_MASTER_SPECIES'].index[0]
    df_sol = df.iloc[idx_start:idx_final, :]

    reactions_list = []
    row_elements = df_sol[df_sol['0'].str[0] != '-']
    row_elements = row_elements.iloc[0:-1:2]

    for i in range(0, row_elements.shape[0]):
        row = row_elements.iloc[i, :]
        phase_name = row.iloc[0]
        row_eq = df_sol.loc[row.name+1, :]
        raw_reac = row_eq.dropna()
        di = list(raw_reac.to_dict().values())
        eles_only = [v for v in di if conditions_is_element_in_datfile(v)]
        there_is_duplicated = len(eles_only) != len(set(eles_only))
        # if there_is_duplicated:
        #     continue #FIXME - removing the gas phase

        #debug
        if di[0] == 'NaCl':
            a = 1

        idx_equal = di.index('=')
        react_raw = di[0:idx_equal]
        if '(g)' in phase_name:
            continue #FIXME Skipping gases for the moment
            react_raw[0] += '(g)'
        else:
            react_raw[0] += '(s)'
        prods_raw = di[idx_equal+1:]

        r_reactants = convert_ele_list_as_db(react_raw)
        r_products = convert_ele_list_as_db(prods_raw)

        r_reactants = {k:-v  for k,v in r_reactants.items()}

        if 'e-' in di:
            type_reac = 'electronic'
        else:
            type_reac = 'rev'

        # Get Coefficient Values
        try:
            idx_next =  row_elements.iloc[i+1, :].name
        except IndexError:
            idx_next = df_sol.iloc[-1,:].name

        df_ele = df_sol.loc[row.name:idx_next, :]
        logk25 = extract_field_phreeqc_db(df_ele, '-log_k')

        logK_coefs = extract_field_phreeqc_db(df_ele, '-analytic')

        r_reaction = dict(
            **r_reactants, **r_products,
            log_K25=logk25,
            log_K_coefs=logK_coefs,
            type=type_reac,
            phase_name= phase_name
        )

        reactions_list.append(r_reaction)
    return reactions_list

def extract_field_phreeqc_db(df_ele, string_match):
    #
    rowlog_K25 = df_ele[df_ele['0'].str.strip() == string_match]
    if rowlog_K25.size > 0:
        rowlog_K25_na = rowlog_K25.dropna(axis=1)
        try:
            val = float(rowlog_K25_na.iloc[-1, 1:])
            # val = float(rowlog_K25_na.iloc[-1, 1])
        except TypeError:
            try:
                val = rowlog_K25_na.iloc[0, 1:].values.astype(np.float).tolist()
            except ValueError:
                val = float(rowlog_K25_na.iloc[-1, 1])
    else:
        val = ''
    return val

def convert_and_save_phreeqc_based(fname):
    df = read_phreeqc_as_df(fname)
    reactions_solution_species = create_db_reaction_entries_solutions(df)
    reactions_phase_species = create_db_reaction_entries_phases(df)

    with open('reactions_solutions.json', 'w') as json_file:
        json.dump(reactions_solution_species, json_file)

    with open('reactions_solids.json', 'w') as json_file:
        json.dump(reactions_phase_species, json_file)
