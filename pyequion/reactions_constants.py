import numpy
global np
np = numpy
from .properties_utils import Indexes_db, tags_db
from collections import namedtuple
import numba

# tags_reactions_db = ['NaOH', 'NaCO3', 'NaHCO3', 'Na2CO3', 'CaOH', 'NaCl', 'HCl',
#     'H', 'a1', 'a2', 'w', 'CaHCO3p', 'CaCO3', 'calcite', 'vaterite', 'aragonite'
# ]
# dtype_logReactionConstants = [(tag, np.float64) for tag in tags_reactions_db]
# LogReactionConstants = namedtuple('LogReactionConstants', tags_reactions_db)

# logK = np.empty(1, dtype=dtype_logReactionConstants)

@numba.njit
def logK_NaOH(TK): return -14.18
@numba.njit
def logK_NaCO3(TK): return 1.27
@numba.njit
def logK_NaHCO3(TK): return -0.25
@numba.njit
def logK_Na2CO3(TK): return 0.672
@numba.njit
def logK_CaOH(TK): return -12.78
@numba.njit
def logK_NaCl(TK): return -1.602
@numba.njit
def logK_HCl(TK): return -6.100
@numba.njit
def logK_H(TK):
    logK = 108.3865 + 0.01985076*TK - 6919.53/TK - 40.45154*np.log10(TK) + 669365.0/(TK**2)
    return logK
@numba.njit
def logK_a1(TK): return -356.3094 - 0.06091964*TK + 21834.37/TK + 126.8339*np.log10(TK) -1684915/(TK**2)
@numba.njit
def logK_a2(TK): return -107.8871 - 0.03252849*TK + 5151.79/TK + 38.92561*np.log10(TK) -563713.9/(TK**2)
@numba.njit
def logK_w(TK): return -283.9710 - 0.05069842*TK + 13323.00/TK + 102.24447*np.log10(TK) -1119669/(TK**2)
@numba.njit
def logK_CaHCO3(TK): return 1209.120 + 0.31294*TK - 34765.05/TK - 478.782*np.log10(TK)
@numba.njit
def logK_CaCO3(TK): return -1228.732 -0.29944*TK + 35512.75/TK + 485.818*np.log10(TK)
@numba.njit
def logK_calcite(TK): return -171.9065 - 0.077993*TK + 2839.319/TK + 71.595*np.log10(TK)
@numba.njit
def logK_vaterite(TK): return -172.1295 - 0.077993*TK + 3074.688/TK + 71.595*np.log10(TK)
@numba.njit
def logK_aragonite(TK): return-171.9773 - 0.077993*TK + 2903.293/TK + 71.595*np.log10(TK)

# THIS SHOULD BE THE NEW NUMBA DICT -> Using just for consulting tags
# How to do with the polymorphs ?
db_reactions = (
    'NaOH',
    'NaCO3m',
    'NaHCO3',
    'Na2CO3',
    'CaOH',
    'NaCl',
    'HCl',
    'H',
    'a1',
    'a2',
    'w',
    'CaHCO3+',
    'CaCO3', #aqueous
    'CaCO3(s)', #calcite
    'vaterite',
    'aragonite',
    'BaCO3',
    'BaHCO3+',
    'BaOH+',
    'BaCO3(s)',
)


@numba.njit
def calculate_log10_equilibrium_constant(TK):
    #TK = T + 273.15

    # Calculate the log10 of the equilibrium constants
    # Nordstrom, D. K., Plummer, L. N., Langmuir, D., Busenberg, E., May, H. M., Jones, B. F., & Parkhurst, D. L. (1990). Revised Chemical Equilibrium Data for Major Water—Mineral Reactions and Their Limitations (pp. 398–413). https://doi.org/10.1021/bk-1990-0416.ch031

    # logK = LogReactionConstants()
    # logK = np.empty(1, dtype=dtype_logReactionConstants)
    logK = np.empty(20) #FIXME
    logK[0] = -14.18 #'NaOH'
    logK[1] = 1.27  #'NaCO3m'
    logK[2] = -0.25  #'NaHCO3'
    logK[3] = 0.672  #'Na2CO3'
    logK[4] = -12.78  #'CaOH'
    logK[5] = -1.602  #'NaCl'
    logK[6] = -6.100 #'HCl'
    logK[7] = 108.3865 + 0.01985076*TK - 6919.53/TK - 40.45154*np.log10(TK) + 669365.0/(TK**2) #'H'
    logK[8] = -356.3094 - 0.06091964*TK + 21834.37/TK + 126.8339*np.log10(TK) -1684915/(TK**2) #'a1'
    logK[9] = -107.8871 - 0.03252849*TK + 5151.79/TK + 38.92561*np.log10(TK) -563713.9/(TK**2) #'a2'
    logK[10] = -283.9710 - 0.05069842*TK + 13323.00/TK + 102.24447*np.log10(TK) -1119669/(TK**2) #'w'
    logK[11] = 1209.120 + 0.31294*TK - 34765.05/TK - 478.782*np.log10(TK) #'CaHCO3+
    logK[12] = -1228.732 -0.29944*TK + 35512.75/TK + 485.818*np.log10(TK) #'CaCO3'
    logK[13] = -171.9065 - 0.077993*TK + 2839.319/TK + 71.595*np.log10(TK) #'calcite'
    logK[14] = -172.1295 - 0.077993*TK + 3074.688/TK + 71.595*np.log10(TK) #'vaterite'
    logK[15] =-171.9773 - 0.077993*TK + 2903.293/TK + 71.595*np.log10(TK) #'aragonite'

    logK[16] = 0.113 + 0.008721*TK #'baco3'
    logK[17] = -3.0938 + 0.013669*TK # bahco3+
    logK[18] = -13.47 # BaOH+ - 25C
    logK[19] = 607.642 + 0.121098*TK - 20011.25/TK - 236.4948*np.log10(TK) #'baco3(s) - witherite'
    return logK
