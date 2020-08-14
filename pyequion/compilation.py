import numba
from . import core

def dynamically_compile():
    core.populate_loggama_activities = numba.jit(
        core.populate_loggama_activities,
        nopython=True
    )
    Index_blueprint = numba.jitclass(core.Index_specs)
    newIndex = Index_blueprint(core.Indexes)
    return
