__doc__ = """
    Symbolic calculation for code generation and automatic jacobian computation.
"""
import sympy
import pyequion
import numpy
import re
from sympy.utilities.lambdify import lambdastr

REGEX_FUNC_DEFINITION = r"def\s+\w+\([\w,\s]+\)\:"
# was u instead of r


def prepare_for_sympy_substituting_numpy():
    sympy.log10 = lambda x: sympy.log(x, 10)
    sympy.empty = lambda n: [None] * n
    sympy.array = lambda vals: [v for v in vals]
    sympy.sum = lambda vals: sum([v for v in vals])
    sympy.isfinite = lambda v: v * 0 == 0

    def any_aux(vals):
        try:
            for v in vals:
                if v is True:
                    return True
        except TypeError:
            return vals is True
        return False

    def isnan_aux(vals):
        try:
            for v in vals:
                if v * 0 != 0:
                    return True
        except TypeError:
            return vals * 0 != 0  # CHECKME
        return False

    sympy.any = any_aux
    sympy.isnan = isnan_aux

    pyequion.activity_coefficients.np = sympy
    # pyequion.reactions_constants.np = sympy
    # pyequion.pyequion.np = sympy
    pyequion.np = sympy
    pyequion.core.np = sympy
    return


def return_to_sympy_to_numpy():
    pyequion.activity_coefficients.np = numpy
    # pyequion.reactions_constants.np = numpy
    # pyequion.pyequion.np = numpy
    pyequion.np = numpy
    pyequion.core.np = numpy
    return


def obtain_symbolic_jacobian(y, x):
    n = len(x)
    J = []
    for i in range(n):
        Jaux = []
        for j in range(n):
            Jaux += [sympy.diff(y[i], x[j])]
        J += [Jaux]
    return J


def save_symbolic_expression_to_file(expr, x, fun_name=None):

    if fun_name is None:
        fun_name = "calc_expression"
    header = "def " + fun_name + "(x, args):\n"

    body = "\t\n"
    body += "\t" + string_inputs_as_tuple_descontruction(x) + "\n"
    body += "\tr = " + str(expr) + "\n"

    body += "\treturn r\n"

    with open("out_expression.py", "w") as f:
        f.write(header + body)
    return


def string_lambdastr_as_function(
    expr, x, a=None, fun_name=None, use_numpy=False, include_imports=False
):
    if fun_name is None:
        fun_name = "calc_expression"
    s_lambda = lambdastr(x, expr)

    x_name = get_symbol_name(x)
    header, content = s_lambda.split(":")
    x_tuple_str = string_inputs_as_tuple_descontruction(x)

    if a is not None:
        a_name = get_symbol_name(a)
        a_tuple_str = string_inputs_as_tuple_descontruction(a)
        header = "def {}({}, {}):\n".format(fun_name, x_name, a_name)
    else:
        header = "def {}({}):\n".format(fun_name, x_name)
        a_tuple_str = ""

    content = content.strip()
    content = content[1:-1]

    # Im not sure when math is applyed to the converted string or not by sympy
    # FIXME: for now i will force removing math to them applying it again
    content = content.replace("math.", "")
    content = content.replace("sqrt(", "math.sqrt(")
    content = content.replace("log(", "math.log(")
    content = content.replace("cos(", "math.cos(")
    content = content.replace(
        "erf(", "math.erf("
    )  # this is an error, lambdastr should be keeping math or numpy, check later

    if use_numpy:
        content = content.replace("math", "numpy")

    full = "{}\t{}\n\t{}\n\tr = {}\n\treturn r\n".format(
        header, x_tuple_str, a_tuple_str, content
    )

    if include_imports:
        is_numpy = "import numpy\n" if "numpy" in full else ""
        is_math = "import math\n" if "math" in full else ""
        full = is_numpy + is_math + "\n" + full

    return full


def numbafy_function_string(s, numba_kwargs_string="", func_additional_arg=""):
    if "import numba" not in s:
        s = "import numba\n" + s
    matches = regex_function_defined(s)
    r = s
    decoration_string = "@numba.njit({})\n".format(numba_kwargs_string)
    for match in matches:
        func_def_str = match.split("):")[0] + ", " + func_additional_arg + "):"
        r = re.sub(REGEX_FUNC_DEFINITION, decoration_string + func_def_str, r)
    return r


def regex_function_defined(s):
    matches = re.findall(REGEX_FUNC_DEFINITION, s)
    return matches


def save_function_string_to_file(s_func, filename="out_expression.py"):

    with open(filename, "w") as f:
        f.write(s_func)
    return


def string_inputs_as_tuple_descontruction(x):
    try:
        _ = len(x)
    except TypeError:
        return ""
    str_var = str(x)
    name = get_symbol_name(x)
    str_var += " = " + name
    return str_var


def get_symbol_name(x):
    try:
        s = get_symbol_name(x[0])
        # s = str(x[0])
        s_aux = s
        for i in range(len(s) - 1, 0 - 1, -1):
            if s[i].isdigit():
                s_aux = s_aux[0:i]
            else:
                break
        return s_aux
    except TypeError:
        return str(x)
