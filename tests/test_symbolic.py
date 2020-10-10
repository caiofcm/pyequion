# -*- coding: utf-8 -*-
import os

os.environ["NUMBA_DISABLE_JIT"] = "1"
import pytest
import sympy

import pyequion

# from pyequion.symbolic_computations import (
#     symb.numbafy_function_string,
#     symb.get_symbol_name,
# )
from pyequion import pCO2_ref
from pyequion import symbolic_computations as symb

x = sympy.symbols("x0:10")

STR_AUX_TEST = "r = [[-a0 + 2*math.sqrt(y0), y0**2*math.log(y1)], [-a1 + math.cos(y1), math.log(y1)*math.erf(2)/a1]]"
STR_AUX_TEST_NUMPY = "r = [[-a0 + 2*numpy.sqrt(y0), y0**2*numpy.log(y1)], [-a1 + numpy.cos(y1), numpy.log(y1)*numpy.erf(2)/a1]]"


@pytest.fixture()
def symbolic_vars():
    y = sympy.symbols("y0:2")
    a = sympy.symbols("a0:2")
    expr = [
        [sympy.sqrt(y[0]) * 2 - a[0], y[0] ** 2 * sympy.log(y[1])],
        [sympy.cos(y[1]) - a[1], sympy.erf(2) / a[1] * sympy.log(y[1])],
    ]
    return expr, y, a


def test_input_name_more_than_a_word():
    v = sympy.symbols("args")
    name = symb.get_symbol_name(v)
    assert name == "args"

    v = sympy.symbols("x0:2")
    name = symb.get_symbol_name(v)
    assert name == "x"

    v = sympy.symbols("args0:2")
    name = symb.get_symbol_name(v)
    assert name == "args"

    v = sympy.symbols("vari2able0:2")
    name = symb.get_symbol_name(v)
    assert name == "vari2able"


def test_string_inputs_as_tuple_descontruction():
    s = symb.string_inputs_as_tuple_descontruction(x)
    assert isinstance(s, str)
    assert s[1] == "x"
    assert s == "(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9) = x"


def test_lambdastr_as_regular_function(symbolic_vars):
    expr, y, a = symbolic_vars
    s = symb.string_lambdastr_as_function(expr, y, a)
    assert "def calc_expression(y, a):\n" in s
    assert STR_AUX_TEST in s


def test_lambdastr_as_regular_function_numpy(symbolic_vars):
    expr, y, a = symbolic_vars
    s = symb.string_lambdastr_as_function(expr, y, a, use_numpy=True)
    assert "def calc_expression(y, a):\n" in s
    assert STR_AUX_TEST_NUMPY in s


def test_lambdastr_as_regular_function_v_simple_fun_numpy(symbolic_vars):
    v = sympy.symbols("v")
    expr = v ** 2 - 4 * v
    s = symb.string_lambdastr_as_function(expr, v, use_numpy=True)
    assert "def calc_expression(v):\n" in s
    assert "r = v**2 - 4*v" in s


def test_lambdastr_as_regular_function_numpy_imports(symbolic_vars):
    expr, y, a = symbolic_vars
    s = symb.string_lambdastr_as_function(
        expr, y, a, use_numpy=True, include_imports=True
    )
    assert "def calc_expression(y, a):\n" in s
    assert STR_AUX_TEST_NUMPY in s
    assert "import numpy" in s


def test_save_to_file_stringfied_function_expression(
    data_files_dir, symbolic_vars
):
    expr, y, a = symbolic_vars
    s = symb.string_lambdastr_as_function(expr, y)
    symb.save_function_string_to_file(s, str(data_files_dir.join("aow.py")))


def test_save_to_file_stringfied_function_expression_save_local(
    data_files_dir, symbolic_vars
):
    expr, y, a = symbolic_vars
    s = symb.string_lambdastr_as_function(expr, y)
    symb.save_function_string_to_file(s)


@pytest.fixture()
def string_function_sample():
    ss = """
import numpy

def cacl2_residual_jacobian(x, args):
    (x0, x1, x2, x3, x4, x5, x6, x7, x8) = x
    ([args0], args1, args2) = args
    r = [[-0.25 * 10**x0 * numpy.log(10), -0.25 * 10**x1 * numpy.log(10), 1, -1.0 * 10**x3 * numpy.log(10), -0.25 * 10**x4 * numpy.log(10), -0.25 * 10**x5 * numpy.log(10), -0.25 * 10**x6 * numpy.log(10), 0, -1.0 * 10**x8 * numpy.log(10)]
    ]
    return r"""
    return ss


def test_numbafied_string_function(string_function_sample):
    ss = string_function_sample

    new_ss = symb.numbafy_function_string(ss)

    assert "import numba" in new_ss
    assert "@numba.njit()\ndef cacl2_residual_jacobian(x, args, )" in new_ss


def test_numbafied_string_function_kwarg(string_function_sample):
    ss = string_function_sample
    kwarg_string = "cache=True"
    new_ss = symb.numbafy_function_string(ss, kwarg_string)

    assert "import numba" in new_ss
    assert (
        "@numba.njit(cache=True)\ndef cacl2_residual_jacobian(x, args, )"
        in new_ss
    )


def test_regex_function_defined(string_function_sample):
    ss = string_function_sample
    r = symb.regex_function_defined(ss)
    assert "def cacl2_residual_jacobian(x, args):" in r[0]


def test_numbafied_string_dummy_args_added(string_function_sample):
    ss = string_function_sample
    kwarg_string = "cache=True"
    add_arg_fun = "dummy"
    new_ss = symb.numbafy_function_string(ss, kwarg_string, add_arg_fun)

    assert "import numba" in new_ss
    assert (
        "@numba.njit(cache=True)\ndef cacl2_residual_jacobian(x, args, {})".format(
            add_arg_fun
        )
        in new_ss
    )


if __name__ == "__main__":
    pass
