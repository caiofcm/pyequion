import os
import numpy as np
import numba

if os.getenv('NUMBA_DISABLE_JIT') != "1":
    from numba.typed import List, Dict
else:
    List = list
    Dict = dict #parei aqui

def create_nb_List(py_list):
    nb_List = List()

    # if py_list is None:
    #     nb_List.append('H+')
    #     del nb_List[0]
    #     return nb_List #numba does not deal with None

    # if len(py_list) == 0:
    # print(py_list)
    if not py_list:
        # print('db1', nb_List, py_list, type(nb_List), type(py_list))
        # print('he1')
        whatNew = List()
        # print('he1.2')
        whatNew.append('a')
        # print('he2')
        whatNew.pop()
        # print('he3')
        # del whatNew[0]
        # print('db2', whatNew, py_list, type(whatNew), type(py_list))
        return whatNew #numba does not deal with None

    for val in py_list:
        nb_List.append(val)
    return nb_List

def create_nb_Dict(py_dict):
    nb_Dict = Dict()
    # d = Dict.empty(
    #     key_type=numba.types.unicode_type,
    #     value_type=numba.types.unicode_type,
    # )
    for tag, val in py_dict.items():
        # print('dbd1', tag, val, type(tag), type(val))
        nb_Dict[tag] = val
    return nb_Dict

@numba.njit(cache=True)
def root_finding_newton(fun, J, x, eps, max_iter, args):
    """
    Solve nonlinear system fun(x)=0 by Newton's method.
    J is the Jacobian of fun(x). Both fun(x) and J must be functions of x.
    At input, x holds the start value. The iteration continues
    until ||F|| < eps.
    """
    F_value = fun(x, args)
    F_value_ = F_value.reshape((-1,1))
    F_norm = np.linalg.norm(F_value, 2)  # l2 norm of vector
    iteration_counter = 0
    while abs(F_norm) > eps and iteration_counter < max_iter:
        delta = np.linalg.solve(J(x, args), -F_value_)

        for i in range(x.size): #wtf numba!?!?!
            x[i] += delta[i,0]

        F_value = fun(x, args)
        F_value_ = F_value.reshape((-1,1))
        F_norm = np.linalg.norm(F_value, 2)
        iteration_counter += 1

    # Here, either a solution is found, or too many iterations
    if abs(F_norm) > eps:
        iteration_counter = -1
        raise ValueError('Maximum iteration reached in newton root finding!')
    return x, iteration_counter

@numba.njit#(cache=True)
def numeric_jacobian(fun, x, diff_eps, args):
    J = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        x1 = x.copy()
        x2 = x.copy()
        x1[i] += diff_eps
        x2[i] -= diff_eps
        f1 = fun(x1, args)
        f2 = fun(x2, args)
        J[:, i] = (f1 - f2) / (2 * diff_eps)

    return J

def create_jacobian(fun):

    @numba.njit()
    def numba_J(x, args):
        return numeric_jacobian(fun, x, 1e-8, args)
    return numba_J


@numba.njit(cache=True)
def root_finding_newton_previously(fun, J, x, eps, max_iter, args):
    """
    Solve nonlinear system fun(x)=0 by Newton's method.
    J is the Jacobian of fun(x). Both fun(x) and J must be functions of x.
    At input, x holds the start value. The iteration continues
    until ||F|| < eps.
    """
    F_value = fun(x, args)
    F_value_ = F_value.reshape((-1,1))
    F_norm = np.linalg.norm(F_value, 2)  # l2 norm of vector
    iteration_counter = 0
    while abs(F_norm) > eps and iteration_counter < max_iter:
        delta = np.linalg.solve(J(x, args), -F_value)
        x = x + delta
        F_value = fun(x, args)
        F_value_ = F_value.reshape((-1,1))
        F_norm = np.linalg.norm(F_value, 2)
        iteration_counter += 1

    # Here, either a solution is found, or too many iterations
    if abs(F_norm) > eps:
        iteration_counter = -1
        raise ValueError('Maximum iteration reached in newton root finding!')
    return x, iteration_counter

# Testing:
@numba.njit(cache=True)
def F(x, args):
    return np.array(
        [x[0]**2 - x[1] + x[0]*np.cos(args[0]*x[0]),
            x[0]*x[1] + np.exp(-x[1]) - x[0]**(-1)])

@numba.njit(cache=True)
def J(x, args):
    return np.array(
        [[2*x[0] + np.cos(args[0]*x[0]) - args[0]*x[0]*np.sin(args[0]*x[0]), -1],
            [x[1] + x[0]**(-2), x[0] - np.exp(-x[1])]])

if __name__ == "__main__":

    expected = np.array([1.0, 0.0])
    tol = 1e-4
    x_guess = np.array([2.0, -1.0])
    args = (np.pi,)

    J_num = numeric_jacobian(F, x_guess, 1e-8, args)
    J_ext = J(x_guess, args)
    J_numba = create_jacobian(F)

    x, n = root_finding_newton(F, J_numba, x_guess, 0.0001, 100, args)
    print(n, x)
    error_norm = np.linalg.norm(expected - x, ord=2)
    assert error_norm < tol, 'norm of error =%g' % error_norm
    print('norm of error =%g' % error_norm)
