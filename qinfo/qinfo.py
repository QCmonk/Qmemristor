"""
A collection of useful functions for problems in quantum information theory.

"""

import numpy as np
from scipy.linalg import sqrtm,block_diag
from itertools import product
from qinfo.operators import op
#import tensorflow as tf
import cvxpy


# TODO: redo partial trace function to compute it for arbitrary dimensional subsystems



# compute effect of a CPTP map
def CPTP(kraus, rho):

    """
    Class definition that handles signal reconstruction of a transformed input signal given
    the measurement basis. Able to perform standard compressed sensing and compressive 
    matched filter processing. 

    Parameters
    ----------
    svector : one dimensional numpy array
        the sensing or measurement vector y such that y = Ax where 
        x is the signal to reconstruct.

    transform : m*n array where m is len(svector) and n is the 
                dimension of the signal to reconstruct.

    verbose : boolean 
        Whether to print progress reports as reconstruction is performed.

    kwargs : optional
        "template": None,       (template signal for matched filtering)
        "epsilon": 0.01         (radius of hyperdisc for CVX problem)
        "length": len(svector) 
        Optional arguments - some are required for different functionality


    Returns
    -------
    CAOptimise class instance

    Raises
    ------
    KeyError
        If no measurement transform has been specified.
    """



    nrho = np.zeros(rho.shape, dtype='complex128')
    for i in kraus:
        nrho += np.dot(i, np.dot(rho, i.conj()))
    return nrho


# generates tensor products of arrayset according to combination
def kronjob(arrayset, combination):

    """
    Class definition that handles signal reconstruction of a transformed input signal given
    the measurement basis. Able to perform standard compressed sensing and compressive 
    matched filter processing. 

    Parameters
    ----------
    svector : one dimensional numpy array
        the sensing or measurement vector y such that y = Ax where 
        x is the signal to reconstruct.

    transform : m*n array where m is len(svector) and n is the 
                dimension of the signal to reconstruct.

    verbose : boolean 
        Whether to print progress reports as reconstruction is performed.

    kwargs : optional
        "template": None,       (template signal for matched filtering)
        "epsilon": 0.01         (radius of hyperdisc for CVX problem)
        "length": len(svector) 
        Optional arguments - some are required for different functionality


    Returns
    -------
    CAOptimise class instance

    Raises
    ------
    KeyError
        If no measurement transform has been specified.
    """



    # initialise array
    matrix = 1
    # construct appropriate tensor product
    for el in combination:
        matrix = np.kron(matrix, arrayset[int(el)])
    return matrix


def dagger(M):
    """
    Return conjugate transpose of input array
    """
    return np.transpose(np.conjugate(M))


def eye_like(m, dtype=np.complex128):
    """
    Returns identity matrix with same dims as square matrix m.
    """
    return np.eye(np.shape(m)[0], dtype=dtype)


def basis_gen(dim, basis="comp"):
    """
    generates a set of orthonormal basis vectors for the input basis set and dimension
    """

    if basis=="comp":
        # slow, using zeroes is faster
        basis = np.eye(dim, dtype=np.complex128)
    
    # TODO
    else:
        print("Unknown basis definition")
        pass

    return basis



def partial_trace2(m, sys_trace, dims=None):
    """"
    Updated version of the partial trace, much more sensible
    """

    # extract input state dimension
    sys_dim = np.shape(m)[0]
    
    # default to uniform qubit case
    if dims is None:
        # compute number of subsystems in input
        sys_num = int(np.round(np.log2(sys_dim)))
        dims = [2]*sys_num
    else:
        sys_num = len(dims)

    # construct a basis of appropriate dimension for each subsystem
    basis_dict = {}
    for i in dims:
        # skip if dimension already has basis
        if str(i) in basis_dict.keys(): continue

        # add to dictionary otherwise
        basis_dict[str(i)] = basis_gen(i)

    # compute dimension of output Hilbert spaces
    output_dim = int(round(sys_dim/np.prod(np.asarray(dims)[sys_trace])))

    # preallocate output density operator
    output = np.zeros((output_dim, output_dim), dtype=np.complex128)

    # initialise projector counter    
    proj_cnt = np.asarray([None]*sys_num)
    proj_cnt[sys_trace] = 0


    for i in range(0, np.prod(np.prod(np.asarray(dims)[sys_trace]))):
        # construct next projector
        proj = 1.0
        for subsys,el in enumerate(proj_cnt):
            # get dimension of subsystem
            sys_dim = dims[subsys]
            # identity projection if subsystem is being kept
            if el is None: 
                proj = np.kron(proj, np.eye(sys_dim))
            else: 

                # retrieve correct basis element
                local_proj = basis_dict[str(sys_dim)][:,el].reshape(sys_dim,1)
                proj = np.kron(proj, local_proj)

        # apply to subsystem and add to trace sum
        output += dagger(proj) @ m @ proj

        # update projector counter from lower to higher until incrementation occurs
        increment = False
        for subsys,el in enumerate(proj_cnt):
            # skip subsystem if not being traced out
            if el is None: 
                continue

            # compute subsystem dimension
            sys_dim = dims[subsys]
            # increment index if less than total number and exit
            if el+1 < sys_dim:
                proj_cnt[subsys] += 1
                increment = True
                break
            else:
                proj_cnt[subsys] = 0

    return output


# # computes the partial trace of density operator m \in L(H_d), tracing out subsystems in the list sys
# def partial_trace(m, sys):

#     """
#     Class definition that handles signal reconstruction of a transformed input signal given
#     the measurement basis. Able to perform standard compressed sensing and compressive 
#     matched filter processing. 

#     Parameters
#     ----------
#     svector : one dimensional numpy array
#         the sensing or measurement vector y such that y = Ax where 
#         x is the signal to reconstruct.

#     transform : m*n array where m is len(svector) and n is the 
#                 dimension of the signal to reconstruct.

#     verbose : boolean 
#         Whether to print progress reports as reconstruction is performed.

#     kwargs : optional
#         "template": None,       (template signal for matched filtering)
#         "epsilon": 0.01         (radius of hyperdisc for CVX problem)
#         "length": len(svector) 
#         Optional arguments - some are required for different functionality


#     Returns
#     -------
#     CAOptimise class instance

#     Raises
#     ------
#     KeyError
#         If no measurement transform has been specified.
#     """


#     # type enforcement
#     m = np.asarray(m)
#     # sort subsystems
#     sys = sorted(sys)
#     # get tensor dimensions
#     qnum = int(round(np.log2(len(m))))
#     # compute dimensions of tensor
#     tshape = (2,)*2*qnum
#     # reshape to tensor
#     mtensor = m.reshape((tshape))
#     # compute dimensions to trace over
#     index1, index2 = sys[0], sys[0] + qnum
#     del sys[0]
#     newdim = 2**(qnum-1)
#     # compute reduced density matrix via recursion
#     if len(sys) > 0:
#         # trace out target subsystem (repeated reshaping is a bit rough but its not worth the fix time)
#         mtensor = np.trace(mtensor, axis1=index1,
#                            axis2=index2).reshape((newdim, newdim))
#         # adjust subsequent target dimensions with shallow copy
#         sys[:] = [i-1 for i in sys]
#         # by the power of recursion!
#         mtensor = partialtrace(mtensor, sys)
#     else:
#         # bottom of the pile, compute and pass up the chain
#         mtensor = np.trace(mtensor, axis1=index1,
#                            axis2=index2).reshape((newdim, newdim))
#     return mtensor








# generates an operator spanning set for any system with qnum qubits
def opbasis(qnum):

    """
    Class definition that handles signal reconstruction of a transformed input signal given
    the measurement basis. Able to perform standard compressed sensing and compressive 
    matched filter processing. 

    Parameters
    ----------
    svector : one dimensional numpy array
        the sensing or measurement vector y such that y = Ax where 
        x is the signal to reconstruct.

    transform : m*n array where m is len(svector) and n is the 
                dimension of the signal to reconstruct.

    verbose : boolean 
        Whether to print progress reports as reconstruction is performed.

    kwargs : optional
        "template": None,       (template signal for matched filtering)
        "epsilon": 0.01         (radius of hyperdisc for CVX problem)
        "length": len(svector) 
        Optional arguments - some are required for different functionality


    Returns
    -------
    CAOptimise class instance

    Raises
    ------
    KeyError
        If no measurement transform has been specified.
    """


    # define operator basis set (Pauli set in this case)
    opset = np.asarray([op['id'],
                        op['x'],
                        op['y'],
                        op['z']])

    # determine all combinations
    combs = iter(''.join(seq)
                 for seq in product(['0', '1', '2', '3'], repeat=qnum))
    operbasis = []
    # construct density basis
    for item in combs:
        operbasis.append(kronjob(opset, item))
    operbasis = np.asarray(operbasis)

    # return operator basis
    return operbasis

def meas_gen(N=1):

    """
    Class definition that handles signal reconstruction of a transformed input signal given
    the measurement basis. Able to perform standard compressed sensing and compressive 
    matched filter processing. 

    Parameters
    ----------
    svector : one dimensional numpy array
        the sensing or measurement vector y such that y = Ax where 
        x is the signal to reconstruct.

    transform : m*n array where m is len(svector) and n is the 
                dimension of the signal to reconstruct.

    verbose : boolean 
        Whether to print progress reports as reconstruction is performed.

    kwargs : optional
        "template": None,       (template signal for matched filtering)
        "epsilon": 0.01         (radius of hyperdisc for CVX problem)
        "length": len(svector) 
        Optional arguments - some are required for different functionality


    Returns
    -------
    CAOptimise class instance

    Raises
    ------
    KeyError
        If no measurement transform has been specified.
    """


    """
    generates a set of projective measurements for N qubits
    """
    # base projectors
    projs_base = [np.kron(projs["0"],dagger(projs["0"])),
                  np.kron(projs["1"],dagger(projs["1"]))]

    if N==1:
        return projs_base
    else:
        pass


def swap_gen(N, targets):
    """
    Generates a swap gate acting on subsytem [targets]. 
    """

    # define a basis set with which to define operations

    pass 



def bell_gen(N=2):

    """
    Class definition that handles signal reconstruction of a transformed input signal given
    the measurement basis. Able to perform standard compressed sensing and compressive 
    matched filter processing. 

    Parameters
    ----------
    svector : one dimensional numpy array
        the sensing or measurement vector y such that y = Ax where 
        x is the signal to reconstruct.

    transform : m*n array where m is len(svector) and n is the 
                dimension of the signal to reconstruct.

    verbose : boolean 
        Whether to print progress reports as reconstruction is performed.

    kwargs : optional
        "template": None,       (template signal for matched filtering)
        "epsilon": 0.01         (radius of hyperdisc for CVX problem)
        "length": len(svector) 
        Optional arguments - some are required for different functionality


    Returns
    -------
    CAOptimise class instance

    Raises
    ------
    KeyError
        If no measurement transform has been specified.
    """


    """"
    Generate an N qubit Bell state |psi^+>
    """

    bell = np.zeros((2**N,2**N), dtype=np.complex128)

    # exploit structure rather than constructing a circuit
    bell[0,0] = 0.5
    bell[-1,0] = 0.5
    bell[0,-1] = 0.5
    bell[-1,-1] = 0.5

    return bell

def rho_gen(N=1):

    """
    Class definition that handles signal reconstruction of a transformed input signal given
    the measurement basis. Able to perform standard compressed sensing and compressive 
    matched filter processing. 

    Parameters
    ----------
    svector : one dimensional numpy array
        the sensing or measurement vector y such that y = Ax where 
        x is the signal to reconstruct.

    transform : m*n array where m is len(svector) and n is the 
                dimension of the signal to reconstruct.

    verbose : boolean 
        Whether to print progress reports as reconstruction is performed.

    kwargs : optional
        "template": None,       (template signal for matched filtering)
        "epsilon": 0.01         (radius of hyperdisc for CVX problem)
        "length": len(svector) 
        Optional arguments - some are required for different functionality


    Returns
    -------
    CAOptimise class instance

    Raises
    ------
    KeyError
        If no measurement transform has been specified.
    """


    """
    Generates a spanning set of density matrices for N qubits
    """
    dim = 2**N

    # initialise basis set
    rhob = np.empty((dim**2, dim, dim), dtype=np.complex128)
    # define component set
    rhobase = np.empty((4, 2, 2), dtype=np.complex128)
    rhobase[0, :, :] = np.asarray([[1, 0], [0, 0]])
    rhobase[1, :, :] = np.asarray([[0, 0], [0, 1]])
    rhobase[2, :, :] = np.asarray([[1, 1], [1, 1]])/2
    rhobase[3, :, :] = np.asarray([[1, -1j], [1j, 1]])/2

    # generate permutation list
    combs = product(['0', '1', '2', '3'], repeat=N)
    for i, comb in enumerate(combs):
        rho = 1.0
        for j in comb:
            rho = np.kron(rho, rhobase[int(j), :, :])

        rhob[i, :, :] = rho
    return rhob


def dual_gen(rhob, N=1):

    """
    Class definition that handles signal reconstruction of a transformed input signal given
    the measurement basis. Able to perform standard compressed sensing and compressive 
    matched filter processing. 

    Parameters
    ----------
    svector : one dimensional numpy array
        the sensing or measurement vector y such that y = Ax where 
        x is the signal to reconstruct.

    transform : m*n array where m is len(svector) and n is the 
                dimension of the signal to reconstruct.

    verbose : boolean 
        Whether to print progress reports as reconstruction is performed.

    kwargs : optional
        "template": None,       (template signal for matched filtering)
        "epsilon": 0.01         (radius of hyperdisc for CVX problem)
        "length": len(svector) 
        Optional arguments - some are required for different functionality


    Returns
    -------
    CAOptimise class instance

    Raises
    ------
    KeyError
        If no measurement transform has been specified.
    """


    """
    Computes the duals of an input basis <rhob>.
    """
    # get system dimension
    dim = np.shape(rhob)[-1]

    # define pauli basis
    pauli = opbasis(N)
    # reshape pauli array
    basis_flat = np.transpose(np.reshape(pauli, [dim**2]*2, order='C'))
    # initialise coefficent array
    coeffs = np.empty((2**(2*N), 2**(2*N)), dtype=float)

    # compute basis coefficients (will need to reformulate with pyconv + constraints)
    for i in range(int(dim**2)):
        rho = np.reshape(rhob[i, :, :], [dim**2, 1])

        # could compute analytically but I want this to be generalisable
        coeffs[i, :] = np.real(np.squeeze(np.linalg.solve(basis_flat, rho)))

        # check that reconstructed matrices are within tolerance
        rhor = np.zeros(([dim, dim]), dtype=np.complex128)

        for j in range(dim**2):
            rhor += coeffs[i, j] * np.reshape(basis_flat[:, j], [dim, dim])
        assert np.allclose(rhor, np.reshape(rho, [
                           dim, dim]), atol=1e-9), "Reconstructed array not within tolerance of target: Aborting"

    # find the inverse of the coefficient matrix
    F = np.conjugate(np.transpose((np.linalg.inv(coeffs))))

    # compute the duals to rhob
    duals = np.zeros_like(rhob)
    for i in range(dim**2):
        for j in range(dim**2):
            duals[i, :, :] += 0.5*F[i, j] * \
                np.reshape(basis_flat[:, j], [dim, dim])
                
    return duals


def dualp_tomography(dual, rhop, estimate=False):

    """
    Class definition that handles signal reconstruction of a transformed input signal given
    the measurement basis. Able to perform standard compressed sensing and compressive 
    matched filter processing. 

    Parameters
    ----------
    svector : one dimensional numpy array
        the sensing or measurement vector y such that y = Ax where 
        x is the signal to reconstruct.

    transform : m*n array where m is len(svector) and n is the 
                dimension of the signal to reconstruct.

    verbose : boolean 
        Whether to print progress reports as reconstruction is performed.

    kwargs : optional
        "template": None,       (template signal for matched filtering)
        "epsilon": 0.01         (radius of hyperdisc for CVX problem)
        "length": len(svector) 
        Optional arguments - some are required for different functionality


    Returns
    -------
    CAOptimise class instance

    Raises
    ------
    KeyError
        If no measurement transform has been specified.
    """


    """
    Perform process tomography using the dual approach and outputs the Choi state of the process
    Inputs - set of duals to the input states and the output states.
    dual = ndarray(d^2, d, d)
    outputs = ndarray(d^2, d, d)
    """
    # get dimensionality of system
    dim = np.shape(rhop)[-1]

    # initialise Aform and Bform
    Aform = np.zeros((dim**2, dim**2), dtype=np.complex128)
    Bform = np.zeros((dim**2, dim**2), dtype=np.complex128)

    # compute A form in terms of the duals
    for i in range(dim**2):
        Aform += np.outer(rhop[i, :, :], np.conjugate(dual[i, :, :]))

    # compute B form in terms of the duals
    for j in range(dim**2):
        Bform += np.kron(rhop[j, :, :], np.conjugate(dual[j, :, :]))

    # force valid CP map
    if estimate:
        # setup estimate problem
        choi = cvxpy.Variable((dim**2,dim**2), hermitian=True)
        psd_constraint = choi >> 0
        partial_constraint1 = partial_trace2(choi, [0]) == np.eye(dim)
        partial_constraint2 = partial_trace2(choi, [1]) == np.eye(dim)
        trace_constraint = cvxpy.trace(choi) == dim
        constraints = [psd_constraint, trace_constraint, partial_constraint1, partial_constraint2]

        prob = cvxpy.Problem(cvxpy.Minimize(cvxpy.norm(choi - Bform,2)), constraints)

        # minimise trace distance between channels 
        prob.solve()

        Aform = AB_shuffle(choi.value)

        return Aform, choi.value

    # ought to add a check that index transformation of A<->B gives the same as above
    return Aform, Bform


def IA_gen(N):

    """
    Class definition that handles signal reconstruction of a transformed input signal given
    the measurement basis. Able to perform standard compressed sensing and compressive 
    matched filter processing. 

    Parameters
    ----------
    svector : one dimensional numpy array
        the sensing or measurement vector y such that y = Ax where 
        x is the signal to reconstruct.

    transform : m*n array where m is len(svector) and n is the 
                dimension of the signal to reconstruct.

    verbose : boolean 
        Whether to print progress reports as reconstruction is performed.

    kwargs : optional
        "template": None,       (template signal for matched filtering)
        "epsilon": 0.01         (radius of hyperdisc for CVX problem)
        "length": len(svector) 
        Optional arguments - some are required for different functionality


    Returns
    -------
    CAOptimise class instance

    Raises
    ------
    KeyError
        If no measurement transform has been specified.
    """


    """
    Computes the A/B--form of the identity channel
    """
    # define a basis set (and hence the output state)
    rhop = rho_gen(N=N)
    # define duals
    duals = dual_gen(rhop, N=N)
    A, B = dualp_tomography(duals, rhop)
    return A, B

def B_apply(choi, rho):
    """
    Applies a map phi in choi form to in anput state rho.
    """

    rhot = np.kron(eye_like(rho), np.transpose(rho))

    return partial_trace2(rhot @ choi, [1])


def UA_gen(U):

    """
    Class definition that handles signal reconstruction of a transformed input signal given
    the measurement basis. Able to perform standard compressed sensing and compressive 
    matched filter processing. 

    Parameters
    ----------
    svector : one dimensional numpy array
        the sensing or measurement vector y such that y = Ax where 
        x is the signal to reconstruct.

    transform : m*n array where m is len(svector) and n is the 
                dimension of the signal to reconstruct.

    verbose : boolean 
        Whether to print progress reports as reconstruction is performed.

    kwargs : optional
        "template": None,       (template signal for matched filtering)
        "epsilon": 0.01         (radius of hyperdisc for CVX problem)
        "length": len(svector) 
        Optional arguments - some are required for different functionality


    Returns
    -------
    CAOptimise class instance

    Raises
    ------
    KeyError
        If no measurement transform has been specified.
    """


    """
    Performs process tomography on an input unitary U
    """
    # get dimension of operator space
    N = int(np.round(np.log2(len(U))))
    # generate spanning set
    rhob = rho_gen(N=N)
    # compute duals
    duals = dual_gen(rhob, N=N)
    # compute process effect on spanning set
    rhop = np.copy(rhob)

    # evolve spanning set under unitary
    for i, rho in enumerate(rhob):
        rhop[i, :, :] = U @ rho @ np.conjugate(np.transpose(U))

    # compute A and B form of map
    A, B = dualp_tomography(duals, rhop)
    return A, B


def AB_join(A1, A2):

    """
    Class definition that handles signal reconstruction of a transformed input signal given
    the measurement basis. Able to perform standard compressed sensing and compressive 
    matched filter processing. 

    Parameters
    ----------
    svector : one dimensional numpy array
        the sensing or measurement vector y such that y = Ax where 
        x is the signal to reconstruct.

    transform : m*n array where m is len(svector) and n is the 
                dimension of the signal to reconstruct.

    verbose : boolean 
        Whether to print progress reports as reconstruction is performed.

    kwargs : optional
        "template": None,       (template signal for matched filtering)
        "epsilon": 0.01         (radius of hyperdisc for CVX problem)
        "length": len(svector) 
        Optional arguments - some are required for different functionality


    Returns
    -------
    CAOptimise class instance

    Raises
    ------
    KeyError
        If no measurement transform has been specified.
    """


    """
    Joins two A (or B? - investigate) form maps into a single operator on the joint Hilbert space. 
    Remap requirements comes from ordering mismatch between state tensor product and A-form tensor product.
    """
    # compute tensor product and get output dimension
    joint = np.kron(A1, A2)
    dim = len(joint)

    # explicitly compute sub--process dimensions
    A1_dim = len(A1)
    A2_dim = len(A2)

    # local system dimensions
    A1_sdim = int(round(np.sqrt(A1_dim)))
    A2_sdim = int(round(np.sqrt(A2_dim)))

    # construct subsystem remap
    cshape = [A1_sdim, A1_sdim, A2_sdim, A2_sdim]*2

    return np.reshape(np.transpose(np.reshape(joint, cshape), [0, 2, 1, 3, 4, 6, 5, 7]), [dim, dim])


def log_base(a, base=np.e):

    """
    Class definition that handles signal reconstruction of a transformed input signal given
    the measurement basis. Able to perform standard compressed sensing and compressive 
    matched filter processing. 

    Parameters
    ----------
    svector : one dimensional numpy array
        the sensing or measurement vector y such that y = Ax where 
        x is the signal to reconstruct.

    transform : m*n array where m is len(svector) and n is the 
                dimension of the signal to reconstruct.

    verbose : boolean 
        Whether to print progress reports as reconstruction is performed.

    kwargs : optional
        "template": None,       (template signal for matched filtering)
        "epsilon": 0.01         (radius of hyperdisc for CVX problem)
        "length": len(svector) 
        Optional arguments - some are required for different functionality


    Returns
    -------
    CAOptimise class instance

    Raises
    ------
    KeyError
        If no measurement transform has been specified.
    """


    """
    Computes the log of a to given base
    """
    return np.log(a)/np.log(base)


def subsystem_num(M, dim=2):

    """
    Class definition that handles signal reconstruction of a transformed input signal given
    the measurement basis. Able to perform standard compressed sensing and compressive 
    matched filter processing. 

    Parameters
    ----------
    svector : one dimensional numpy array
        the sensing or measurement vector y such that y = Ax where 
        x is the signal to reconstruct.

    transform : m*n array where m is len(svector) and n is the 
                dimension of the signal to reconstruct.

    verbose : boolean 
        Whether to print progress reports as reconstruction is performed.

    kwargs : optional
        "template": None,       (template signal for matched filtering)
        "epsilon": 0.01         (radius of hyperdisc for CVX problem)
        "length": len(svector) 
        Optional arguments - some are required for different functionality


    Returns
    -------
    CAOptimise class instance

    Raises
    ------
    KeyError
        If no measurement transform has been specified.
    """


    """
    Given a square matrix M and subsystem dimension dim, computes the number
    of subsystems that make up matrix space. Assumes uniform dimension.
    """
    # compute log of arbitrary dimension
    return int(round(log_base(np.shape(M)[0], dim)))


def tracenorm(m):

    """
    Class definition that handles signal reconstruction of a transformed input signal given
    the measurement basis. Able to perform standard compressed sensing and compressive 
    matched filter processing. 

    Parameters
    ----------
    svector : one dimensional numpy array
        the sensing or measurement vector y such that y = Ax where 
        x is the signal to reconstruct.

    transform : m*n array where m is len(svector) and n is the 
                dimension of the signal to reconstruct.

    verbose : boolean 
        Whether to print progress reports as reconstruction is performed.

    kwargs : optional
        "template": None,       (template signal for matched filtering)
        "epsilon": 0.01         (radius of hyperdisc for CVX problem)
        "length": len(svector) 
        Optional arguments - some are required for different functionality


    Returns
    -------
    CAOptimise class instance

    Raises
    ------
    KeyError
        If no measurement transform has been specified.
    """


    import numpy.linalg
    return np.sum(np.abs(numpy.linalg.eigh(m)[0]))


def AB_shuffle(form):

    """
    Class definition that handles signal reconstruction of a transformed input signal given
    the measurement basis. Able to perform standard compressed sensing and compressive 
    matched filter processing. 

    Parameters
    ----------
    svector : one dimensional numpy array
        the sensing or measurement vector y such that y = Ax where 
        x is the signal to reconstruct.

    transform : m*n array where m is len(svector) and n is the 
                dimension of the signal to reconstruct.

    verbose : boolean 
        Whether to print progress reports as reconstruction is performed.

    kwargs : optional
        "template": None,       (template signal for matched filtering)
        "epsilon": 0.01         (radius of hyperdisc for CVX problem)
        "length": len(svector) 
        Optional arguments - some are required for different functionality


    Returns
    -------
    CAOptimise class instance

    Raises
    ------
    KeyError
        If no measurement transform has been specified.
    """


    """
    Switches between the A/B form of a map. Assumes same input/output dimensions.
    """
    # get dimension of map
    dim = len(form)
    # get subsystem dimension
    sub_dim = int(round(np.sqrt(dim)))
    # reshape subsystems into 4-tensor
    return np.reshape(np.transpose(np.reshape(form, [sub_dim]*4), (0, 2, 1, 3)), (dim, dim))


def random_U(dim, num=1):

    """
    Class definition that handles signal reconstruction of a transformed input signal given
    the measurement basis. Able to perform standard compressed sensing and compressive 
    matched filter processing. 

    Parameters
    ----------
    svector : one dimensional numpy array
        the sensing or measurement vector y such that y = Ax where 
        x is the signal to reconstruct.

    transform : m*n array where m is len(svector) and n is the 
                dimension of the signal to reconstruct.

    verbose : boolean 
        Whether to print progress reports as reconstruction is performed.

    kwargs : optional
        "template": None,       (template signal for matched filtering)
        "epsilon": 0.01         (radius of hyperdisc for CVX problem)
        "length": len(svector) 
        Optional arguments - some are required for different functionality


    Returns
    -------
    CAOptimise class instance

    Raises
    ------
    KeyError
        If no measurement transform has been specified.
    """


    """
    Generate num random unitaries on N qubits using standard sampling strategy
    """
    # preallocate U array
    Us = np.zeros((num, dim,dim), dtype=np.complex128)

    # generate unitaries using naive method
    for i in range(0,num):
        # generate a random complex matrix (yes I know I could do this in one go rather than iterate)
        U = np.random.rand(dim,dim) + 1j*np.random.rand(dim,dim) 

        # QR factorisation
        [Q,R] = np.linalg.qr(U/np.sqrt(2))
        R = np.diag(np.diag(R)/np.abs(np.diag(R)))
        
        # compute the unitary
        Us[i,:,:] = Q @ R

    return Us


def random_O(dim, num=1):

    """
    Class definition that handles signal reconstruction of a transformed input signal given
    the measurement basis. Able to perform standard compressed sensing and compressive 
    matched filter processing. 

    Parameters
    ----------
    svector : one dimensional numpy array
        the sensing or measurement vector y such that y = Ax where 
        x is the signal to reconstruct.

    transform : m*n array where m is len(svector) and n is the 
                dimension of the signal to reconstruct.

    verbose : boolean 
        Whether to print progress reports as reconstruction is performed.

    kwargs : optional
        "template": None,       (template signal for matched filtering)
        "epsilon": 0.01         (radius of hyperdisc for CVX problem)
        "length": len(svector) 
        Optional arguments - some are required for different functionality


    Returns
    -------
    CAOptimise class instance

    Raises
    ------
    KeyError
        If no measurement transform has been specified.
    """


    """
    Generate num random unitaries on N qubits using standard sampling strategy
    """
    # preallocate U array
    Us = np.zeros((num, dim,dim))

    # generate unitaries using naive method
    for i in range(0,num):
        # generate a random complex matrix (yes I know I could do this in one go rather than iterate)
        U = np.random.rand(dim,dim) 

        # QR factorisation
        [Q,R] = np.linalg.qr(U/np.sqrt(2))
        R = np.diag(np.diag(R)/np.abs(np.diag(R)))
        
        # compute the unitary
        Us[i,:,:] = Q @ R

    return Us



def haar_sample(dim=2, num=10, pure=True):

    """
    Class definition that handles signal reconstruction of a transformed input signal given
    the measurement basis. Able to perform standard compressed sensing and compressive 
    matched filter processing. 

    Parameters
    ----------
    svector : one dimensional numpy array
        the sensing or measurement vector y such that y = Ax where 
        x is the signal to reconstruct.

    transform : m*n array where m is len(svector) and n is the 
                dimension of the signal to reconstruct.

    verbose : boolean 
        Whether to print progress reports as reconstruction is performed.

    kwargs : optional
        "template": None,       (template signal for matched filtering)
        "epsilon": 0.01         (radius of hyperdisc for CVX problem)
        "length": len(svector) 
        Optional arguments - some are required for different functionality


    Returns
    -------
    CAOptimise class instance

    Raises
    ------
    KeyError
        If no measurement transform has been specified.
    """


    """
    Generates <num> quantum states of dimension 2**N from Haar distribution.
    """
    # generate random complex arrays
    states = np.random.uniform(low=-1, high=1, size=(num, dim, dim)) + \
        np.random.uniform(low=-1, high=1, size=(num, dim, dim))*1j
    for i in range(num):
        # compute Hilbert Schmidt norm
        A2 = np.sqrt(np.trace(dagger(states[i, :, :]) @ states[i, :, :]))
        # normalise
        states[i, :, :] /= A2
        # compute random density matrix
        states[i, :, :] = dagger(states[i, :, :]) @ states[i, :, :]

        if pure:
            # TODO: did the ordering from eigh change at some point?
            _,state = np.linalg.eigh(states[i,:,:])
            state = np.asarray(state[:,-1]).reshape(dim,1)
            states[i, :, :] = np.kron(dagger(state), state)

    return states


def rhoplot(rho, axislabels=None, save=False):

    """
    Class definition that handles signal reconstruction of a transformed input signal given
    the measurement basis. Able to perform standard compressed sensing and compressive 
    matched filter processing. 

    Parameters
    ----------
    svector : one dimensional numpy array
        the sensing or measurement vector y such that y = Ax where 
        x is the signal to reconstruct.

    transform : m*n array where m is len(svector) and n is the 
                dimension of the signal to reconstruct.

    verbose : boolean 
        Whether to print progress reports as reconstruction is performed.

    kwargs : optional
        "template": None,       (template signal for matched filtering)
        "epsilon": 0.01         (radius of hyperdisc for CVX problem)
        "length": len(svector) 
        Optional arguments - some are required for different functionality


    Returns
    -------
    CAOptimise class instance

    Raises
    ------
    KeyError
        If no measurement transform has been specified.
    """


    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm

    # extract real and imaginary components of density matrix
    realrho = np.real(rho)
    imagrho = np.imag(rho)

    # instantiate new figure
    fig = plt.gcf()
    fig.canvas.set_window_title('Density Plot')
    #rax = Axes3D(fig)
    rax = fig.add_subplot(121, projection='3d')
    iax = fig.add_subplot(122, projection='3d')

    # set titles
    rax.title.set_text('Real$(\\rho)$')
    iax.title.set_text('Imag$(\\rho)$')
    # apply custom labelling
    if axislabels is not None:
        rax.set_xticklabels(axislabels)
        rax.set_yticklabels(axislabels)
        iax.set_xticklabels(axislabels)
        iax.set_yticklabels(axislabels)

    # dimension of space
    dim = np.shape(realrho)[0]
    # create indexing vectors
    x, y = np.meshgrid(range(0, dim), range(0, dim), indexing='ij')
    x = x.flatten('F')
    y = y.flatten('F')
    z = np.zeros_like(x)

    # create bar widths
    dx = 0.5*np.ones_like(z)
    dy = dx.copy()
    dzr = realrho.flatten()
    dzi = imagrho.flatten()

    # compute colour matrix for real matrix and set axes bounds
    norm = colors.Normalize(dzr.min(), dzr.max())
    rcolours = cm.BuGn(norm(dzr))
    rax.set_zlim3d([dzr.min(), 1.5*np.max(dzr)])
    iax.set_zlim3d([dzi.min(), 1.5*np.max(dzr)])

    inorm = colors.Normalize(dzi.min(), dzi.max())
    icolours = cm.jet(inorm(dzi))

    # plot image
    rax.bar3d(x, y, z, dx, dy, dzr, color=rcolours)
    iax.bar3d(x, y, z, dx, dy, dzi, color=icolours)
    #plt.ticklabel_format(style='sci', axis='z', scilimits=(0, 0))
    plt.show()


def vecstate(state):

    """
    Class definition that handles signal reconstruction of a transformed input signal given
    the measurement basis. Able to perform standard compressed sensing and compressive 
    matched filter processing. 

    Parameters
    ----------
    svector : one dimensional numpy array
        the sensing or measurement vector y such that y = Ax where 
        x is the signal to reconstruct.

    transform : m*n array where m is len(svector) and n is the 
                dimension of the signal to reconstruct.

    verbose : boolean 
        Whether to print progress reports as reconstruction is performed.

    kwargs : optional
        "template": None,       (template signal for matched filtering)
        "epsilon": 0.01         (radius of hyperdisc for CVX problem)
        "length": len(svector) 
        Optional arguments - some are required for different functionality


    Returns
    -------
    CAOptimise class instance

    Raises
    ------
    KeyError
        If no measurement transform has been specified.
    """


    """
    Vectorises state using the computational basis or devectorises.
    """
    # get dimension of first axis
    dim = np.shape(state)[0]

    if dim == np.shape(state)[1]:
        return np.reshape(state, [dim**2, 1])
    else:
        sdim = int(round(np.sqrt(dim)))
        return np.reshape(state, [sdim, sdim])


def subsyspermute(rho, perm, dims):

    """
    Class definition that handles signal reconstruction of a transformed input signal given
    the measurement basis. Able to perform standard compressed sensing and compressive 
    matched filter processing. 

    Parameters
    ----------
    svector : one dimensional numpy array
        the sensing or measurement vector y such that y = Ax where 
        x is the signal to reconstruct.

    transform : m*n array where m is len(svector) and n is the 
                dimension of the signal to reconstruct.

    verbose : boolean 
        Whether to print progress reports as reconstruction is performed.

    kwargs : optional
        "template": None,       (template signal for matched filtering)
        "epsilon": 0.01         (radius of hyperdisc for CVX problem)
        "length": len(svector) 
        Optional arguments - some are required for different functionality


    Returns
    -------
    CAOptimise class instance

    Raises
    ------
    KeyError
        If no measurement transform has been specified.
    """


    # get dimensions of system
    d = np.shape(rho)
    # get number of subsystems
    sys = len(dims)
    # perform permutation
    perm = [(sys - 1 - i) for i in perm[-1::-1]]
    perm = listflatten([perm, [sys + j for j in perm]])
    return np.transpose(rho.reshape(dims[-1::-1]*2), perm).reshape(d)


def qre(rho, gamma):

    """
    Class definition that handles signal reconstruction of a transformed input signal given
    the measurement basis. Able to perform standard compressed sensing and compressive 
    matched filter processing. 

    Parameters
    ----------
    svector : one dimensional numpy array
        the sensing or measurement vector y such that y = Ax where 
        x is the signal to reconstruct.

    transform : m*n array where m is len(svector) and n is the 
                dimension of the signal to reconstruct.

    verbose : boolean 
        Whether to print progress reports as reconstruction is performed.

    kwargs : optional
        "template": None,       (template signal for matched filtering)
        "epsilon": 0.01         (radius of hyperdisc for CVX problem)
        "length": len(svector) 
        Optional arguments - some are required for different functionality


    Returns
    -------
    CAOptimise class instance

    Raises
    ------
    KeyError
        If no measurement transform has been specified.
    """


    """
    computes the quantum relative entropy between two states rho and gamma
    """
    return np.trace(np.dot(rho, (logm(rho) - logm(gamma))))


def kolmogorov(rho, gamma):

    """
    Class definition that handles signal reconstruction of a transformed input signal given
    the measurement basis. Able to perform standard compressed sensing and compressive 
    matched filter processing. 

    Parameters
    ----------
    svector : one dimensional numpy array
        the sensing or measurement vector y such that y = Ax where 
        x is the signal to reconstruct.

    transform : m*n array where m is len(svector) and n is the 
                dimension of the signal to reconstruct.

    verbose : boolean 
        Whether to print progress reports as reconstruction is performed.

    kwargs : optional
        "template": None,       (template signal for matched filtering)
        "epsilon": 0.01         (radius of hyperdisc for CVX problem)
        "length": len(svector) 
        Optional arguments - some are required for different functionality


    Returns
    -------
    CAOptimise class instance

    Raises
    ------
    KeyError
        If no measurement transform has been specified.
    """


    """
    Computes the trace or Kolmogorov distance between two quantum states
    """
    return np.trace(abs(rho-gamma))/2


def qfid(rho, gamma):

    """
    Class definition that handles signal reconstruction of a transformed input signal given
    the measurement basis. Able to perform standard compressed sensing and compressive 
    matched filter processing. 

    Parameters
    ----------
    svector : one dimensional numpy array
        the sensing or measurement vector y such that y = Ax where 
        x is the signal to reconstruct.

    transform : m*n array where m is len(svector) and n is the 
                dimension of the signal to reconstruct.

    verbose : boolean 
        Whether to print progress reports as reconstruction is performed.

    kwargs : optional
        "template": None,       (template signal for matched filtering)
        "epsilon": 0.01         (radius of hyperdisc for CVX problem)
        "length": len(svector) 
        Optional arguments - some are required for different functionality


    Returns
    -------
    CAOptimise class instance

    Raises
    ------
    KeyError
        If no measurement transform has been specified.
    """


    """
    Computes the quantum fidelity between two quantum states (not a metric)
    """
    print(sqrtm(rho))
    return (np.trace(sqrtm(sqrtm(rho)*gamma*sqrtm(rho))))**2


def bures(rho,gamma):

    """
    Class definition that handles signal reconstruction of a transformed input signal given
    the measurement basis. Able to perform standard compressed sensing and compressive 
    matched filter processing. 

    Parameters
    ----------
    svector : one dimensional numpy array
        the sensing or measurement vector y such that y = Ax where 
        x is the signal to reconstruct.

    transform : m*n array where m is len(svector) and n is the 
                dimension of the signal to reconstruct.

    verbose : boolean 
        Whether to print progress reports as reconstruction is performed.

    kwargs : optional
        "template": None,       (template signal for matched filtering)
        "epsilon": 0.01         (radius of hyperdisc for CVX problem)
        "length": len(svector) 
        Optional arguments - some are required for different functionality


    Returns
    -------
    CAOptimise class instance

    Raises
    ------
    KeyError
        If no measurement transform has been specified.
    """


    """
    Computes Bures angle between two states
    """
    return np.arccos(np.clip(np.sqrt(np.trace(sqrtm(sqrtm(rho) @ gamma @ sqrtm(rho)))**2), 0.0,1.0))


def helstrom(rho, gamma):

    """
    Class definition that handles signal reconstruction of a transformed input signal given
    the measurement basis. Able to perform standard compressed sensing and compressive 
    matched filter processing. 

    Parameters
    ----------
    svector : one dimensional numpy array
        the sensing or measurement vector y such that y = Ax where 
        x is the signal to reconstruct.

    transform : m*n array where m is len(svector) and n is the 
                dimension of the signal to reconstruct.

    verbose : boolean 
        Whether to print progress reports as reconstruction is performed.

    kwargs : optional
        "template": None,       (template signal for matched filtering)
        "epsilon": 0.01         (radius of hyperdisc for CVX problem)
        "length": len(svector) 
        Optional arguments - some are required for different functionality


    Returns
    -------
    CAOptimise class instance

    Raises
    ------
    KeyError
        If no measurement transform has been specified.
    """


    """
    Computes the Helstrom distance between two quantum states
    """
    return sqrtm(2*(1-sqrtm(qfid(rho, gamma))))


def isdensity(rho):

    """
    Class definition that handles signal reconstruction of a transformed input signal given
    the measurement basis. Able to perform standard compressed sensing and compressive 
    matched filter processing. 

    Parameters
    ----------
    svector : one dimensional numpy array
        the sensing or measurement vector y such that y = Ax where 
        x is the signal to reconstruct.

    transform : m*n array where m is len(svector) and n is the 
                dimension of the signal to reconstruct.

    verbose : boolean 
        Whether to print progress reports as reconstruction is performed.

    kwargs : optional
        "template": None,       (template signal for matched filtering)
        "epsilon": 0.01         (radius of hyperdisc for CVX problem)
        "length": len(svector) 
        Optional arguments - some are required for different functionality


    Returns
    -------
    CAOptimise class instance

    Raises
    ------
    KeyError
        If no measurement transform has been specified.
    """


    """
    Checks if an input matrix is a valid density operator
    """
    res = True
    res &= np.all(np.isclose(rho - dagger(rho), 0.0))  # symmetric
    res &= np.all(np.linalg.eigvals(rho) >= 0)         # positive semidefinite
    res &= np.isclose(np.trace(rho), 1.0)              # trace one
    return res

def eye_like(m):

    """
    Class definition that handles signal reconstruction of a transformed input signal given
    the measurement basis. Able to perform standard compressed sensing and compressive 
    matched filter processing. 

    Parameters
    ----------
    svector : one dimensional numpy array
        the sensing or measurement vector y such that y = Ax where 
        x is the signal to reconstruct.

    transform : m*n array where m is len(svector) and n is the 
                dimension of the signal to reconstruct.

    verbose : boolean 
        Whether to print progress reports as reconstruction is performed.

    kwargs : optional
        "template": None,       (template signal for matched filtering)
        "epsilon": 0.01         (radius of hyperdisc for CVX problem)
        "length": len(svector) 
        Optional arguments - some are required for different functionality


    Returns
    -------
    CAOptimise class instance

    Raises
    ------
    KeyError
        If no measurement transform has been specified.
    """


    """
    Returns identity matrix with same dims as square matrix m.
    """
    dim = np.shape(m)[0]
    return np.eye(dim)


def mem_check(dims, type=np.float64):

    """
    Class definition that handles signal reconstruction of a transformed input signal given
    the measurement basis. Able to perform standard compressed sensing and compressive 
    matched filter processing. 

    Parameters
    ----------
    svector : one dimensional numpy array
        the sensing or measurement vector y such that y = Ax where 
        x is the signal to reconstruct.

    transform : m*n array where m is len(svector) and n is the 
                dimension of the signal to reconstruct.

    verbose : boolean 
        Whether to print progress reports as reconstruction is performed.

    kwargs : optional
        "template": None,       (template signal for matched filtering)
        "epsilon": 0.01         (radius of hyperdisc for CVX problem)
        "length": len(svector) 
        Optional arguments - some are required for different functionality


    Returns
    -------
    CAOptimise class instance

    Raises
    ------
    KeyError
        If no measurement transform has been specified.
    """


    """
    Returns the number of bits an array with dimensions <dims> and datatype <type> will require.
    """
    return np.prod(dims)*np.finfo(type).bits


def povm_gen(N=1):

    """
    Class definition that handles signal reconstruction of a transformed input signal given
    the measurement basis. Able to perform standard compressed sensing and compressive 
    matched filter processing. 

    Parameters
    ----------
    svector : one dimensional numpy array
        the sensing or measurement vector y such that y = Ax where 
        x is the signal to reconstruct.

    transform : m*n array where m is len(svector) and n is the 
                dimension of the signal to reconstruct.

    verbose : boolean 
        Whether to print progress reports as reconstruction is performed.

    kwargs : optional
        "template": None,       (template signal for matched filtering)
        "epsilon": 0.01         (radius of hyperdisc for CVX problem)
        "length": len(svector) 
        Optional arguments - some are required for different functionality


    Returns
    -------
    CAOptimise class instance

    Raises
    ------
    KeyError
        If no measurement transform has been specified.
    """


    """
    Generates a simple POVM for N qubits. Corresponds to a spanning set of the NxN Hermitian matrices.
    """
    # explicitly calculate for later modifcation to generalised subsystems
    dim = 2**N
    # initialise set array
    povm = np.empty((dim**2, dim, dim), dtype=np.complex128)
    # Set of N matrices with one 1 on the diagonal union with N(N−1)/2 [1,+i/-i] on off--diagonal.
    # define N=1 spanning set and build from there
    povmbase = np.empty((4, 2, 2), dtype=np.complex128)
    alpha = (np.sqrt(3)*1j-1)/2
    alphac = np.conjugate(alpha)
    povmbase[0, :, :] = np.asarray([[1, 0], [0, 0]])/2
    povmbase[1, :, :] = np.asarray([[1, np.sqrt(2)], [np.sqrt(2), 2]])/6
    povmbase[2, :, :] = np.asarray(
        [[1, np.sqrt(2)*alpha], [np.sqrt(2)*alphac, 2]])/6
    povmbase[3, :, :] = np.asarray(
        [[1, np.sqrt(2)*alphac], [np.sqrt(2)*alpha, 2]])/6

    # since larger Hilbert spaces are simply chained tensor products, so too will the joint basis set 
    # be the tensor product of the constituent spaces.
    # generate permutation list
    combs = product(['0', '1', '2', '3'], repeat=N)
    for i, comb in enumerate(combs):
        pvm = 1.0
        for j in comb:
            pvm = np.kron(pvm, povmbase[int(j), :, :])

        povm[i, :, :] = pvm
    return povm

def randCP_gen(d):
    """
    Generates a random CPTP map, pretty clunky functionallity right now. I'm
    *sure* I'll fix that up sometime in the next 5 years. 

    Parameters
    ----------
    d : int
        Dimension of the generated POVM such that each element of the set
        is a d-dimensional square matrix.

    Returns
    -------
    array_like
        d x d numpy complex array with first dimension indexing the POVM element.

    Raises
    ------
    AssertionError
        If dimension is non-integer or negative.

    """
    assert d>1 and type(d) is int, "Dimension is either less than one or non-int"

    # generate a single random density operator
    rho = np.squeeze(haar_sample(N=int(round(np.log2(d))), num=1))
    
    
    




def randpovm_gen(d, ):
    """
    Generates a random POVM set using the method described in https://arxiv.org/pdf/1902.04751
    

    Parameters
    ----------
    d : int
        Dimension of the generated POVM such that each element of the set
        is a d-dimensional square matrix.

    verbose : boolean 
        Whether to print progress reports as reconstruction is performed.

    Returns
    -------
    array_like
        d^2 x d x d numpy complex array with first dimension indexing the POVM element.

    Raises
    ------
    AssertionError
        If dimension is non-integer or negative.

    """



class ControlSpan():

    """
    Class definition that handles signal reconstruction of a transformed input signal given
    the measurement basis. Able to perform standard compressed sensing and compressive 
    matched filter processing. 

    Parameters
    ----------
    svector : one dimensional numpy array
        the sensing or measurement vector y such that y = Ax where 
        x is the signal to reconstruct.

    transform : m*n array where m is len(svector) and n is the 
                dimension of the signal to reconstruct.

    verbose : boolean 
        Whether to print progress reports as reconstruction is performed.

    kwargs : optional
        "template": None,       (template signal for matched filtering)
        "epsilon": 0.01         (radius of hyperdisc for CVX problem)
        "length": len(svector) 
        Optional arguments - some are required for different functionality


    Returns
    -------
    CAOptimise class instance

    Raises
    ------
    KeyError
        If no measurement transform has been specified.
    """


    """
    Computes a spanning set of control operations (the space B(B(H_d)) with d=2**N). Assumes qubits for the moment.
    Must be done as an iterator else the memory requirements are simply too severe for even short processes. 
    """
    def __init__(self, N=1, k=1):

        # number of time steps in process
        self.k = k
        # number of qubits controls will be applied too
        self.N = N
        # compute spanning set of density matrices
        self.rhob = rho_gen(N=N)
        # compute POVM
        self.povm = povm_gen(N=N)


    def __iter__(self):
        """
        initialise control sequence number and loop constants
        """ 
        self.cseq_num = 0
        # preallocate current control sequence
        self.control_sequence = np.empty((self.k, 2**(2*self.N),2**(2*self.N)), dtype=np.complex128)
        # current rho to iterate over
        self.rho_num = [0]*self.k
        # current povm to iterate over
        self.povm_num = [0]*self.k
        
        return self


    def __next__(self):
        """
        Iteratively generate the control maps for a k--step process tensor
        """
        
        # compute control sequence
        if self.rho_num[-1] < len(self.rhob):

            # catch start sequence case
            if self.cseq_num == 0:
                for i in range(0, self.k):
                    rho_sel = self.rho_num[i]
                    povm_sel = self.povm_num[i]
                    self.control_sequence[i, :,:] = np.kron(self.rhob[rho_sel,:,:], self.povm[povm_sel,:,:])
                    self.cseq_num += 1
                    return self.control_sequence

            # perform incrementation of counters
            inc_flag = True
            inc_ind = 0
            while inc_flag:
                # check if incrementation overflows
                if self.povm_num[inc_ind]+1>=len(self.povm):
                    inc_ind += 1

                    # check if rhob needs to be incremented
                    if inc_ind >= len(self.povm_num):
                        inc_ind = 0
                        while inc_flag:
                            if self.rho_num[inc_ind]+1 >= len(self.rhob):
                                inc_ind += 1
                                # exit if we have are at the end of the final iter
                                if inc_ind == len(self.rho_num):
                                    raise StopIteration

                            else:
                                self.rho_num[inc_ind] += 1
                                self.rho_num[:inc_ind] = [0]*inc_ind
                                self.povm_num = [0]*len(self.povm_num)
                                inc_flag = False

                else:
                    self.povm_num[inc_ind] += 1
                    self.povm_num[:inc_ind] = [0]*inc_ind
                    inc_flag = False

            # iterate over number of time steps
            for i in range(0, self.k):
                rho_sel = self.rho_num[i]
                povm_sel = self.povm_num[i]
                self.control_sequence[i, :,:] = np.kron(self.rhob[rho_sel,:,:], self.povm[povm_sel,:,:])

            # iterate loop couinter
            self.cseq_num += 1
            return self.control_sequence

        else:
            raise StopIteration


def Universal_U():

    """
    Class definition that handles signal reconstruction of a transformed input signal given
    the measurement basis. Able to perform standard compressed sensing and compressive 
    matched filter processing. 

    Parameters
    ----------
    svector : one dimensional numpy array
        the sensing or measurement vector y such that y = Ax where 
        x is the signal to reconstruct.

    transform : m*n array where m is len(svector) and n is the 
                dimension of the signal to reconstruct.

    verbose : boolean 
        Whether to print progress reports as reconstruction is performed.

    kwargs : optional
        "template": None,       (template signal for matched filtering)
        "epsilon": 0.01         (radius of hyperdisc for CVX problem)
        "length": len(svector) 
        Optional arguments - some are required for different functionality


    Returns
    -------
    CAOptimise class instance

    Raises
    ------
    KeyError
        If no measurement transform has been specified.
    """


    """
    Generates an interesting unitary drawing from the universal gate set
    """

    # base unitary
    U = np.eye(2**5)

    # hadamard combo to get started
    H = np.kron(kronjob([ops['h']],[0,0]), np.eye(8))
    U = H @ U

    # controlled nots
    U = kronjob([ops['cx'], ops['id']], [0,0,1]) @ U

    # some local operators
    U = kronjob([ops['id'],ops['t'],ops['s']],[2,1,2,0,0]) @ U

    # some more controlled nots
    U = kronjob([ops['cx'], ops['id']], [1,0,1,1]) @ U

    # hit some phase gates
    U = kronjob([ops['s'], ops['id']], [1,1,0,1,1]) @ U

    # finally some T gate action
    U = kronjob([ops['id'],ops['t'],ops['s']],[1,1,1,2,0])

    return U


class ProcessTensor():

    """
    Class definition that handles signal reconstruction of a transformed input signal given
    the measurement basis. Able to perform standard compressed sensing and compressive 
    matched filter processing. 

    Parameters
    ----------
    svector : one dimensional numpy array
        the sensing or measurement vector y such that y = Ax where 
        x is the signal to reconstruct.

    transform : m*n array where m is len(svector) and n is the 
                dimension of the signal to reconstruct.

    verbose : boolean 
        Whether to print progress reports as reconstruction is performed.

    kwargs : optional
        "template": None,       (template signal for matched filtering)
        "epsilon": 0.01         (radius of hyperdisc for CVX problem)
        "length": len(svector) 
        Optional arguments - some are required for different functionality


    Returns
    -------
    CAOptimise class instance

    Raises
    ------
    KeyError
        If no measurement transform has been specified.
    """


    """
    Class that defines the process tensor given specified simulation parameters. Assumes qubits for now. 
    """

    def __init__(self, rho_se, U, sys_dim=2, time_steps=1, force=False):
        # store input variables
        # the initial system environmental state
        self.rho_se = rho_se
        # Unitary operator that is applied on the system/environment between controls
        self.U = U
        # the Hilbert dimension of the system
        self.sys_dim = sys_dim
        # the number of timesteps to simulate for
        self.time_steps = time_steps
        # whether to force simulation for questionable inputs
        self.force = force

        # compute utility parameters for later use
        # number of qubits in system and environment
        self.sq = int(np.round(np.log2(self.sd)))
        self.eq = int(np.round(np.log2(len(self.rho_se)))) - self.sq
        # dimension of environmental subsystem (assumes qubits)
        self.ed = 2**self.eq

        # check input parameters are valid
        assert np.shape(self.U)[0] == np.shape(self.U)[1] and len(np.shape(
            U)) == 2, "Unitary must be a square matrix but has dimensions: {}".format(np.shape(U))
        assert isdensity(
            self.rho_se), "Initial system/environment must be a valid density operator"
        assert np.shape(self.U)[1] == np.shape(self.rho_se)[
            0], "Unitary and initial state dimension mismatch: {} and {}".format(np.shape(self.U, self.rho_se))

    def apply(self, A, env=True):
        """
        Apply a sequence of control operations, assumes A is a DxDxk matrix made up of k A--forms. If env is false 
        will return just the system subsystem state i.e. will trace out the environment
        """

        # check if process tensor will be too large (not needed for now)
        if self.k > 5 and not self.force:
            raise ValueError("a {} step process is very large, set force parameter to True to compute this process tenor".format(self.k))

        # TODO: yikes
        # assert that the length of the controls is less than the time length of the process tensor
        try:
            assert np.shape(A)[2] == self.k, "Number of control operations does not equal k length of process tensor: {} != {}".format(
                self.k, np.shape(A)[2])

        except AssertionError as e:
            # catch force case
            if self.force:
                pass
            else:
                raise AssertionError(e)

        # simple evaluation of process tensor
        # create copy of inital system/environment to evolve and vectorise
        rho = vecstate(np.copy(self.rho_se))
        # compute identity channel to apply to environment system on control step (assumes qubits)
        env_identity, _ = IA_gen(N=self.eq)

        # iterate over time steps, performing control then unitary
        for step in range(0, self.k):
            # extract control operation to perform - assume A--form
            A_step = A[:, :, step]
            # pad with identity channel acting on environmental subsystem
            control_op = AB_join(env_identity, A_step)
            # apply control operation channel
            rho = control_op @ rho
            # convert to density operator and apply unitary (avoids calculating the channel rep of the unitary)
            rho = vecstate(self.U @ vecstate(rho) @ dagger(self.U))

        # devectorise final output state
        rho = vecstate(rho)

        # return full system state by default or trace out environment if requested
        if env:
            return np.asarray(rho)
        else:
            # list of qubits in environmental system to trace out
            t_qubits = list(range(0, self.eq))
            return partialtrace(rho, t_qubits)

    def pt_tomog(self):
        """
        Perform process tomography on the process tensor. Easiest way of discerning the full map if a bit
        computationally intensive. Can be done more efficiently with direct calculation but not by much and
        it is far easier to make a mistake.
        """

        # ensure the memory requirements are not beyond us (limiting case is storing the duals)
        if not self.force:
            # abort if required memory is more than 2 GB
            if mem_check([self.k], type=np.complex) > 1.6e+10:
                raise MemoryError(
                    'Process tensor dimension is too large (set force parameter to override)')

        # construct control operation iterator
        controls = ControlSpan(N=self.sq, k=self.k)

        # preallocate process tensor array
        ptensor = np.zeros((),dtype=np.complex128)

        

    def pt_compute(self): 
        """
        Directly compute the process tensor, sidestepping any tomography calculations. A very unpleasant function 
        to write, due solely to the subsystem operations that need to be performed. 
        """

    def cji(self):
        """
        Compute the Choi-Jamiokowlski form of the process tensor. We can't save anywhere on memory requirements 
        so we may as welll minimise the amount of time it takes. 
        """
        pass

def is_choi(A, tol=1e-6):

    """
    Class definition that handles signal reconstruction of a transformed input signal given
    the measurement basis. Able to perform standard compressed sensing and compressive 
    matched filter processing. 

    Parameters
    ----------
    svector : one dimensional numpy array
        the sensing or measurement vector y such that y = Ax where 
        x is the signal to reconstruct.

    transform : m*n array where m is len(svector) and n is the 
                dimension of the signal to reconstruct.

    verbose : boolean 
        Whether to print progress reports as reconstruction is performed.

    kwargs : optional
        "template": None,       (template signal for matched filtering)
        "epsilon": 0.01         (radius of hyperdisc for CVX problem)
        "length": len(svector) 
        Optional arguments - some are required for different functionality


    Returns
    -------
    CAOptimise class instance

    Raises
    ------
    KeyError
        If no measurement transform has been specified.
    """


    """ 
    Check if an input map is a valid Choi matrix/state. Ignores normalisation. 
    """
    Q = True
    # check shape
    if len(np.shape(A)) != 2 or np.shape(A)[0] != np.shape(A)[1]:
        Q = False
    
    # compute eigenvalues
    eigen_vals = np.linalg.eigvals(A)

    # zero small negative values within tolerance
    small_neg_ind = -tol<eigen_vals<0 
    eigen_vals[-tol<eig_vals[eigvals<0]] = 0

    # ensure positive semidefinitess



def is_deterministic(A, tol=1e-6):

    """
    Class definition that handles signal reconstruction of a transformed input signal given
    the measurement basis. Able to perform standard compressed sensing and compressive 
    matched filter processing. 

    Parameters
    ----------
    svector : one dimensional numpy array
        the sensing or measurement vector y such that y = Ax where 
        x is the signal to reconstruct.

    transform : m*n array where m is len(svector) and n is the 
                dimension of the signal to reconstruct.

    verbose : boolean 
        Whether to print progress reports as reconstruction is performed.

    kwargs : optional
        "template": None,       (template signal for matched filtering)
        "epsilon": 0.01         (radius of hyperdisc for CVX problem)
        "length": len(svector) 
        Optional arguments - some are required for different functionality


    Returns
    -------
    CAOptimise class instance

    Raises
    ------
    KeyError
        If no measurement transform has been specified.
    """



    """ 
    Checks if the input CP map is a deterministic process. Check method
    is fast but quite crude
    """
    # check if Kraus decomposition

    # check if Choi state/matrix

    # check if A-form

def Ejk(d,j,k):
    """
    Returns the zero dxd complex matrix with a 1 at index j,k 

    Parameters
    ----------
    d : positve integer specifying Hilbert space dimension.
    j : row index
    k : column index

    Returns
    -------
    ejk :  d x d complex numpy array containing a single 1 at [j,k]

    Raises
    ------
    """
    # construct zero complex matrix
    ejk = np.zeros((d,d), dtype=np.complex128)
    ejk[j,k] = 1
    return ejk


def gellman_gen(d):
    """
    Constructs a generalised Gellman matrix spanning set for complex space d x d. Use this for spanning SU(d). 

    Parameters
    ----------
    d : positve integer specifying Hilbert space dimension.

    Returns
    -------
    gellman : d^2 x d x d complex numpy array containing spanning set

    Raises
    ------
    ValueError if dimension is non-int or negative
    """

    # basic input parsing
    assert d>1 and type(d) is int, "Dimension must be positive integer greater than 1"

    # preallocate set arry
    gellman = np.empty((d**2, d, d), dtype=np.complex128)

    # iterate through use cases
    ind = 0
    for k in range(1, d):
        for j in range(0, k):
            # create symmetric component
            set_el_sym = Ejk(d, j, k) + Ejk(d, k, j)

            # create antisymmetric component
            set_el_asym = -1j*(Ejk(d, j, k) - Ejk(d, k, j)) 

            # add to set
            gellman[ind,:,:] = set_el_sym
            gellman[ind+1,:,:] = set_el_asym

            # step counter
            ind += 2

    # create diagonal elements 
    for l in range(1, d):

        # initialise zero matrix
        diagonal = np.zeros((d,d), dtype=np.complex128)
        coeff = np.sqrt(2/((l)*(l+1)))
        for i in range(0,l):
            diagonal += (Ejk(d,i,i))  

        diagonal -= l*Ejk(d,l,l)    

        # add to collection
        gellman[ind,:,:] = coeff*diagonal
        ind += 1

    # add identity to set
    gellman[-1,:,:] = np.eye(d, dtype=np.complex128)
    return gellman


def raise_lower_gen(n):
    """
    Returns the raising and lowering operators for a maximum of dim photons.

    Parameters
    ----------
    n : maximum number of photons - truncates the infinite Fock space.


    Returns
    -------
    tuple : tuple with structure (a^dagger, a).
    """

    # define diagonal entries
    raise_diag = np.sqrt(np.asarray(range(1,n+1)))
    lower_diag = np.sqrt(np.asarray(range(1,n+1)))

    # construct raising and lowering operators
    raise_op = np.diag(raise_diag, k=-1)
    lower_op = np.diag(lower_diag, k=1)

    return raise_op,lower_op


def is_tp(M, input_dims=None):
    """"
    Checks if an input Choi operator is trace preserving
    """ 
    # TODO
    pass 

def link_product(Choi_A, Choi_B, subsystems=1, common=[]):
    """
    Performs the link product (https://arxiv.org/abs/0904.4483) for two input Choi states
    with arbitrary input/output spaces such that Choi_C =  Choi_B o Choi_A. This assumes
    that Choi_A acts before Choi_B.  

    Parameters
    ----------
    Choi_A : m x m numpy complex array specifying the first Choi matrix.
    Choi_B : n x n numpy complex array specifiying the second Choi matrix.
    subsystems: integer number of subsystems on joint input space.
    common : indexed subsystems (starting from 0) that both maps act on.
             Empty list if no common input spaces


    Returns
    -------
    Choi_C : Composition of the input maps such that C = B o A[rho]

    Raises
    ------
    ValueError if inputs are not square.
    """

    if np.shape(Choi_A)[0] != np.shape(Choi_A)[1] or np.shape(Choi_B)[0] != np.shape(Choi_B)[1]:
        raise ValueError("Input Choi state(s) are non-square")

    # trivial case of no common subsystems
    if len(common) == 0:
        # TODO: This is not actually correct, there needs to be soem reshuffling
        Choi_C = np.kron(Choi_A, Choi_B)
    else:
        pass

    # normalise and return
    return np.shape(Choi_C)[0]*Choi_C/np.trace(Choi_C)


# generates direct product sum of arrayset according to combination
def dirsum(arrayset, combination):
    # initialise array with first element of combination
    U = arrayset[combination[0]]
    for i in combination[1:]:
        # append next operator
        U = block_diag(U, arrayset[i])
    return U

def multikron(A,r):
    """
    Computes the tensor power of A^\otimes r
    """
    B = 1.0
    for i in range(r):
        B = np.kron(B,A)
    return B


def tf_kron(A,B):
    """
    Computes the tensor power of A^\otimes r using tensorflow ops
    """
    # convert to linear operator objects
    A = tf.linalg.LinearOperatorFullMatrix(A)
    B = tf.linalg.LinearOperatorFullMatrix(B)
    # compute kronecker product
    C = tf.linalg.LinearOperatorKronecker([A,B])

    return tf.convert_to_tensor(C, dtype=tf.complex64)


def tf_multikron(A,r):
    """
    Computes the tensor power of A^\otimes r using tensorflow ops
    """
    A = tf.linalg.LinearOperatorFullMatrix(A)
    B = tf.linalg.LinearOperatorFullMatrix(tf.constant([[1]],dtype=tf.complex64))

    for i in range(r):
        B = tf.linalg.LinearOperatorKronecker([B,A])

    return tf.convert_to_tensor(B.to_dense(),dtype=tf.complex64)
    


def dagger(M):
    """
    Return conjugate transpose of input array
    """
    return np.transpose(np.conjugate(M))


def MUB_gen(d):
    """
    Generates a maximal MUB in d-dimensional Hilbert space for prime d
    """

    # base constant
    w = np.exp(2*np.pi*1j/d)
    # MUB container
    mub = np.zeros((d+1,d,d,d),dtype=np.complex128)
    # assign computational basis
    for i in range(d):
        mub[0,i,i,i] = 1.0


    for k in range(1,d+1):
        for m in range(d):
            state = np.zeros((d,1), dtype=np.complex128)
            for l in range(d):
                el = mub[0,l,:,l].reshape(d,1)
                state += w**(k*(l**2)+m*l) * el/np.sqrt(d)   
            mub[k,m,:,:] = np.kron(state, dagger(state))

    return mub

