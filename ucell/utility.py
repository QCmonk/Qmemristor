
import tensorflow as tf
import math
import numpy as np
import string
import scipy
from math import isnan
from scipy.special import comb
from qinfo.qinfo import dagger,dirsum, haar_sample, rhoplot, tf_multikron
from contextlib import contextmanager
from ucell.operators import op
import sys
import os



@contextmanager
def silence_stdout():
    new_target = open(os.devnull, "w")
    old_target = sys.stdout
    sys.stdout = new_target
    try:
        yield new_target
    finally:
        sys.stdout = old_target

def tf_clements_stitch(beam_spec, theta, phi, diag, rev=tf.constant(True)):
    """
    Computes the unitary given by a clements decomposition with tensorflow compliance
    """

    nmax = beam_spec[0][-1]

    # construct adjoint of the phase shifts
    U = tf.linalg.diag(tf.exp(tf.complex(0.0, diag)))

    # iterate over specified beam splitter arrays
    for i, spec in enumerate(beam_spec):
        # construct nd scatter indices
        m, n = spec[:2]

        # construct indices to update with beam splitter params
        indices = index_gen(m,n,nmax)

        # retrieve iterations variable set
        th = tf.slice(theta, tf.constant([i]), size=tf.constant([1]))
        ph = tf.slice(phi, tf.constant([i]), size=tf.constant([1]))


        # construct beam splitter entries with compliant datatypes
        # a = tf.math.exp(tf.complex(0.0, ph))*tf.complex(tf.cos(th), 0.0)
        # b = tf.complex(-tf.sin(th), 0.0)
        # c = tf.math.exp(tf.complex(0.0, ph))*tf.complex(tf.sin(th), 0.0)
        # d = tf.complex(tf.cos(th), 0.0)

        # Valeria decomposition
        a = tf.math.exp(tf.complex(0.0, th))*tf.complex(tf.sin(ph/2), 0.0)
        b = tf.math.exp(tf.complex(0.0, th))*tf.complex(tf.cos(ph/2), 0.0)
        c = tf.complex(tf.cos(ph/2), 0.0)
        d = tf.complex(-tf.sin(ph/2), 0.0)

        # concatenate matrix elements into vector for scatter operation
        var_list = [tf.constant([1.0+0.0j], dtype=tf.complex64,shape=[1,])]*(2+nmax)
        var_list[:4] = [a, b, c, d]
        var_list = tf.stack(var_list, 0, name="varlist_{}".format(i))
        # place variables with appropriate functionals (see Clements paper for ij=mn variable maps)
        Tmn = tf.scatter_nd(indices, var_list, tf.constant([nmax**2,1], dtype=tf.int64))
        # reshape into rank 2 tensor
        
        # cannot use update as the variable reference does not seem to like gradient computation
        #Tmn = tf.scatter_nd_add(tf.zeros((nmax**2,), dtype=tf.complex64), indices, var_list, name="scatter_{}".format(i))
        Tmn = tf.reshape(Tmn, tf.constant([nmax]*2))

        # return unitary using specified order convention
        U = tf.cond(rev, lambda: tf.matmul(U, Tmn), lambda: tf.matmul(Tmn, U))

    return U


def index_gen(m,n,nmax):
    """
    Generates index pair mappings for scatter update
    """
    rows,cols= [m, m, n, n], [m, n, m, n]
    for i in range(nmax):
        # skip values that will be covered by beam splitter
        if (i == m) or (i==n): continue

        rows.append(i)
        cols.append(i)

    # comute index mappings when reshaped to rank one tensor
    indices = np.ravel_multi_index([rows, cols], (nmax, nmax)).reshape(nmax+2, 1).tolist()
    return tf.constant(indices, dtype=tf.int64)


def T(m, n, theta, phi, nmax):
    r"""The Clements T matrix from Eq. 1 of Clements et al. (2016)"""
    mat = np.identity(nmax, dtype=np.complex128)
    # mat[m, m] = np.exp(1j*phi)*np.cos(theta)
    # mat[m, n] = -np.sin(theta)
    # mat[n, m] = np.exp(1j*phi)*np.sin(theta)
    # mat[n, n] = np.cos(theta)
    mat[m, m] = np.exp(1j*theta)*np.sin(phi/2)
    mat[m, n] = np.exp(1j*theta)*np.cos(phi/2)
    mat[n, m] = np.cos(phi/2)
    mat[n, n] = -np.sin(phi/2)
    return mat


def tfT_batch(bms_spec, theta, phi):
    r"""The Clements T matrix with tensorflow compliance.
            -Most importantly works on batches
    """
    # from bms_spec, extract m,n and nmax
    m = [spec[0] for spec in bms_spec]
    n = [spec[1] for spec in bms_spec]
    nmax = bms_spec[0][-1]

    mat_mm = tf.multiply(tf.exp(tf.complex(0.0, phi)),tf.complex(tf.cos(theta), 0.0))

    mat_mn = -tf.complex(tf.sin(theta), 0.0)

    mat_nm = tf.multiply(tf.exp(tf.complex(0.0, phi)),tf.complex(tf.sin(theta), 0.0))

    mat_nn = tf.complex(tf.cos(theta), 0.0)



    return mat_mm


def Ti(m, n, theta, phi, nmax):
    r"""The inverse Clements T matrix"""
    return np.transpose(T(m, n, theta, -phi, nmax))


def nullTi(m, n, U):
    r"""Nullifies element m,n of U using Ti"""
    (nmax, mmax) = U.shape

    if nmax != mmax:
        raise ValueError("U must be a square matrix")

    if U[m, n+1] == 0:
        thetar = np.pi/2
        phir = 0
    else:
        r = U[m, n] / U[m, n+1]
        thetar = np.arctan(np.abs(r))
        phir = np.angle(r)

    return [n, n+1, thetar, phir, nmax]


def nullT(n, m, U):
    r"""Nullifies element n,m of U using T"""
    (nmax, mmax) = U.shape

    if nmax != mmax:
        raise ValueError("U must be a square matrix")

    if U[n-1, m] == 0:
        thetar = np.pi/2
        phir = 0
    else:
        r = -U[n, m] / U[n-1, m]
        thetar = np.arctan(np.abs(r))
        phir = np.angle(r)

    return [n-1, n, thetar, phir, nmax]


def clements(V, tol=1e-11):
    r"""Clements decomposition of a unitary matrix, with local
    phase shifts applied between two interferometers.
    See :ref:`clements` or :cite:`clements2016` for more details.
    This function returns a circuit corresponding to an intermediate step in
    Clements decomposition as described in Eq. 4 of the article. In this form,
    the circuit comprises some T matrices (as in Eq. 1), then phases on all modes,
    and more T matrices.
    The procedure to construct these matrices is detailed in the supplementary
    material of the article.
    Args:
        V (array[complex]): unitary matrix of size n_size
        tol (float): the tolerance used when checking if the matrix is unitary:
            :math:`|VV^\dagger-I| \leq` tol
    Returns:
        tuple[array]: tuple of the form ``(tilist,tlist,np.diag(localV))``
            where:
            * ``tilist``: list containing ``[n,m,theta,phi,n_size]`` of the Ti unitaries needed
            * ``tlist``: list containing ``[n,m,theta,phi,n_size]`` of the T unitaries needed
            * ``localV``: Diagonal unitary sitting sandwiched by Ti's and the T's
    """
    localV = V
    (nsize, _) = localV.shape

    diffn = np.linalg.norm(V @ V.conj().T - np.identity(nsize))
    if diffn >= tol:
        raise ValueError("The input matrix is not unitary")

    tilist = []
    tlist = []
    for k, i in enumerate(range(nsize-2, -1, -1)):
        if k % 2 == 0:
            for j in reversed(range(nsize-1-i)):
                tilist.append(nullTi(i+j+1, j, localV))
                localV = localV @ Ti(*tilist[-1])
        else:
            for j in range(nsize-1-i):
                tlist.append(nullT(i+j+1, j, localV))
                localV = T(*tlist[-1]) @ localV

    return tilist, tlist, np.diag(localV)


def clements_phase_end(V, tol=1e-11):
    r"""Clements decomposition of a unitary matrix.
    See :cite:`clements2016` for more details.
    Final step in the decomposition of a given discrete unitary matrix.
    The output is of the form given in Eq. 5.
    Args:
        V (array[complex]): unitary matrix of size n_size
        tol (float): the tolerance used when checking if the matrix is unitary:
            :math:`|VV^\dagger-I| \leq` tol
    Returns:
        tuple[array]: returns a tuple of the form ``(tlist,np.diag(localV))``
            where:
            * ``tlist``: list containing ``[n,m,theta,phi,n_size]`` of the T unitaries needed
            * ``localV``: Diagonal unitary matrix to be applied at the end of circuit
    """
    tilist, tlist, diags = clements(V, tol)
    new_tlist, new_diags = tilist.copy(), diags.copy()

    # Push each beamsplitter through the diagonal unitary
    for i in reversed(tlist):
        em, en = int(i[0]), int(i[1])
        alpha, beta = np.angle(new_diags[em]), np.angle(new_diags[en])
        theta, phi = i[2], i[3]

        # The new parameters required for D',T' st. T^(-1)D = D'T'
        new_theta = theta
        new_phi = np.fmod((alpha - beta + np.pi), 2*np.pi)
        new_alpha = beta - phi + np.pi
        new_beta = beta

        new_i = [i[0], i[1], new_theta, new_phi, i[4]]
        new_diags[em], new_diags[en] = np.exp(
            1j*new_alpha), np.exp(1j*new_beta)

        new_tlist = new_tlist + [new_i]

    return (new_tlist, new_diags)


def clements_stitch(tlist, diags):
    """
    Computes the unitary given by a clements decomposition
    """
    # construct adjoint of the phase shifts
    U = np.diag(np.conjugate(diags))
    # iterate over specified beam splitter arrays
    for Tmn in tlist[::-1]:
        # construct beamsplitter given parameters
        bm = T(*Tmn)
        U = U @ bm
    return U

def str_base(num, base, length, numerals = '0123456789'):
    if base < 2 or base > len(numerals):
        raise ValueError("str_base: base must be between 2 and %i" % len(numerals))

    if num == 0:
        return '0'*length

    if num < 0:
        sign = '-'
        num = -num
    else:
        sign = ''

    result = ''
    while num:
        result = numerals[num % (base)] + result
        num //= base


    out = sign + result

    if len(out) < length:
        out = '0'*(length - len(out)) + out
    return out


def opto_gen(modes, targets):
    """
    Gives a Mach-Zedner Interferometer decomposition for an arbitrary number of modes and mode
    target pairs.
    """

    # initliase specification container
    bms_spec = []

    # for each target pair, generate a template decomp using standard functionality
    for pair in targets:
        
        # generate MZI template
        template = [0,1,0,0,modes]

        # assume indexing starts at 1, update values explicitly  
        template[0] = pair[0]-1
        template[1] = pair[1]-1

        # add element to blueprint list
        bms_spec.append(template)

    return bms_spec




def symmetric_map(m_num, p_num):
    """
    Computes the permutation matrix to map to the isomorphic Hilbert space for p_num photons
    and m_num modes, eliminating degenerate degrees of freedom. Exponentially faster than matrix permanent method
    everyone else insists on using but at the cost of higher memory complexity. Should probably tell
    someone about this...
    """

    # compute size of output matrix
    row_num = comb(m_num + p_num-1, p_num);
    col_num = m_num**p_num;

    # compute photon states as an m-ary number of p_num bits
    photon_state = np.asarray([list(str_base(n,m_num,p_num)) for n in range(m_num**p_num)]).astype(int)

    # compute mode occupation number for each row
    fock = np.zeros((col_num, m_num), dtype=np.int32);
    # iterate over rows
    for i in range(np.shape(photon_state)[0]):
        # iterate over columns
        for j in range(m_num):
            fock[i,j] = np.sum(photon_state[i,:]==j);

    # # compute unique Fock states
    uniques = np.fliplr(np.unique(fock, axis=0))
    ldim = np.shape(uniques)[0]


    # preallocate symmetric transform matrix
    P = np.zeros((ldim, col_num));

    # iterate over symmetric dimension
    for k in range(ldim):
        for m in range(col_num):
            if (uniques[k,:] == fock[m,:]).all():
                P[k,m] = 1;

        
        # ensure normalisation property holds
        P[k,:] /= np.sqrt(np.sum(P[k,:]))

    return P


def number_states(m_num,p_num):
    """
    outputs a list of the number states in each mode
    """

    # compute size of output matrix
    row_num = comb(m_num + p_num-1, p_num);
    col_num = m_num**p_num;

    # compute photon states as an m-ary number of p_num bits
    photon_state = np.asarray([list(str_base(n,m_num,p_num)) for n in range(m_num**p_num)]).astype(int)

    # compute mode occupation number for each row
    fock = np.zeros((col_num, m_num), dtype=np.int32);
    # iterate over rows
    for i in range(np.shape(photon_state)[0]):
        # iterate over columns
        for j in range(m_num):
            fock[i,j] = np.sum(photon_state[i,:]==j);

    # compute unique Fock states
    uniques = np.fliplr(np.unique(fock, axis=0))
    return uniques


def nl_gen(nldesc, convert=False):
    """
    Generates a non-linear operation and ancillary information given description

    nldesc := {type=[swap, sswap, pphase], u_prob = [0,1], targets=[0...1,1,0..,0]}
    """
    
    if type(nldesc) is not dict:
        raise TypeError("Non-linear layer description must be dictionary of parameters")

    # extract general success probability of non linear activation
    u_prob = tf.constant(nldesc['u_prob'], dtype=tf.complex64)

    # compute dimension of system
    modes = nldesc["modes"]
    photons = nldesc["photons"]
    dim = comb(modes+photons-1, photons, exact=True)


    try:
        # this type is a bit more complicated, will come back to implement later
        if nldesc['type']=="pphase":
            # explicitly extract mode number and photon number
            m_num,p_num = len(nldesc['targets']), nldesc['photons']
            # compute number state descriptor
            nstates = number_states(m_num, p_num)
            # compute nonlinear unitary
            unitary = np.eye(comb(m_num+p_num-1, p_num, exact=True), dtype=np.complex128)
            for i in range(np.shape(nstates)[0]):

                # multiply by mask to remove nonlinear term on those modes
                fock_nums = nldesc['targets']*nstates[i,:]
                # compute phase factor nonlinear layer 
                unitary[i,i] *= np.prod(np.exp(1j*np.pi*(np.clip(fock_nums-1,0,None))))

        elif nldesc['type']=='swap':
            # compute single photon unitary corresponding to requestion nonlinearity
            unitary = dirsum([np.asarray([1.0]), op['swap']], nldesc['targets'])

        # sqrt of swap gate unitary applied between arbitrary modes
        elif nldesc['type']=='sswap':
            # get target pairs
            pairs = nldesc['pairs']
            # generate swap gate on given pairs
            U = np.eye(dim, dtype=np.complex128)
            # construct sqrt swap gate on given targets
            for pair in pairs:
                V = swap_gen(modes, photons, pair, partial=True)
                U = V @ U
                
            unitary = U

        elif nldesc['type']=='cx':
            unitary = dirsum([np.asarray([1.0]), op['cx']], nldesc['targets'])
            # generate random unitary of specified dimension - easy nonlinearity
        elif nldesc["type"]=="rand":
            # set seed if parsed
            if 'seed' in nldesc:
                np.random.seed(nldesc['seed'])
            # construct random unitary on specified 
            unitary = dirsum([np.asarray([1.0]), randU(nldesc['dim'])], nldesc['targets'])
        
        elif nldesc["type"]=="id":
            # simply perform identity operation (useful for noisy channels)
            unitary = np.eye(dim)
        else:
            # non-recognised gate parameter
            raise ValueError("Nonlinear type \'{}\' not recognised".format(nldesc['type']))

    # raise error if missing a keyword argument
    except KeyError as e:
        raise KeyError("Non-linear layer specification is missing argument {}".format(e))

    # convert to 3 dimensional tensor if requested
    if convert:
        unitary = tf.convert_to_tensor(unitary, dtype=tf.complex64)

    return nldesc["type"], nldesc["u_prob"], unitary

def swap_gen(modes, photons, pair, partial=True):
    """
    Constructs the partial swap on the full photonic output space
    """

    # dimension of system
    dim = comb(modes+photons-1, photons, exact=True)
    # basis
    basis = np.eye(dim)
    # number states
    nstates = number_states(modes,photons)

    # construct the unitary transform based on required mappings
    U = np.zeros((dim,dim), dtype=np.complex128)

    # iterate over all combinations of states (yuck code)
    for i,start_state in enumerate(nstates):
        # compute the swapped state
        swap_state = list(start_state)
        swap_state[pair[0]-1],swap_state[pair[1]-1] = swap_state[pair[1]-1],swap_state[pair[0]-1]
        
        for j,end_state in enumerate(nstates):
            if (swap_state==end_state).all():
                U[i,j] = 1.0
                break

    if partial:
        U = scipy.linalg.sqrtm(U)
    return U






def randU(dim):
    """
    Generate random unitary matrices of dimension D
    """

    X = (np.random.randn(dim,dim) + 1j*np.random.randn(dim,dim))/np.sqrt(2);
    Q,R = np.linalg.qr(X);
    R = np.diag(np.divide(np.diag(R),np.abs(np.diag(R))));
    U = Q @ R;
    return U


def bellgen(num=1):
    """
    Generates an optical bell state on 4 modes
    """

    # prelims
    nstates = number_states(4,2)
    dim = comb(4+2-1,2, exact=True)
    basis = np.eye(dim)
    proj = np.zeros([dim,])

    # generate psi-
    if num==2:
        # basis states |1001> and -|0110>
        s1 = np.asarray([1, 0, 0, 1])
        s2 = np.asarray([0, 1, 1, 0])
        
        # this is bad even for me
        for i in range(len(nstates)):
            if (s1 == nstates[i,:]).all():
                proj = proj + basis[:,i]

            if (s2 == nstates[i,:]).all():
                proj = proj - basis[:,i]

        
    # generate phi+
    elif num==3:
        # states |1010> and |0101>
        s1 = np.asarray([1, 0, 1, 0])
        s2 = np.asarray([0, 1, 0, 1])
        
        for i in range(len(nstates)):
            if (s1 == nstates[i,:]).all():
                proj = proj + basis[:,i]
            if (s2 == nstates[i,:]).all():
                proj = proj + basis[:,i]

    # generate phi-
    elif num==4:
        # states |1010> and -|0101>
        s1 = np.asarray([1, 0, 1, 0])
        s2 = np.asarray([0, 1, 0, 1])
        
        for i in range(len(nstates)):
            if (s1 == nstates[i,:]).all():
                proj = proj + basis[:,i]
            
            if (s2 == nstates[i,:]).all():
                proj = proj - basis[:,i]

    # default to psi+
    else:
        # basis states |1001> and |0110>
        s1 = np.asarray([1, 0, 0, 1])
        s2 = np.asarray([0, 1, 1, 0])
        for i in range(len(nstates)):
            if (s1 == nstates[i,:]).all():
                proj = proj + basis[:,i]
            
            if (s2 == nstates[i,:]).all():
                proj = proj + basis[:,i]

    proj = proj.reshape(dim,1)
    return np.kron(proj, dagger(proj))/2


def noon_gen(modes, photons, N=None):
    """
    Generates an NOON state on the last two modes
    """
    # basic input value check
    if N is None:
        N=photons
    else:
        if N>photons:
            raise ValueError("NOON state {} cannot be generated with {} photons".format(N,photons))

        if modes<=2 and N!=photons:
            raise ValueError("Mode number must be greater than two to support ancilla photons")
    

    # prelims
    nstates = number_states(modes,photons)
    dim = comb(modes+photons-1,photons, exact=True)
    basis = np.eye(dim)
    proj = np.zeros([dim,])

    # basis states |1001> and |0110>
    if modes>2:
        # gross
        s1 = [0]*(modes-2)
        s2 = [0]*(modes-2)
        s1[0] = photons-N
        s2[0] = photons-N
        s1.append(N)
        s1.append(0)
        s2.append(0)
        s2.append(N)
        s1 = np.asarray(s1)
        s2 = np.asarray(s2)
    else:
        s1 = np.asarray([N, 0])
        s2 = np.asarray([0, N])
    
    # this is bad even for me
    for i in range(len(nstates)):
        if (s1 == nstates[i,:]).all():
            proj = proj + basis[:,i]

        if (s2 == nstates[i,:]).all():
            proj = proj + np.exp(1j*np.pi*N)*basis[:,i]

    proj = proj.reshape(dim,1)
    return np.kron(proj, dagger(proj))/2




def povm_gen(pdesc, convert=False):
    """
    Generate a projection operator that collapses superpositions and can purify mixed states.
    pdesc := {targets=[1,2...m], modes, photons, eff=[0,1]}.
    """

    # perform some basic input parsing
    if type(pdesc) is not dict:
        raise TypeError("Projection layer description must be dictionary of parameters")

    try: 
        # get explicit layer parameters
        modes = pdesc["modes"]
        photons = pdesc["photons"]
        targets = pdesc["targets"]

        # generate null mask for target modes
        null = np.asarray([1.0 if i in targets else 0.0 for i in range(1,modes+1)])

        dim = comb(modes+photons-1, photons, exact=True)
        # compute number states        
        nstates = number_states(modes, photons)
        # generate a basis set to draw from
        basis = np.eye(dim)
        # for each measurement outcome on designated modes, compute projector
        projectors = []
        complete = []
        for i in range(dim):
            # get outcome target state
            fvec = nstates[i,:]
            fvec = null*fvec
            # pick measurement outcome 
            if list(fvec) in complete: 
                continue
            else:
                # null free modes with elementwise multiplication
                complete.append(list(fvec))
                
   
                # iterate over number states and find correct mappings
                M = np.zeros((dim,dim))
                for j in range(dim):
                    if (fvec == (null*nstates[j,:])).all():
                        M += np.kron(basis[:,j].reshape(dim,1), basis[:,j].reshape(1,dim))

                projectors.append(M)


    except KeyError as e:
        raise KeyError("Projection layer specification is missing argument {}".format(e))

    # repackage into projector tensor
    proj = np.zeros((len(projectors), dim,dim))
    for i in range(len(proj)):
        proj[i,:,:] = projectors[i]

    # convert to 3 dimensional tensor if requested
    if convert:
        proj = tf.convert_to_tensor(proj, dtype=tf.complex64)

    return proj


def iso_gen(idesc, convert=False):
    """ 
    Generates a pseudo isometry for mapping ancilla modes contents to a single
    state. Targets specifies the ancilla modes, dest is the ancilla 
    idesc: = {targets=[1,2,3], modes,photons, dest=1,2,...} 
    """

    # perform some basic input parsing
    if type(idesc) is not dict:
        raise TypeError("Projection layer description must be dictionary of parameters")

    try:
        # get explicit layer parameters
        modes = idesc["modes"]
        photons = idesc["photons"]
        targets = idesc["targets"]
        dest = idesc["dest"]

        if dest not in targets:
            raise ValueError("Destination mode must be one of the specified ancilla modes")

        # compute system dimension 
        dim = comb(modes+photons-1, photons, exact=True)
        # generate null mask for target modes
        null = np.asarray([1 if i in targets else 0 for i in range(1,modes+1)])
        # generate null mask for non-target modes
        null_inv = np.asarray([0 if i in targets else 1 for i in range(1,modes+1)])
        # compute number states        
        nstates = number_states(modes, photons)
        # generate a basis set to draw from
        basis = np.eye(dim)

        # for each measurement outcome on designated modes, compute projector
        projectors = []
        complete = []
        for i in range(dim):
            # get input state to map
            fvec = nstates[i,:]
            fvec = null*fvec

            # pick measurement outcome 
            if list(fvec) in complete: 
                continue
            else:
                # null free modes with elementwise multiplication
                complete.append(list(fvec))
                
                # compute number of photons in ancilla modes
                anc_photons = np.sum(fvec)
                # construct state that this should map to (all photons in dest mode)
                map_state = np.zeros((modes,),dtype=np.int16)
                map_state[dest-1] = anc_photons
                # cannot find basis element to map to as this depends on target modes
                # iterate over number states and find correct mappings

                M = np.zeros((dim,dim))
                for j in range(dim):
                    if (fvec == (null*nstates[j,:])).all():
                        # compute basis state to map from
                        input_state = basis[:,j]
                        # compute basis state to map to
                        output_state = map_state+null_inv*nstates[j,:]
                        # get basis element of mapped state
                        ind = np.where(np.all(nstates==output_state,axis=1))[0][0]
                        output_state = basis[:,ind]
                        # compute projector mapping
                        M += np.kron(output_state.reshape(dim,1), input_state.reshape(1,dim))

                projectors.append(M)


    except KeyError as e:
        raise KeyError("Isometric layer specification is missing argument {}".format(e))

    # repackage into projector tensor
    proj = np.zeros((len(projectors), dim,dim))
    for i in range(len(proj)):
        proj[i,:,:] = projectors[i]

    # convert to 3 dimensional tensor if requested
    if convert:
        proj = tf.convert_to_tensor(proj, dtype=tf.complex64)

    return proj


def Udata_gen(U, num=1000):
    """
    Generates some input/output data for unitary evolution
    """
    # compute dimension of system
    dim = np.size(U, 1)

    # generate randomly sampled pure input states
    psi = haar_sample(dim=dim, num=num, pure=True, operator=True)

    # preallocate unitary output states
    phi = np.zeros_like(psi)

    # compute output states phi subject to U*psi*U'
    for i in range(num):
        phi[i, :, :] = U @ psi[i, :, :] @ dagger(U)

    psi = tf.convert_to_tensor(psi, dtype=tf.complex64, name='psi')
    phi = tf.convert_to_tensor(phi, dtype=tf.complex64, name='phi')

    return psi, phi

def M_apply(M,rhob):
    """
    Applies a map M to an input batch of states
    """
    for i in range(np.shape(rhob)[0]):
        rho = np.asarray(rhob[i,:,:])
        rhob[i,:,:] *= 0
        for j in range(np.shape(M)[0]):
            rho += M[j,:,:] @ rho @ dagger(M[j,:,:])
        rhob[i,:,:] = rho

    return rhob


def keraskol(rho, gamma):
    """
    tensorflow compatible quantum kolmogorov distance
    """
    return tf.linalg.trace(tf.abs(rho-gamma))


def purekol(rho, gamma):
    """
    Computes a mixed metric of trace distance and purity
    """
    purity = tf.abs(tf.linalg.trace(tf.einsum('bjk,bkl->bjl', gamma, gamma)))
    kol = keraskol(rho, gamma)
    return tf.divide(kol, purity)


def kerasfid(rho, gamma):
    """
    keras compatible quantum fidelity as a minimisation task
    """
    print(rho)
    return 1 - tf.real(tf.linalg.trace(tf.matmul(rho, gamma)))


def mean_error(rho, gamma):
    """
    tensorflow mean error
    """
    return tf.abs(tf.metrics.mean(rho-gamma))


def lrscheduler(epoch):
    """
    Scheduler callback function for learning rate
    """
    lrate = 1e-3/((1+epoch)/100)

    return lrate

def null_matrix(modes, photons, convert=False):
    """
    computes null matrix for loss calcuation updates by penalising 
    indistingushable states for NOON states
    """

    # compute dimension of system
    dim = comb(modes+photons-1, photons, exact=True)
    null = np.zeros((dim,dim))
    nstates = number_states(modes, photons)
    basis = np.eye(dim)

    # iterate over dimension of space
    for i in range(dim):
        for j in range(dim):
            # get index element states
            avec = nstates[i,:]
            bvec = nstates[j,:]
            # check if either are bad and weight appropriately
            if np.sum(avec[-2:])==photons:
                if np.prod(avec[-2:])>=0 and np.sum(avec[:-2])==0:
                    null[i,j] = 1.0

            if np.sum(bvec[-2:])==photons:
                if np.prod(bvec[-2:])>=0 and np.sum(bvec[:-2])==0:
                    null[i,j] = 1.0
    if convert:
        null = tf.convert_to_tensor(null, dtype=tf.complex64)

    return null

def weight_matrix(modes, photons, w=5, convert=False):
    """
    computes weight matrix for loss calcuation updates by penalising 
    indistingushable states for NOON states
    """

    # compute dimension of system
    dim = comb(modes+photons-1, photons, exact=True)
    weights = np.ones((dim,dim))
    nstates = number_states(modes, photons)
    basis = np.eye(dim)

    # iterate over dimension of space
    for i in range(dim):
        for j in range(dim):
            # get index element states
            avec = nstates[i,:]
            bvec = nstates[j,:]
            # check if either are bad and weight appropriately
            if np.sum(avec[-2:])==photons:
                if np.prod(avec[-2:])>0 and np.sum(avec[:-2])==0:
                    weights[i,j] *= w

            if np.sum(bvec[-2:])==photons:
                if np.prod(bvec[-2:])>0 and np.sum(bvec[:-2])==0:
                    weights[i,j] *= w
    if convert:
        weights = tf.convert_to_tensor(weights, dtype=tf.complex64)

    return weights


def wkfid(weights):
    """
    Computes quantum keras fidelity with weights matrix
    """
    def weightfid(rho, gamma):
        """
        tensorflow compatible quantum kolmogorov distance
        """
        gam = tf.math.multiply(weights,gamma)
        return 1 - tf.real(tf.linalg.trace(tf.matmul(rho,gam)))

    return weightfid

def probfid(null):
    """
    Computes quantum keras fidelity with weights matrix
    """
    def combi(rho, gamma):
        """
        tensorflow compatible quantum kolmogorov distance
        """
        gam = tf.multiply(null,gamma)
        #gam = tf.divide(gam, tf.linalg.trace(gam))
        return 1 - tf.real(tf.linalg.trace(tf.matmul(rho,gamma)))*tf.math.pow(tf.real(tf.linalg.trace(tf.matmul(rho,gam))), tf.constant(2.0))

    return combi

def bell_gen(modes, photons, bell=1):
    """
    Generates a bell state of two photons using the last 4 modes
    """

    assert modes>= 4, "Need at least 4 modes for Bell state generation"
    assert photons >=2, "Need at least two photons for Bell state generation"

    
    # number of ancilla photons
    aphotons = photons - 2
    # number of ancilla modes
    amodes = modes - 4
    if amodes == 0 and photons > 2:
        aphotons = 0
        photons = 2
        print("Warning: Must have ancilla modes for ancilla photons, truncating")

    # prelims
    nstates = number_states(modes,photons)
    # dimension of space
    dim = comb(modes+photons-1,photons, exact=True)
    # basis set for complete space
    basis = np.eye(dim)
    # output projector
    proj = np.zeros([dim,])

    # ancilla output state
    if amodes>0:
        aout = [0]*amodes
        aout[0] = aphotons
    else:
        aout = []

    # generate psi-
    if bell==2:
        # basis states |1001> and -|0110>
        s1 = aout + [1, 0, 0, 1]
        s2 = aout + [0, 1, 1, 0]
        
        
        # this is bad even for me
        for i in range(dim):
            if (s1 == nstates[i,:]).all():
                proj = proj + basis[:,i]

            if (s2 == nstates[i,:]).all():
                proj = proj - basis[:,i]

        
    # generate phi+
    elif bell==3:
        # states |1010> and |0101>
        s1 = aout + [1, 0, 1, 0]
        s2 = aout + [0, 1, 0, 1]
        
        for i in range(dim):
            if (s1 == nstates[i,:]).all():
                proj = proj + basis[:,i]

            if (s2 == nstates[i,:]).all():
                proj = proj + basis[:,i]

    # generate phi-
    elif bell==4:
        
        # states |1010> and -|0101>
        s1 = aout + [1, 0, 1, 0]
        s2 = aout + [0, 1, 0, 1]
        
        for i in range(dim):
            if (s1 == nstates[i,:]).all():
                proj = proj + basis[:,i]
            
            if (s2 == nstates[i,:]).all():
                proj = proj - basis[:,i]
            
        
    # default to psi+
    else:
        # basis states |1001> and |0110>
        s1 = aout + [1, 0, 0, 1]
        s2 = aout + [0, 1, 1, 0]
       
        for i in range(dim):
            if (s1 == nstates[i,:]).all():
                proj = proj + basis[:,i]

            if (s2 == nstates[i,:]).all():
                proj = proj + basis[:,i]

    proj = proj.reshape(dim,1)
    return np.kron(proj, dagger(proj))/2

def proj_gen(outcome):
    """
    Generates the measurement projector for specified Fock state.
    """

    # input parsing
    modes = len(outcome)
    photons = sum(outcome)

    # dimension
    dim = dim = comb(modes+photons-1, photons, exact=True)
    # compute number states        
    nstates = number_states(modes, photons)
    # basis set for complete space
    basis = np.eye(dim)
    # output projector
    proj = np.zeros([dim,])

    # find basis element
    for i in range(dim):
            if (outcome == nstates[i,:]).all():
                proj = proj + basis[:,i]

    proj = proj.reshape(dim,1)
    return np.kron(proj, dagger(proj))


def bell_train_gen(amodes, aphotons, convert=False):
    """
    Generates the training set for bell state discrimination
    """
    modes = 4 + amodes
    photons = 2 + aphotons

    # ancillary outcomes
    if amodes>0:
        aout = [0]*amodes
        aout[0] = aphotons
    else:
        aout = []

    # dimension
    dim = dim = comb(modes+photons-1, photons, exact=True)

    # preallocate input states out measurement projectors
    bells = np.zeros((4, dim,dim), dtype=np.complex128)
    projs = np.zeros((4, dim,dim), dtype=np.complex128)
    
    # generate each bell state and store
    for i in range(4):
        bells[i,:,:] = bell_gen(modes, photons, bell=i+1)

    # no clever way of doing this, hardcode each desired outcome
    psip1 = aout +  [1,0,0,1]
    psip2 = aout +  [0,1,1,0]
    projs[0,:,:] = proj_gen(psip1) + proj_gen(psip2)

    psip1 = aout +  [1,1,0,0]
    psip2 = aout +  [0,0,1,1]
    projs[1,:,:] = proj_gen(psip1) + proj_gen(psip2)

    # psip1 = aout +  [2,0,0,0]
    # psip2 = aout +  [0,2,0,0]
    # projs[2,:,:] = proj_gen(psip1) + proj_gen(psip2)

    psip1 = aout +  [2,0,0,0]
    psip2 = aout +  [0,2,0,0]
    projs[2,:,:] += proj_gen(psip1) + proj_gen(psip2)

    if convert:
        bells = tf.convert_to_tensor(bells, dtype=tf.complex64)
        projs = tf.convert_to_tensor(projs, dtype=tf.complex64)

    return bells, projs




def fid_min(rho, gamma):
    """
    Computes the fidelity of rho with itself and other members
    """

    # get shape of tensor
    gamma_shape = gamma.get_shape().as_list()
    
    # multiplication of outputs with each other
    
    # iterate over fid
    # fid = 0.0
    # for i in range(1,4):
    #     # roll matrix
    #     gamma_roll = tf.roll(gamma, shift=i,axis=0)
    #     error = gamma - gamma_roll

    #     abseig = tf.abs(tf.linalg.eigvalsh(error))


    #     # compute intermediate matrices
    #     # emult1 = tf.einsum('ijk,ikl->ijl', gammasqrtm, gamma_roll)
    #     # emult2 = tf.einsum('ijk,ikl->ijl', emult1, gammasqrtm)
    #     # emultsqrtm = tf.linalg.sqrtm(emult2)
        
    #     # compute fidelity measure
    #     fid = fid + tf.reduce_sum(abseig)

    gamma = tf.abs(gamma)

    #gamma = tf.multiply(gamma, tf.eye(num_rows=gamma_shape[-1], batch_shape=gamma_shape[0]))
    fid_ten = tf.einsum('ijk,mkl->imjl', gamma, gamma)
    purity = tf.einsum('ijk,ikl->ijl', gamma, gamma)
    # trace on each multiplication - fidelity measure
    fid_ten = tf.linalg.trace(fid_ten)
    purity = tf.linalg.trace(purity)
    fid = tf.real(tf.reduce_sum(fid_ten))-tf.real(tf.reduce_sum(purity))

    return fid

def cellkey(brain):
    """
    fitness extractor for sorting purposes
    """
    return brain.fitness

def mute():
    sys.stdout = open(os.devnull, 'w')   


if __name__ == '__main__':
    print(T(1,1,0,0,2))

