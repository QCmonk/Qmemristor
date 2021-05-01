import numpy as np
from scipy.linalg import block_diag


# Plancks constant
pbar = 6.626070040e-34
# reduced
hbar = pbar/(2*np.pi)
# Bohr magneton in J/Gauss
mub = (9.274009994e-24)/1e4
# g factor
gm = 2.00231930436
# Gyromagnetic ratio
gyro = 699.9e3

# identity matrix
_ID = np.matrix([[1, 0], [0, 1]])
# X gate
_X = np.matrix([[0, 1], [1, 0]])
# Z gate
_Z = np.matrix([[1, 0], [0, -1]])
# Hadamard gate
_H = (1/np.sqrt(2))*np.matrix([[1, 1], [1, -1]])
# Y Gate
_Y = np.matrix([[0, -1j], [1j, 0]])
# S gate
_S = np.matrix([[1, 0], [0, 1j]])
# Sdg gate
_Sdg = np.matrix([[1, 0], [0, -1j]])
# T gate
_T = np.matrix([[1, 0], [0, (1 + 1j)/np.sqrt(2)]])
# Tdg gate
_Tdg = np.matrix([[1, 0], [0, (1 - 1j)/np.sqrt(2)]])
# CNOT gate
_CX = np.matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
# CNOT inverse
_CXdg = np.matrix([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
# SWAP gate
_SWAP = np.matrix([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
# toffoli gate
_TOFFOLI = block_diag(_ID, _ID, _CX)
# zero state
_pz = np.matrix([[1,0],[0,0]])
# one state
_po = np.matrix([[0,0],[0,1]])
# entangled |psi+> state
_ent = np.matrix([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]])/2


# define operators for spin 1/2 
op = {'h':   _H,
        'id':  _ID,
        'x':   _X,
        'y':   _Y,
        'z':   _Z,
        't':   _T,
        'tdg': _Tdg,
        's':   _S,
        'sdg': _Sdg,
        'cx':  _CX,
        'cxdg': _CXdg,
        'swap': _SWAP,
        'toff': _TOFFOLI}

states = {'pz': _pz,
          'po': _po,
          'ent': _ent}


# measurement projections for spin 1/2
meas1 = {"0":np.asarray([[1,0]]),
		 "1":np.asarray([[0,1]]),
		 "+":np.asarray([[1,1]]/np.sqrt(2)),
		 "-":np.asarray([[1,-1]]/np.sqrt(2)),
		 "+i":np.asarray([[1,1j]]/np.sqrt(2)),
		 "-i":np.asarray([[1,-1j]]/np.sqrt(2)),
		}