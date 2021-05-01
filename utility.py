import numpy as np
import matplotlib.pyplot as plt 
from keras.layers import Input, Layer
import collections
from math import log, exp

import qfunk.qoptic as qop
from qinfo.qinfo import dagger, haar_sample, multikron, gellman_gen, random_U
from ucell import *
from ucell.utility import *
from test_data import *




class QMemristor(object):
    """
    Advanced data map that implements a stack of quantum memristors acting in parallel. 
    each memristor takes as input 4 modes - two being the input state, a third 
    being the detector input (instantiated to no input) and a final sacrifical mode
    that contains the destroyed photons. 

    pdesc = {'theta_init':None,'MZI_target': [m,m+1],'proj_mode': n!=m,m+1, 'sacrificial_mode': o!=n,m,m+1}
    """

    def __init__(self, modes, photons, pdesc, force=False, **kwargs):
        # layer identifier
        self.id = "qmemristor"
        # total number of modes
        self.modes = modes
        # total number of photons
        self.photons = photons
        # number of variables on SU(modes)
        self.vars = (self.modes**2 - self.modes)//2
        # projection operator description
        self.pdesc = pdesc
        # length of time series
        self.tlen = self.pdesc["tlen"]
        # emergent dimension
        self.input_dim = self.output_dim = comb(
            self.modes+self.photons-1, self.photons, exact=True)
        # catch extreme cases
        if self.input_dim > 100 and not force:
            raise ValueError(
                "System dimension is large ({}), decrease system size or set force flag to True".format(self.input_dim))

        # define memory state vector
        self.mem_states = np.zeros((self.tlen), dtype=np.float64)
        # initialise unitary array
        self.unitary_array = np.zeros((self.tlen, self.input_dim, self.input_dim), dtype=np.complex128)

        # initial value of phase shifter 

        if 'theta_init' not in self.pdesc or self.pdesc['theta_init']==None:
            self.mem_states[0] = np.random.rand()*np.pi
        else:
            self.mem_states[0] = self.pdesc['theta_init'] 

        # save symmetric map for dilation to complete Fock space (much, much faster than permanents)
        self.S = symmetric_map(self.modes, self.photons)
        # build initial unitary array
        self.update(time_index=0)
        # build projector tensor for measure/prepare stage
        self.projector_build()

        # memory state vector 
        super(QMemristor, self).__init__(**kwargs)

    def projector_build(self):
        """
        Construct quantum memristor with stateful phase shifter and projective measure
        """

        # generate the projector that takes the memristor from a given quantum state to its average state
        # generate null vector that ignores photons not in measurement mode
        null = np.zeros((self.modes,))
        null[self.pdesc['proj_mode']] = 1

        # generate basis set
        basis = np.eye(self.input_dim)
        # and associate with number states
        nstates = number_states(self.modes, self.photons)
        # construct projector list
        projectors = []
        # construct completeness list
        complete = []
        # construct photon mode number list 
        self.proj_photon = []

        # iterate over all basis states
        for i in range(self.input_dim):
            # get number state
            fvec = nstates[i,:]
            # remove elements we don't care about
            fvec = null*fvec

            # skip nullified basis elements already completed (why we work with integers)
            if list(fvec) in complete:
                continue
            else:
                # add to completed projector list
                complete.append(list(fvec))

                # generate isometric projector
                M = np.zeros((self.input_dim, self.input_dim))

                # iterate over the number states
                for j in range(self.input_dim):
                    # if feature vector matches a basis in the modes we care about
                    if (fvec == (null*nstates[j,:])).all():
                        
                        #TODO: allow for inability to perform photon counting statistics
                        # feature vector must be mapped to one photon in measurement mode and additional photons in sacrificial mode
                        # collapse superpositions if no more than one photon
                        if sum(fvec)>=0:
                            M += np.kron(basis[:,j].reshape(self.input_dim,1), basis[:,j].reshape(1,self.input_dim))
                        # more than one photon in measurement mode, map to 

                # add that projector to the list
                projectors.append(M)
                # add photon number associated with projection
                self.proj_photon.append(fvec[self.pdesc['proj_mode']])
                
        # repackage into projector tensor
        self.proj = np.zeros((len(projectors), self.input_dim, self.input_dim))
        # iterate over all projectors
        for i in range(len(projectors)):
            self.proj[i,:,:] = projectors[i]

        # set build flag
        self.built = True

    def call(self, input_states):
        """
        Method that applies memristor element to encoded quantum state. Takes as input a single quantum
        state sequence in density operator form 
        """

        # some basic data checking to catch annoying to track errors
        if len(input_states) != self.tlen:
            print("Dimension mismatch: Specified time series length does not match data length {}!={}".format(self.tlen, len(input_states)))

        # iteratively apply memristor for each time interval and produce next output state in sequence
        for i in range(self.tlen):
            # get input for time step
            time_input = np.squeeze(input_states[i,:,:])

            # get unitary for this time step 
            unitary = self.unitary_array[i,:,:]

            # apply unitary to density operator
            output_state = unitary @ time_input @ dagger(unitary)

            # apply unitary transform as left hand operator
            #left = np.einsum('ij,jl->il', unitary, time_input)u
            # right hand operator
            #out = np.einsum('il,lj->ij', left, dagger(unitary))
            
            # apply projection operators to input state
            #compute effect of  measurement projector on modes specified by photonic projectors 
            left = np.einsum('ijk,kl->ijl', self.proj, output_state)
            # can skip adjoint calculation since projectors are all real diagonals (does that seem right?)
            right = np.einsum('ijl,ilk->ijk', left, self.proj) #TODO
            # collapse projector outcome sum into single array, taking of batch broadcasting rules
            out = np.sum(right, axis=0)

            # add state to output tensor
            input_states[i,:,:] = out

            if i+1<self.tlen:
                # compute the average number of photons in measurement mode
                self.mem_states[i+1] = np.real(np.sum([np.trace(out @ self.proj[j,:,:])*self.proj_photon[j] for j in range(len(self.proj))]))

                # now normalise and scale to total number of photons
                self.mem_states[i+1] *= 2*np.pi/self.photons

                # build next unitary operator in chain
                self.update(i+1)

        #print(self.mem_states[-1])
        # return (probably) coherent output state 
        return input_states

    def update(self, time_index):
        """
        Updates state of internal network and rebuilds unitary with updated beam splitter values
        """

        # average of previous outputs
        theta = self.mem_states[time_index]#np.sum(self.mem_states + self.pdesc['theta_init'])/len(self.mem_states) #  #
        # modes to act on 
        self.targets = self.pdesc['MZI_target']
        # define stateful unitary operator that applies current state of MZI to input 
        unitary = T(self.targets[0], self.targets[1], theta, 0, self.modes)

        # map to complete Fock space and save to unitary array
        self.unitary_array[time_index,:,:] = self.S @ multikron(unitary, self.photons) @ np.transpose(self.S)




class BSArray():
    """
    Advanced layer that implements a stack of quantum memristors acting in parallel. 
    each memristor takes as input 4 modes - two being the input state, a third 
    being the detector input (instantiated to no input) and a final sacrifical mode
    that contains the destroyed photons. 

    pdesc = {'theta_init':None,'MZI_target': [m,m+1],'proj_mode': n!=m,m+1, 'sacrificial_mode': o!=n,m,m+1}
    """

    def __init__(self, modes, photons, force=False, **kwargs):
        # layer identifier
        self.id = "bsarray"
        # total number of modes
        self.modes = modes
        # total number of photons
        self.photons = photons
        # emergent dimension
        self.input_dim = self.output_dim = comb(
            self.modes+self.photons-1, self.photons, exact=True)
        # catch extreme cases
        if self.input_dim > 100 and not force:
            raise ValueError(
                "System dimension is large ({}), decrease system size or set force flag to True".format(self.input_dim))

        # build operator for class
        self.build()

        super(BSArray, self).__init__(**kwargs)

    def build(self):
        """
        Construct fully connected beam splitter array
        """

        # create beam splitter decomposition
        bms_spec = clements_phase_end(np.eye(self.modes))[0]

        # retest to beamsplitters with no phase effect (not that it matter)
        for bm in bms_spec:
            bm[2] = 0
            bm[3] = np.random.rand()

        # construct single photon unitary
        self.unitary = clements_stitch(bms_spec, [1.0]*self.modes)
        # map unitary to multiphoton space if required
        if self.photons > 1:
            # save symmetric map for dilation to complete Fock space (much, much faster than permanents)
            S = symmetric_map(self.modes, self.photons)

            self.unitary = S @ multikron(self.unitary, self.photons) @ np.transpose(S)

        # set build flag
        self.built = True


    def call(self, input_states):
        """This is where the layer's logic lives.
        Arguments:
            inputs: Input tensor, or list/tuple of input tensors.
            **kwargs: Additional keyword arguments.

        Returns:
            A tensor or list/tuple of tensors.
        """

        # apply unitary channel as left operator
        input_left = np.einsum('ij,bkjl->bkil', self.unitary, input_states)
        # now as right hand operator and return 
        input_states = np.einsum('bkil,lj->bkij', input_left, dagger(self.unitary))        

        return input_states



class MeasureLayer(Layer):
    """
    Advanced layer that implements a stack of quantum memristors acting in parallel. 
    each memristor takes as input 4 modes - two being the input state, a third 
    being the detector input (instantiated to no input) and a final sacrifical mode
    that contains the destroyed photons. 

    pdesc = {'theta_init':None,'MZI_target': [m,m+1],'proj_mode': n!=m,m+1, 'sacrificial_mode': o!=n,m,m+1}
    """

    def __init__(self, modes, photons, force=False, **kwargs):
        # layer identifier
        self.id = "measure"
        # total number of modes
        self.modes = modes
        # total number of photons
        self.photons = photons
        # emergent dimension
        self.input_dim = self.output_dim = comb(
            self.modes+self.photons-1, self.photons, exact=True)

        #self.output_dim /= 2

        # catch extreme cases
        if self.input_dim > 100 and not force:
            raise ValueError(
                "System dimension is large ({}), decrease system size or set force flag to True".format(self.input_dim))

        super(MeasureLayer, self).__init__(**kwargs)


    def build(self, input_shape):
        """
        Build method for layer
        """

        # set build flag
        self.built = True


    def call(self, inputs):
        """This is where the layer's logic lives.
        Arguments:
            inputs: Input tensor, or list/tuple of input tensors.
            **kwargs: Additional keyword arguments.

        Returns:
            A tensor or list/tuple of tensors.
        """
        # get diagonal elements - corresponds to photon counting 
        diag = tf.math.real(tf.linalg.diag_part(inputs))[:,::]

        #scalar = 1/tf.reduce_sum(diag, axis=1)

        # rescale and return
        return diag


    def compute_output_shape(self, input_shape):
        return tf.TensorShape([None, self.output_dim])

    def get_config(self):
        base_config = super(MeasureLayer, self).get_config()
        base_config['output_dim'] = self.output_dim
        base_config['modes'] = self.modes
        base_config['photons'] = self.photons
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @property
    def output_size(self):
        return [self.output_dim]


class LearningRateDecay():
    """ 
    """
    def plot(self, epochs, title="Learning Rate Schedule"):
        # compute the set of learning rates for each corresponding
        # epoch
        lrs = [self(i) for i in epochs]
 
        # the learning rate schedule
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(epochs, lrs)
        plt.title(title)
        plt.xlabel("Epoch #")
        plt.ylabel("Learning Rate")

class PolynomialDecay(LearningRateDecay):
    def __init__(self, maxEpochs=100, initAlpha=0.01, power=1.0):
        # store the maximum number of epochs, base learning rate,
        # and power of the polynomial
        self.maxEpochs = maxEpochs
        self.initAlpha = initAlpha
        self.power = power
 
    def __call__(self, epoch):
        # compute the new learning rate based on polynomial decay
        decay = (1 - (epoch / float(self.maxEpochs))) ** self.power
        alpha = self.initAlpha * decay
        
        # return the new learning rate
        return float(alpha)

def basis_generator(dim):
    """
    Generates a set of orthogonal basis elements in qubit basis
    """

    # generate Pauli basis
    basis = gellman_gen(dim)

    # normalised states
    state_encoding = np.zeros((dim**3-dim,dim),dtype=np.complex64)

    # extract eigenvectors
    cnt = 0
    for i in range(dim**2-1):
        w,v = np.linalg.eigh(basis[i,:,:])
        for j in range(dim): 
            state_encoding[cnt,:] = v[:,j]
            cnt +=1

    return state_encoding


class QEncoder(object):
    """
    Takes as input an NxMxW data array and applies a quantum encoding in given dimension
    """

    # setup class ready for data encoding
    def __init__(self, modes, photons, topology=None, density=False):
        # store all initialisation arguments
        # number of modes
        self.modes = modes
        # number of photons
        self.photons = photons
        # density operator output flag
        self.density = density
        # store topology (only needed for milano or other specific encoding)
        self.topology = topology
        # calculate system dimension
        self.dim = qop.fock_dim(self.modes, self.photons)
        # define random vector map
        self.vector_map = None

    def _granulate(self, data, levels=2, positive=True):
        """
        Applies a granulation function to input data as preporcessing for the encoder. Assumes real data.
        """
        # rescale data from [-infty, infty] range into zero to one using logistic function
        data_parse = 1/(1+np.exp(-data))

        # rescale again if strictly positive range
        if positive:
            # logistic(0) = 0.5
            data_parse -= 0.5
            # renormalise
            data_parse /= np.max(data_parse)

        # now granulate according to desired number of levels
        inc = 1/levels
        lbnd = 0.0
        ubnd = 1/levels
        for i in range(levels):
            # assign value to interval
            truth_mask = (data_parse<ubnd) & (data_parse>lbnd)
            data_parse[truth_mask] = i
            # update 
            lbnd += inc 
            ubnd += inc

        # return as smallest non-boolean type available
        return data_parse.astype(np.uint8)   

    def encode(self, data, method="amplitude", normalise=True):
        """
        Performs requested encoding as well as basic data parsing
        """

        # normalise if requested and be noisy about it
        if normalise:
            print("Data normalisation has been requested")
            data /= np.max(data)

        # separate map mathods for compartmentalisation
        if method is "eigen":
            print("Applying Gellman basis eigenstate map")
            return self._eigen_map(data=data)

        elif method is "milano":
            print("Applying milano topological map")
            return self._milano_map(data=data)

        elif method is "amplitude":
            
            if self.vector_map is None:
                print("Random vector map not initialised, creating one now")
                # get shape of input data
                _,in_dim,_ = np.shape(data)
                # generate a random map 
                self.vector_map = np.random.choice(range(self.dim), size=in_dim)

            print("Applying analog amplitude map")
            return self._amplitude_map(data=data)

        else:
            raise ValueError("Invalid encoding map {} requested".format(method))

    def _eigen_map(self, data, rand_encoding=False):
        """
        Takes as input an NxMxW data array and applies a quantum encoding in given dimension
        """

        # sanitise data to be accepted to our encoding scheme
        data = self._granulate(data, levels=2)

        # get shape of array and package as instance, input dimension and time series length
        N,in_dim,tlen = np.shape(data)

        # generate state encodings - assume modes and photons are enough
        encodings = basis_generator(self.dim)

        # perform in-place shuffle of encoding states if requested 
        if rand_encoding:
            np.random.shuffle(encodings)

        # assert encoding is sufficent assuming binary data
        assert 2**in_dim<=len(encodings), "Woah hey woah, your system size is not sufficient to encode all states: {}<{}".format(len(encodings), 2**in_dim) 

        # preallocate output encoding array as vector state or density operator
        if self.density:
            data_encode = np.zeros((N,tlen,self.dim,self.dim), dtype=np.complex64)
        else:
            data_encode = np.zeros((N,tlen,self.dim), dtype=np.complex64)

        # no doubt there is a smarter way of computing this but it doesn't matter for such small problem sizes
        for i in range(N):
            if i % 100==0:
                print("{}/{} images encoded in quantum state space".format(i,N), end="\r")
            for j in range(tlen):
                # get classical state
                classical_state = data[i,:,j]

                # compute index of state to map it to
                ind = int(''.join(str(s) for s in classical_state), 2)

                # map classical binary data to a quantum state
                if self.density:
                    state = encodings[ind,:].reshape([-1,1])
                    data_encode[i,j,:,:] = np.kron(state, dagger(state))
                else:
                    data_encode[i,j,:] = encodings[ind,:]

        return data_encode


    def _amplitude_map(self, data, levels=0,):
        """
        Generates a standard amplitude encoding scheme. Hard to prepare in practice
        but acceptable for prototyping.
        """

        # get shape of array and package as instance, input dimension and time series length
        N,in_dim,tlen = np.shape(data)

        # check whether to apply any kind of granulation effect
        if levels>0:
            print("Parse number set: applying logistic scaling and granulation")
            data = self._granulate(data, levels=levels)

        # preallocate output encoding array as vector state or density operator
        if self.density:
            data_encode = np.zeros((N,tlen,self.dim,self.dim), dtype=np.complex64)
        else:
            data_encode = np.zeros((N,tlen,self.dim), dtype=np.complex64)

        # generate computational basis elements
        basis = np.eye(self.dim, dtype=np.complex128)

        # convert each data element - likely a smart way to vectorise it but its not worth it here
        for i in range(N):
            if i % 100==0:
                print("{}/{} images encoded in quantum state space using amplitude map scheme".format(i,N), end="\r")
            
            # iterate over each column 
            for t in range(tlen):
                # construct a new state vector
                state = np.zeros((self.dim,), dtype=np.complex128)
                # phase encoding to random states
                state[self.vector_map] = np.exp(1j*data[i,:,t]*2*np.pi)
                # normalise state vector
                state /= np.linalg.norm(state, 2)
                
                # assign to output set
                if self.density:
                    state = state.reshape([-1,1])
                    data_encode[i,t,:,:] = np.kron(state, dagger(state))
                else:
                    data_encode[i,t,:] = state

        print()
        return data_encode


    def _milano_topology_gen(self):
        """
        Creates a topology specifier for the Milano chip.
        """

        # Topology is fixed, thus hardcoding
        first_layer = [[2,3],[6,7,],[10,11]]
        second_layer = [[1,2],[3,4],[5,6],[7,8],[9,10],[11,12]]
        third_layer = [[2,3],[4,5],[6,7],[8,9],[10,11]]
        fourth_layer = [[1,2],[3,4],[5,6],[7,8],[9,10],[11,12]]

        # compile topology list and return in reverse order 
        return fourth_layer+third_layer+second_layer+first_layer


    def _milano_map(self, data, parse=False, masks=None, input_state=None, base=[0,2*np.pi/3,4*np.pi/3]):
        """
        A highly specific encoding scheme using Milano optical chip - use eigenstate or amplitude encoding unless you know 
        what you are doing!

        This encoding assumes a constant input state and then configures an optical map that  
        """
        # get shape of array and package as instance, input dimension and time series length
        N,in_dim,tlen = np.shape(data)

        # granulate data to be accepted to our encoding scheme

        if parse:
            print("Parse flag set: applying logistic scaling and granulation")
            data = self._granulate(data, levels=len(base))

        # generate initial input state if not defined (assumes mode and photon numbers)
        if input_state is None:
            # 12 modes, 3 photons
            nstate = [0,0,1,0,0,0,1,0,0,0,1,0]
            # make use of qfunk optical library
            input_state = qop.optical_state_gen(m_num=self.modes, p_num=self.photons, nstate=nstate, density=False)

        # generate topology if not set
        if self.topology is None:
            self.topology = self._milano_topology_gen()

        # generate a set of masks if none are given
        if masks is None:
            masks = mask_generator(in_dim, 3)

        # initialise a dictionary so generated gates don't have to be rebuilt
        gate_collector = {}

        # preallocate output encoding array as vector state or density operator
        if self.density:
            data_encode = np.empty((N,tlen,self.dim,self.dim), dtype=np.complex128)
        else:
            data_encode = np.empty((N,tlen,self.dim), dtype=np.complex128)


        # iterate over each and every input image
        for i in range(N):
            if i % 100==0:
                print("{}/{} images encoded in quantum state space using Milano scheme".format(i,N), end="\r")
            # iterate over each column
            for t in range(tlen):
                # get the classical data to be encoded
                classical_state = list(data[i,:,t])
                
                # convert to string description
                state_string = ''.join(str(s) for s in classical_state)

                # check if we have already generated the resultant gate
                if state_string in gate_collector:
                    U = gate_collector[state_string]
                else:
                    # compute trinary mapping 
                    phase_shifters = [base[el] for el in classical_state]

                    # apply weighted mask for each pre-encoder MZI
                    for mask in masks:
                        # compute maximum value possible for mask
                        max_val = np.sum(mask*(len(base)-1))
                        # compute unscaled phase value for pre-encoder
                        phase_val = (base[-1]-base[0])*mask @ classical_state
                        # add to end of phase shifters and rescale with range
                        phase_shifters.append(phase_val/max_val)


                    # now iterate over topology to generate encoder

                    # compute unitary from values
                    #T(self.targets[0], self.targets[1], theta, theta, self.modes)
                    U = np.eye(self.dim)

                    # save computed unitary for repeated use
                    gate_collector[state_string] = U

                # apply computed unitary to input state
                state = U @ input_state

                if self.density:
                    data_encode[i,t,:,:] = np.kron(state, dagger(state))
                else:
                    data_encode[i,t,:] = state

        return data_encode




def eigen_encode_map(data, modes, photons, rand_encoding=False, density=True):
    """
    Takes as input an NxMxW data array and applies a quantum encoding in given dimension
    """

    # get shape of array and package as instance, input dimension and time series length
    N,in_dim,tlen = np.shape(data)

    # calculate system dimension
    dim = comb(modes+photons-1, photons, exact=True)

    # generate state encodings - assume modes and photons are enough
    encodings = basis_generator(dim)

    # perform in-place shuffle of encoding states if requested 
    if rand_encoding:
        np.random.shuffle(encodings)

    # assert encoding is sufficent assuming binary data
    assert 2**in_dim<=len(encodings), "Woah hey woah, your system size is not sufficent to encode all states: {}<{}".format(len(encodings), 2**in_dim) 

    # preallocate output encoding array as vector state or density operator
    if density:
        data_encode = np.zeros((N,tlen,dim,dim), dtype=np.complex64)
    else:
        data_encode = np.zeros((N,tlen,dim), dtype=np.complex64)

    # no doubt smarter way of computing this but it doesn't matter for the problem sizes we consider
    for i in range(N):
        if i % 100==0:
            print("{}/{} images encoded in quantum state space".format(i,N), end="\r")
        for j in range(tlen):
            # get classical state
            classical_state = data[i,:,j]

            # compute index of state to map it to
            ind = int(''.join(str(s) for s in classical_state), 2)

            # map classical binary data to a quantum state
            if density:
                state = encodings[ind,:].reshape([-1,1])
                data_encode[i,j,:,:] = np.kron(state, dagger(state))
            else:
                data_encode[i,j,:] = encodings[ind,:]
    print()
    return data_encode



def reservoir_map(data, modes, photons, pdesc, targets, temporal_num):
    """
    Applies resevoir channel to input data sequence
    """
    # generate a Hadamard channel first
    hadamard_layer = BSArray(modes, photons, force=True) 
    
    # iterate through each layer of resevoir
    for layer_num in range(temporal_num):
        
        # apply hadamard channel
        print("Applying Hadamard layer")
        data = hadamard_layer.call(data)

        # iterate over list of targets for resevoir layer
        for tnum,target in enumerate(targets):
            continue
            # generate memristor element class
            pdesc["MZI_target"] = target[:2]
            pdesc["proj_mode"] = target[-1]
            qmemristor = QMemristor(modes=modes, photons=photons, pdesc=pdesc, force=True)

            # apply layer element to every data sample
            for state_num in range(len(data)):
                if state_num % 10 ==0:
                    print("Computing {}/{}:{}/{}:{}/{} component of reservoir channel".format(state_num, len(data), tnum, len(targets), layer_num, temporal_num), end="\r")
                # propagate all input data through network, one sample at a time
                data[state_num,:,:,:] = qmemristor.call(data[state_num,:,:,:])
            
    print()
    # only need last output state as a time series
    return np.squeeze(data[:,-1,:,:])



def filter_36(x, y):
    keep = (y == 0) | (y==2) | (y==3)# | (y == 4) | (y == 5) | (y==6)
    x, y = x[keep], y[keep]
    return x,y


def integer_remap(data, maps):
    """
    Sets all values in data to values in maps - not for floats
    """

    for item in maps:
        # set all values of a particular kind to another
        data[data==item[0]] = item[1]
    return data



def remove_contradicting(xs, ys):
    mapping = collections.defaultdict(set)
    # Determine the set of labels for each unique image:
    for x,y in zip(xs,ys):
       mapping[tuple(x.flatten())].add(y)
    
    new_x = []
    new_y = []
    for x,y in zip(xs, ys):
      labels = mapping[tuple(x.flatten())]
      if len(labels) == 1:
          new_x.append(x)
          new_y.append(list(labels)[0])
      else:
          # Throw out images that match more than one label.
          pass
    
    num_3 = sum(1 for value in mapping.values() if True in value)
    num_6 = sum(1 for value in mapping.values() if False in value)
    num_both = sum(1 for value in mapping.values() if len(value) == 2)

    print("Number of unique images:", len(mapping.values()))
    print("Number of 3s: ", num_3)
    print("Number of 6s: ", num_6)
    print("Number of contradictory images: ", num_both)
    print()
    print("Initial number of examples: ", len(xs))
    print("Remaining non-contradictory examples: ", len(new_x))
    
    return np.array(new_x), np.array(new_y)



def binarise(data):
    """
    Converts input image data to a classical binary representation. assumes uint8 input
    """

    data *= 1/np.max(data)
    # set floor and ceiling values
    data[data>0.5] = 1
    data[data<=0.5] = 0

    return data.astype(np.uint8)



def probfid2(rho, gamma):
    """
    Computes output probability distribution of two density operators and computes
    their relative entropy
    """
    return tf.keras.losses.KLD(rho, gamma)

def ent_gen(dim, vec=False):
    """
    Generates a maximally entangled bi-partite system each of dimension dim
    """ 

    # pre allocate entangled state array
    ent = np.zeros((dim**2,1),dtype=np.complex128)

    # iterate over each basis element
    for i in range(dim):
        # generate computaional basis element
        comput_el = np.zeros((dim, 1), dtype=np.complex128)
        # assign value 
        comput_el[i] = 1.0

        # add to state
        ent += np.kron(comput_el, comput_el)

    if vec:
        return ent
    else:
        return np.kron(ent, dagger(ent))/dim

def entanglement_gen(dim, num=100, partition=0.5, embed_dim=None):
    """
    Generates a set of training and testing data for bipartite entanglement witnesses.
    dim gives the dimension of the subsystem and embed dim places the generated states
    into larger Hilbert space
    """


    # generate entangled state
    ent_state = ent_gen(dim)

    # generate base seperable state
    sep_state = np.ones((dim**2, dim**2),dtype=np.complex128)/dim**2

    # check if embedding must occur
    if embed_dim is not None:
        assert embed_dim>=dim**2, "embedded dimension is of insufficient size"
        data_dim = embed_dim
    else:
        data_dim = dim**2

    # initialise data constructs
    ent_states = np.zeros((num,1, data_dim, data_dim), dtype=np.complex128)
    sep_states = np.zeros((num,1, data_dim, data_dim), dtype=np.complex128)

    # iterate over it all
    for i in range(num):
        # generate random local unitaries
        #U_ent = np.kron(np.squeeze(random_U(dim=dim, num=1)), np.squeeze(random_U(dim=dim, num=1)))
        U_sep = np.kron(np.squeeze(random_U(dim=dim, num=1)), np.squeeze(random_U(dim=dim, num=1)))
        # apply to entangled state and add to collection
        ent_states[i,0,:dim**2,:dim**2] = np.squeeze(haar_sample(dim=dim**2,num=1)) #U_ent @ ent_state @ dagger(U_ent)
        # apply seperable state and add to collection
        sep_states[i,0,:dim**2,:dim**2] = U_sep @ sep_state @ dagger(U_sep)

    # generate labels
    sep_labels = np.zeros((num, 1), dtype=int)
    ent_labels = np.ones((num, 1), dtype=int)

    # concatenate everything
    entangled_data = np.concatenate([sep_states, ent_states], axis=0)
    entangled_labels = np.concatenate([sep_labels, ent_labels], axis=0)

    # shuffle everything using repeateable rng state
    seed_state = np.random.get_state()
    np.random.shuffle(entangled_data)
    np.random.set_state(seed_state)
    np.random.shuffle(entangled_labels)

    # apply partition
    partition_index = int(partition*len(entangled_data))
    # training data
    train_data = entangled_data[:partition_index]
    train_label = entangled_labels[:partition_index]

    # testing data
    test_data = entangled_data[partition_index:]
    test_label = entangled_labels[partition_index:]

    # return it all
    return train_data,train_label,test_data,test_label




if __name__ == '__main__':
    pdesc = {'modes':3, 'photons':2}
    modes = 3
    photons = 2
    print(number_states(modes, photons))
    new_data = encode_map(data, modes, photons)
