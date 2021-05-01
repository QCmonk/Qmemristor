
class QEncoder(object):
    """
    Takes as input an NxMxW data array and applies a quantum encoding in given dimension
    """

    # setup class ready for data encoding
    def __init__(self, modes, photons, topology=None, rand_encoding=False, density=False):
        # store all initialisation arguments
        # number of modes
        self.modes = modes
        # number of photons
        self.photons = photons
        # random encoding flag
        self.rand_encoding = rand_encoding
        # density operator output flag
        self.density = density
        # store topology (only needed for milano or other specific encoding)
        self.topology = topology
        # calculate system dimension
        self.dim = qop.fock_dim(self.modes, self.photons)


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

    def encode(self, data, method="amplitude", normalise=True, **kwargs):
        """
        Performs requested encoding as well as basic data parsing
        """

        # normalise if requested and be noisy about it
        if normalise:
            print("Data normalisation requested: Computing")
            data /= np.max(data)

        # separate map mathods for compartmentalisation
        if method is "eigen":
            print("Applying Gellman basis eigenstate map")
            return self._eigen_map(data=data, **kwargs)

        elif method is "milano":
            print("Applying milano topological map")
            return self._milano_map(data=data, **kwargs)

        elif method is "amplitude":
            print("Applying analog amplitude map")
            return self._amplitude_map(data=data, **kwargs)

        else:
            raise ValueError("Invalid encoding map {} requested".format(method))

    def _eigen_map(self, data):
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
                state[:in_dim] = data[i,:,t]

                # normalise state vector
                state /= np.linalg.norm(state, 2)

                # assign to output set
                if self.density:
                    state = state.reshape([-1,1])
                    data_encode[i,t,:,:] = np.kron(state, dagger(state))
                else:
                    data_encode[i,t,:] = state

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


