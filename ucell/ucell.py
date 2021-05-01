from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import math
import numpy as np

from scipy.special import comb
from keras import optimizers
from keras.layers import Input, Layer
from keras.models import Model
from keras import callbacks
from keras import backend
from datetime import datetime
from qinfo.qinfo import haar_sample, dagger, multikron, dirsum
from ucell.utility import *
from ucell.operators import *

# For the love of god, please be quiet
#tf.logging.set_verbosity(tf.logging.ERROR)


class ReNormaliseLayer(Layer):
    """
    Performs a renormalisation process for a set of input variables.
    """

    def __init__(self, dim, **kwargs):
        # layer identifier
        self.id = "renorm"
        # store input/output dimension
        self.input_dim = self.output_dim = dim
        # pass additional keywords to superclass initialisation
        super(ReNormaliseLayer, self).__init__(**kwargs)


    def build(self, input_shape):
        """Creates the variables of the layer (optional, for subclass implementers).

        This is a method that implements of subclasses of `Layer` or `Model`
        can override if they need a state-creation step in-between
        layer instantiation and layer call.

        This is typically used to create the weights of `Layer` subclasses.

        Arguments:
          input_shape: Instance of `TensorShape`, or list of instances of
            `TensorShape` if the layer expects a list of inputs
            (one instance per input).
        """

        # call build method of super class
        super(ReNormaliseLayer, self).build(input_shape)

    def call(self, inputs):
        """
        This is where the layer's logic lives.
        Arguments:
            inputs: Input tensor, or list/tuple of input tensors.
            **kwargs: Additional keyword arguments.

        Returns:
            A tensor or list/tuple of tensors.
        """
        # extract input tensors
        u_params = inputs[0]
        norms = inputs[1]

        # roll back normalisation effect done in preprocessing
        out = tf.einsum('ij,i->ij', u_params, norms)

        return out

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0], self.output_dim), shape_b[:-1]]

    def get_config(self):
        base_config = super(ReNormaliseLayer, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @property
    def output_size(self):
        return [self.output_dim, self.output_dim]


class ULayer(Layer):
    """
    Subclass Keras because I'm a busy guy. Untitary layer using Clements 2016 decomposition, universal 
    for single photon input. 
    """
    # initialise as subclass of keras general layer description
    def __init__(self, modes, photons=1, dim=None, u_noise=None, pad=0, vec=False, full=False, force=False, **kwargs):
        # layer identifier
        self.id = "unitary"
        # pad dimensions
        self.pad = pad 
        # number of modes and number of pad modes
        self.modes = modes + self.pad
        # number of variables in operator
        self.vars = (self.modes**2 - self.modes)//2
        # number of photons
        self.photons = photons
        # whether to implement on full finite Fock space
        self.full = full # not currently used
        # pad modes - becomes dimension with 1 photon which we generally desire for universality
        self.pad = pad
        # whether to expect vector inputs
        self.vec = tf.constant(vec)
        # emergent dimension
        self.input_dim = self.output_dim = comb(self.modes+self.photons-1, self.photons, exact=True) + self.pad
        # catch extreme cases
        if self.input_dim > 100 and not force:
            raise ValueError(
                "System dimension is large ({}), decrease system dimension or set force flag to True".format(self.input_dim))
        
        # keep local copy of beam splitter decomposition table
        # TODO: This is such a cop out
        self.bms_spec = clements_phase_end(np.eye(self.modes))[0]

        # pass additional keywords to superclass initialisation
        super(ULayer, self).__init__(**kwargs)


    def build(self, input_shape):
        """Creates the variables of the layer (optional, for subclass implementers).

        This is a method that implements of subclasses of `Layer` or `Model`
        can override if they need a state-creation step in-between
        layer instantiation and layer call.

        This is typically used to create the weights of `Layer` subclasses.

        Arguments:
          input_shape: Instance of `TensorShape`, or list of instances of
            `TensorShape` if the layer expects a list of inputs
            (one instance per input).
        """

        # define weight initialisers with very tight distribution - corresponds to an identity
        with tf.init_scope():
            diag_init = tf.initializers.RandomNormal(mean=0, stddev=0.01)
            theta_init = tf.initializers.RandomNormal(mean=np.pi, stddev=0.01)
            phi_init = tf.initializers.RandomNormal(mean=-np.pi, stddev=0.01)

            # superclass method for variable construction, get_variable has trouble with model awareness
            self.diag = self.add_weight(name="diags",
                                        shape=[self.modes],
                                        dtype=tf.float32,
                                        initializer=diag_init,
                                        trainable=True)

            self.theta = self.add_weight(name='theta',
                                         shape=[self.vars],
                                         dtype=tf.float32,
                                         initializer=theta_init,
                                         trainable=True)

            self.phi = self.add_weight(name='phi',
                                       shape=[self.vars],
                                       dtype=tf.float32,
                                       initializer=phi_init,
                                       trainable=True)

        # add weights to layer
        # construct single photon unitary
        self.unitary = tf_clements_stitch(
            self.bms_spec, self.theta, self.phi, self.diag)

        # construct multiphoton unitary using memory hungry (but uber fast) method
        if self.photons > 1:
            if self.full:
                # preallocate on zero dimensional subspace
                U = tf.linalg.LinearOperatorFullMatrix(tf.constant([[1.0]], dtype=tf.complex64))

                for pnum in range(1,self.photons+1):
                    # use symmetric map to compute multi photon unitary
                    S = tf.constant(symmetric_map(
                    self.modes, pnum), dtype=tf.complex64)
                    # map to product state then use symmetric isometry to reduce to isomorphic subspace
                    V = tf.matmul(S, tf.matmul(tf_multikron(self.unitary, pnum), tf.linalg.adjoint(S)))
                    U = tf.linalg.LinearOperatorBlockDiag([U, tf.linalg.LinearOperatorFullMatrix(V)])
                # convert unit
                self.unitary = tf.convert_to_tensor(U.to_dense())

            else:
                # use symmetric map to compute multi photon unitary
                S = tf.constant(symmetric_map(
                    self.modes, self.photons), dtype=tf.complex64)
                # map to product state then use symmetric isometry to reduce to isomorphic subspace
                self.unitary = tf.matmul(S, tf.matmul(tf_multikron(self.unitary, self.photons), tf.linalg.adjoint(S)))

        # adds identity channels to operator - not clear what use I thought these would be
        # if self.pad is not None:
        #     # pad operator dimension as many times as requested
        #     pad_op = [tf.constant([[1.0]], dtype=tf.complex64)]*self.pad
        #     pad_op.append(self.unitary)

        #     # perform block diagonal operator
        #     linop_blocks = [tf.linalg.LinearOperatorFullMatrix(block) for block in pad_op]
        #     self.unitary = tf.convert_to_tensor(tf.linalg.LinearOperatorBlockDiag(linop_blocks).to_dense())

        # call build method of super class
        super(ULayer, self).build(input_shape)

    def call(self, inputs):
        """
        This is where the layer's logic lives.
        Arguments:
            inputs: Input tensor, or list/tuple of input tensors.
            **kwargs: Additional keyword arguments.

        Returns:
            A tensor or list/tuple of tensors.
        """
        # perform matrix calculation using multiple einsums


        #if self.vec:
        inputs = tf.complex(inputs, 0.0)
        out = tf.einsum('ij,bj->bi', self.unitary,
                          inputs, name="Einsum_left")
        out = tf.math.real(out)
        # else:
            
        #     leftm = tf.einsum('ij,bjl->bil', self.unitary,
        #                       inputs, name="Einsum_left")
        #     out = tf.einsum('bil,lj->bij', leftm,
        #                        tf.linalg.adjoint(self.unitary), name="Einsum_right")
        return out

    def invert(self):
        """
        Computes the inverse of the unitary operator
        """
        # compute transpose of unitary
        self.unitary = tf.tranpose(self.unitary, conjugate=True, name="dagger_op")


    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(ULayer, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @property
    def output_size(self):
        return [self.output_dim, self.output_dim]


class InvertibleLeakyReLU(Layer):
    """Leaky version of a Rectified Linear Unit.

    It allows a small gradient when the unit is not active:
    `f(x) = alpha * x for x < 0`,
    `f(x) = x for x >= 0`.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as the input.

    # Arguments
        alpha: float >= 0. Negative slope coefficient.

    # References
        - [Rectifier Nonlinearities Improve Neural Network Acoustic Models](
           https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf)
    """

    def __init__(self, alpha=0.3, **kwargs):
        super(InvertibleLeakyReLU, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha = K.cast_to_floatx(alpha)

    def call(self, inputs):
        return K.relu(inputs, alpha=self.alpha)

    def get_config(self):
        config = {'alpha': float(self.alpha)}
        base_config = super(InvertibleLeakyReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class UParamLayer(Layer):
    """
    Paramterised MZI that takes progamming and input state as input

        targets: modes to act on, assumes and enforces sorted list pairs, indexing starting at 1
    
        Set model's input and output specs based on the input data received.

        This is to be used for Model subclasses, which do not know at instantiation
        time what their inputs look like.

        # Arguments
          inputs: Single array, or list of arrays. The arrays could be placeholders,
            Numpy arrays, or data tensors.
            - if placeholders: the model is built on top of these placeholders,
              and we expect Numpy data to be fed for them when calling `fit`/etc.
            - if Numpy data: we create placeholders matching the shape of the Numpy
              arrays. We expect Numpy data to be fed for these placeholders
              when calling `fit`/etc.
            - if data tensors: the model is built on top of these tensors.
              We do not expect any Numpy data to be provided when calling `fit`/etc.
          outputs: Optional output tensors (if already computed by running
            the model).
          training: Boolean or None. Only relevant in symbolic mode. Specifies
            whether to build the model's graph in inference mode (False), training
            mode (True), or using the Keras learning phase (None).
    """
    # initialise as subclass of keras general layer description
    def __init__(self, modes, photons, targets, force=False, **kwargs):
        # layer identifier
        self.id = "param_unitary"
        # total number of modes present on chip
        self.modes = modes
        # number of single photons to be sent into system
        self.photons = photons
        # number of MZIs
        self.element_num = len(targets)
        # store MZI target modes
        self.targets = targets

        # sort mode pairs into ascending order
        for i,pair in enumerate(targets):
            # check for basic problems before getting to Tensorflow's abysmal bug reporting
            if max(pair)>modes:
                raise ValueError("One or more pair targets is greater than specified number of modes: {}>{}".format(pair,self.modes))
            pair.sort()
            # sort and assign to target list
            self.targets[i] = pair

        # compute and store dimension of Hilbert space isomorphic to symmetric Fock subspace
        self.input_dim = self.output_dim = comb(self.modes+self.photons-1, self.photons, exact=True)
        
        # catch extreme cases
        if self.input_dim > 100 and not force:
            raise ValueError(
                "System dimension is large ({}), decrease system dimension or set force flag to True".format(self.input_dim))
        
        # pass additional keywords to superclass initialisation
        super(UParamLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """Set model's input and output specs based on the input data received.

        This is to be used for Model subclasses, which do not know at instantiation
        time what their inputs look like.

        # Arguments
          inputs: Single array, or list of arrays. The arrays could be placeholders,
            Numpy arrays, or data tensors.
            - if placeholders: the model is built on top of these placeholders,
              and we expect Numpy data to be fed for them when calling `fit`/etc.
            - if Numpy data: we create placeholders matching the shape of the Numpy
              arrays. We expect Numpy data to be fed for these placeholders
              when calling `fit`/etc.
            - if data tensors: the model is built on top of these tensors.
              We do not expect any Numpy data to be provided when calling `fit`/etc.
          outputs: Optional output tensors (if already computed by running
            the model).
          training: Boolean or None. Only relevant in symbolic mode. Specifies
            whether to build the model's graph in inference mode (False), training
            mode (True), or using the Keras learning phase (None).
        """

        # compute and save a blueprint of the desired optical setup
        self.bms_spec = opto_gen(self.modes, self.targets)

        #TODO: Should perform the symmetric mapping here and save that rather than reperforming it every time the 
        # channel is called
        # use symmetric map to compute multi photon unitary
        if self.photons>1:
            self.S = tf.constant(symmetric_map(self.modes, self.photons), dtype=tf.complex64)
            self.Sadj = tf.linalg.adjoint(self.S)
        
        # call build method of super class
        super(UParamLayer, self).build(input_shape)

    def call(self, inputs):
        """Set model's input and output specs based on the input data received.

        This is to be used for Model subclasses, which do not know at instantiation
        time what their inputs look like.

        # Arguments
          inputs: Single array, or list of arrays. The arrays could be placeholders,
            Numpy arrays, or data tensors.
            - if placeholders: the model is built on top of these placeholders,
              and we expect Numpy data to be fed for them when calling `fit`/etc.
            - if Numpy data: we create placeholders matching the shape of the Numpy
              arrays. We expect Numpy data to be fed for these placeholders
              when calling `fit`/etc.
            - if data tensors: the model is built on top of these tensors.
              We do not expect any Numpy data to be provided when calling `fit`/etc.
          outputs: Optional output tensors (if already computed by running
            the model).
          training: Boolean or None. Only relevant in symbolic mode. Specifies
            whether to build the model's graph in inference mode (False), training
            mode (True), or using the Keras learning phase (None).
        """
        
        # ensure input is list
        # if not isinstance(inputs, list):
        #     raise TypeError("Input to UParamLayer must be list of tensors containg input states and Unitary spec, instead is {}".format(type(inputs)))

        # segregate inputs - why the fuck is the operator being rebuilt every time!?
        params = inputs[0]
        input_state = inputs[1]
        
        # construct batch of unitaries given optical blueprint and input parameters, parallelise on batches 
        self.unitary = tf.map_fn(fn=self.map_clements, 
                                  elems=params, 
                                  dtype=tf.complex64, 
                                  back_prop=True, 
                                  parallel_iterations=True,
                                  name="Optical_Map")
        
        # construct multiphoton unitary using memory hungry method
        # if self.photons > 1:
        #     # map to product state then use symmetric isometry to reduce to isomorphic subspace
        #     self.unitary = tf.matmul(self.S, tf.matmul(tf_multikron(
        #         self.unitary, self.photons), self.Sadj))

        # perform matrix calculation using multiple einsums
        leftm = tf.einsum('bij,bjl->bil', self.unitary,
                          input_state, name="Einsum_left")
        rightm = tf.einsum('bil,blj->bij', leftm,
                           tf.linalg.adjoint(self.unitary), name="Einsum_right")
        
        return rightm

    def map_clements(self, params):
        """
        gives a map_fn compatible version of tf_clements_stitch
        """

        # segregate theta and phi parameters of the MZI
        theta = params[:self.element_num]
        phi = params[self.element_num:2*self.element_num]


        # compute unitary by hijacking clements decomposition code
        return tf_clements_stitch(beam_spec=self.bms_spec, 
                                  theta=theta, phi=phi, 
                                  diag=tf.constant([0.0]*self.modes, dtype=tf.float32),
                                  rev=tf.constant(False))

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0], self.output_dim), shape_b[:-1]]

    def get_config(self):
        base_config = super(UParamLayer, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @property
    def output_size(self):
        return [self.output_dim, self.output_dim]


class IsometricLayer(Layer):
    """
    Advanced layer that acts as a kind of partial trace operation, decreasing the effective dimensionality
    of the output state in a non-trivial way. Force maps all photons detected in the ancilla modes into mode 1, 
    so any variation in this scratch space is disregarded by the optimiser while maintaining the target modes and
    keeping the state space dimension constant. 

    """

    def __init__(self, modes, photons, idesc, force=False, **kwargs):
        # layer identifier
        self.id = "isometric"
        # number of modes
        self.modes = modes
        # number of variables on SU(modes)
        self.vars = (self.modes**2 - self.modes)//2
        # number of photons
        self.photons = photons
        # projection operator description
        self.idesc = idesc
        # emergent dimension
        self.input_dim = self.output_dim = comb(
            self.modes+self.photons-1, self.photons, exact=True)
        # catch extreme cases
        if self.input_dim > 100 and not force:
            raise ValueError(
                "System dimension is large ({}), decrease system dimension or set force flag to True".format(self.input_dim))

        super(IsometricLayer, self).__init__(**kwargs)


    def build(self, input_shape):
        """
        Construct projection operator for layer - non-trainable.
        Creates the variables of the layer (optional, for subclass implementers).

        This is a method that implements of subclasses of `Layer` or `Model`
        can override if they need a state-creation step in-between
        layer instantiation and layer call.

        This is typically used to create the weights of `Layer` subclasses.

        Arguments:
          input_shape: Instance of `TensorShape`, or list of instances of
            `TensorShape` if the layer expects a list of inputs
            (one instance per input).

        """

        # returns an m x dim x dim array with m being the number of projectors to apply
        self.iso = iso_gen(self.idesc, convert=True)

        # call build method of super class
        super(IsometricLayer, self).build(input_shape)


    def call(self, inputs):
        """This is where the layer's logic lives.
        Arguments:
            inputs: Input tensor, or list/tuple of input tensors.
            **kwargs: Additional keyword arguments.

        Returns:
            A tensor or list/tuple of tensors.
        """
        # compute effect of  measurement projector on modes specified by photonic projectors 
        left = tf.einsum('ijk,bkl->ibjl', self.iso, inputs, name='Projection_Left')
        # can skip adjoint calculation since projectors are all real diagonals (does that seem right?)
        right = tf.einsum('ibjl,ilk->ibjk', left, tf.linalg.adjoint(self.iso))
        # collapse projector outcome sum into single, taking of batch broadcasting rules
        out = tf.reduce_sum(right, axis=[0], name='Projector_sum')

        return out


    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(IsometricLayer, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @property
    def output_size(self):
        return [self.output_dim, self.output_dim]


class ProjectionLayer(Layer):
    """
    Advanced layer that acts as a photonic detector on specified nodes. Allows for the possibility 
    of returning to a pure state from a mixed one, a counter to the nonlinear layer.
    """

    def __init__(self, modes, photons, pdesc, force=False, **kwargs):
        # layer identifier
        self.id = "projection"
        # number of modes
        self.modes = modes
        # number of variables on SU(modes)
        self.vars = (self.modes**2 - self.modes)//2
        # number of photons
        self.photons = photons
        # projection operator description
        self.pdesc = pdesc
        # emergent dimension
        self.input_dim = self.output_dim = comb(
            self.modes+self.photons-1, self.photons, exact=True)
        # catch extreme cases
        if self.input_dim > 100 and not force:
            raise ValueError(
                "System dimension is large ({}), decrease system dimension or set force flag to True".format(self.input_dim))

        super(ProjectionLayer, self).__init__(**kwargs)


    def build(self, input_shape):
        """
        Construct projection operator for layer - non-trainable.
        Creates the variables of the layer (optional, for subclass implementers).

        This is a method that implements of subclasses of `Layer` or `Model`
        can override if they need a state-creation step in-between
        layer instantiation and layer call.

        This is typically used to create the weights of `Layer` subclasses.

        Arguments:
          input_shape: Instance of `TensorShape`, or list of instances of
            `TensorShape` if the layer expects a list of inputs
            (one instance per input).

        """

        # returns an m x dim x dim array with m being the number of projectors to apply
        self.proj = povm_gen(self.pdesc, convert=True)


    def call(self, inputs):
        """This is where the layer's logic lives.
        Arguments:
            inputs: Input tensor, or list/tuple of input tensors.
            **kwargs: Additional keyword arguments.

        Returns:
            A tensor or list/tuple of tensors.
        """

        # compute effect of  measurement projector on modes specified by photonic projectors 
        left = tf.einsum('ijk,bkl->ibjl', self.proj, inputs, name='Projection_Left')
        # can skip adjoint calculation since projectors are all real diagonals (does that seem right?)
        right = tf.einsum('ibjl,ilk->ibjk', left, self.proj)
        # collapse projector outcome sum into single, taking of batch broadcasting rules
        out = tf.reduce_sum(right, axis=[0], name='Projector_sum')

        return out


    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(ProjectionLayer, self).get_config()
        base_config['output_dim'] = self.output_dim
        base_config['modes'] = self.modes
        base_config['vars'] = self.vars
        base_config['photons'] = self.photons 
        base_config['pdesc'] = self.pdesc 
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @property
    def output_size(self):
        return [self.output_dim, self.output_dim]


class NoiseLayer(Layer):
    """
    Simple Layer that applies a variety of noisy operations on a quantum optical network.
    """

    def __init__(self, dim, noisedesc, force=False, **kwargs):
        # layer identifier
        self.id = "noise"
        # dimension of operator
        self.dim = dim
        # number of variables on SU(modes)
        self.vars = (self.dim**2 - self.dim)//2
        # non-linear operator description
        self.noisedesc = noisedesc

        # add in default values 
        if "systematic" not in self.noisedesc:
            self.noisedesc["systematic"] = False

        if "pad" not in self.noisedesc:
            self.noisedesc["pad"] = 0

        # catch extreme cases
        if self.dim > 100 and not force:
            raise ValueError(
                "System dimension is large ({}), decrease system dimension or set force flag to True".format(self.dim))

        # follow layer naming conventions
        self.input_dim = self.output_dim = self.dim

        # keep local copy of beam splitter decomposition table
        self.bms_spec = clements_phase_end(np.eye(self.dim))[0]

        super(NoiseLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Construct non-linear unitary to apply.
        Creates the variables of the layer (optional, for subclass implementers).

        This is a method that implements of subclasses of `Layer` or `Model`
        can override if they need a state-creation step in-between
        layer instantiation and layer call.

        This is typically used to create the weights of `Layer` subclasses.

        Arguments:
          input_shape: Instance of `TensorShape`, or list of instances of
            `TensorShape` if the layer expects a list of inputs
            (one instance per input).

        """


        # define a systematic error unitary
        if self.noisedesc["systematic"]:
            # set random seed if supplied
            if "seed" in self.noisedesc:
                np.random.seed(self.noisedesc['seed'])

            # generate a random unitary and take a fractional power of it
            Unoise = scipy.linalg.fractional_matrix_power(randU(self.dim), self.noisedesc['s_noise'])
            
            # convert to tensorflow compatible object
            self.systematic = tf.convert_to_tensor(Unoise, dtype=tf.complex64)


        # call build method of super class
        super(NoiseLayer, self).build(input_shape)

    def call(self, inputs):
        """This is where the layer's logic lives.
        Arguments:
            inputs: Input tensor, or list/tuple of input tensors.
            **kwargs: Additional keyword arguments.

        Returns:
            A tensor or list/tuple of tensors.
        """

        md = self.noisedesc["w_noise"]

        # superclass method for variable construction, get_variable has trouble with model awareness
        self.diag = backend.random_uniform(shape=[self.dim], dtype=tf.float32, minval=-np.pi*md, maxval=np.pi*md)

        self.theta = backend.random_uniform(shape=[self.vars], dtype=tf.float32, minval=0, maxval=np.pi*md/2)

        self.phi = backend.random_uniform(shape=[self.vars], dtype=tf.float32, minval=0, maxval=2*np.pi*md)

        # construct noisy operation on required number of modes
        self.unitary = tf_clements_stitch(self.bms_spec, self.theta, self.phi, self.diag)

        # apply systematic error
        if self.noisedesc["systematic"]:
            self.unitary = tf.matmul(self.unitary, self.systematic)

        # pad noise unitary if required
        if self.noisedesc["pad"] > 0:
            # pad operator dimension as many times as requested
            pad_op = [tf.constant([[1.0]], dtype=tf.complex64)]*self.noisedesc["pad"]
            pad_op = [self.unitary] + pad_op

            # perform block diagonal operator
            linop_blocks = [tf.linalg.LinearOperatorFullMatrix(block) for block in pad_op]
            # but y tho
            self.unitary = tf.convert_to_tensor(tf.linalg.LinearOperatorBlockDiag(linop_blocks).to_dense())

        # perform matrix calculation using multiple einsums on input/output spaces
        leftm = tf.einsum('ij,bjl->bil', self.unitary,
                          inputs, name="Einsum_left")
        output = tf.einsum('bil,lj->bij', leftm,
                           tf.linalg.adjoint(self.unitary), name="Einsum_right")

        # # return probabalistic output
        # out = tf.math.scalar_mul(1-self.u_prob, inputs) + \
        #     tf.math.scalar_mul(self.u_prob, rightm)

        return output

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(NoiseLayer, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @property
    def output_size(self):
        return [self.output_dim, self.output_dim]



class NonLinearLayer(Layer):
    """
    Simple Layer that implements a variety of nonlinear behaviours on a quantum optical network. 
    """

    def __init__(self, modes, photons, nldesc, force=False, **kwargs):
        # layer identifier
        self.id = "nonlinear"
        # number of modes
        self.modes = modes
        # number of variables on SU(modes)
        self.vars = (self.modes**2 - self.modes)//2
        # number of photons
        self.photons = photons
        # non-linear operator description
        self.nldesc = nldesc

        # emergent dimension
        self.input_dim = self.output_dim = comb(
            self.modes+self.photons-1, self.photons, exact=True)
        # catch extreme cases
        if self.input_dim > 100 and not force:
            raise ValueError(
                "System dimension is large ({}), decrease system dimension or set force flag to True".format(self.input_dim))

        super(NonLinearLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Construct non-linear unitary to apply.
        Creates the variables of the layer (optional, for subclass implementers).

        This is a method that implements of subclasses of `Layer` or `Model`
        can override if they need a state-creation step in-between
        layer instantiation and layer call.

        Arguments:
          input_shape: Instance of `TensorShape`, or list of instances of
            `TensorShape` if the layer expects a list of inputs
            (one instance per input).

        """
        # construct non-linear operator from dictionary description
        self.nltype, self.u_prob, self.unitary = nl_gen(self.nldesc, convert=True)
    
        # construct multiphoton unitary using either CPU intensive (slow) or memory hungry method
        if self.photons > 1 and np.shape(self.unitary)[0] == self.modes:
            # use symmetric map to compute multi photon unitary
            S = tf.constant(symmetric_map(
                self.modes, self.photons), dtype=tf.complex64)
            # map to product state then use symmetric isometry to reduce to isomorphic subspace
            self.unitary = tf.matmul(S, tf.matmul(tf_multikron(
                self.unitary, self.photons), tf.linalg.adjoint(S)))

        # call build method of super class
        super(NonLinearLayer, self).build(input_shape)

    def call(self, inputs):
        """This is where the layer's logic lives.
        Arguments:
            inputs: Input tensor, or list/tuple of input tensors.
            **kwargs: Additional keyword arguments.

        Returns:
            A tensor or list/tuple of tensors.
        """

        # perform matrix calculation using multiple einsums
        leftm = tf.einsum('ij,bjl->bil', self.unitary,
                          inputs, name="Einsum_left")
        rightm = tf.einsum('bil,lj->bij', leftm,
                           tf.linalg.adjoint(self.unitary), name="Einsum_right")

        # return probabalistic output
        out = tf.math.scalar_mul(1-self.u_prob, inputs) + \
            tf.math.scalar_mul(self.u_prob, rightm)

        return out

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(NonLinearLayer, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @property
    def output_size(self):
        return [self.output_dim, self.output_dim]


class InvertLayer(Layer):
    """
    Advanced layer that encapsulates a fully connected dense layer while retaining invertibility.
    This layer is composed of bijective functions and thus is itself bijective! 
    """

    def __init__(self, dim, pad=0, init='glorot_uniform', permutation=True, force=False, **kwargs):
        # layer identifier
        self.id = "affine"
        # number of variables on SU(modes)
        self.var_num = dim
        # pad dimensions
        self.pad = pad
        # kernel initialiser
        self.kernel_init = init
        # whether to apply a fixed permutation matrix to input
        self.perm_flag = permutation
        # input dimension and output dimension match
        self.input_dim = self.output_dim = self.var_num + self.pad
        # split dimension
        self.split_dim = self.input_dim//2
        # layer direction flag
        self.invert = False
        # inheritance superclass
        super(InvertLayer, self).__init__(**kwargs)


    def build(self, input_shape):
        """
        Construct projection operator for layer - non-trainable.
        Creates the variables of the layer (optional, for subclass implementers).

        This is a method that implements of subclasses of `Layer` or `Model`
        can override if they need a state-creation step in-between
        layer instantiation and layer call.0

        This is typically used to create the weights of `Layer` subclasses.

        Arguments:
          input_shape: Instance of `TensorShape`, or list of instances of
            `TensorShape` if the layer expects a list of inputs
            (one instance per input).

        """

        # build a randomly generated fixed permutation if requested
        if self.perm_flag:
            pass
            #self.permutation = tf.convert_to_tensor()

        # build the internal models 

        # call build method of super class
        super(InvertLayer, self).build(input_shape)

    def layer_apply(self, input_vec, kernel):
        """
        Internal method for applying a layer to an input vector
        """
        return tf.einsum('ij,bj->bi', kernel, input_vec, name="einsum_dense")
         


    def call(self, inputs):
        """This is where the layer's logic lives.
        Arguments:
            inputs: Input tensor, or list/tuple of input tensors.
            **kwargs: Additional keyword arguments.

        Returns:
            A tensor or list/tuple of tensors.
        """

        # check if model is set to be inverted
        if self.invert:
            pass
        else:
            # split input into two evenly divided vectors
            u1, u2 = tf.split(inputs, num_or_size_splits=2, axis=1, name='split')

            # compute output vectors using bijective map 
            s2 = self.layer_apply(u2, self.skernel2)
            t2 = self.layer_apply(u2, self.tkernel2)
            v1 = tf.math.multiply(u1, tf.math.exp(s2)) + t2 

            s1 = self.layer_apply(v1, self.skernel1)
            t1 = self.layer_apply(v1, self.tkernel1) 
            v2 = tf.math.multiply(u2, tf.math.exp(s1)) + t1 

            # # compute effect of  measurement projector on modes specified by photonic projectors 
            # left = tf.einsum('ijk,bkl->ibjl', self.proj, inputs, name='Projection_Left')
            # # can skip adjoint calculation since projectors are all real diagonals (does that seem right?)
            # right = tf.einsum('ibjl,ilk->ibjk', left, self.proj)
            # # collapse projector outcome sum into single, taking of batch broadcasting rules
            # out = tf.reduce_sum(right, axis=[0], name='Projector_sum')

            # concatenate output vectors and return
            return tf.concat([v1, v2], axis=-1)
            #return self.layer_apply(inputs, self.skernel)


    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(InvertLayer, self).get_config()
        base_config['output_dim'] = self.output_dim
        base_config['vars'] = self.vars
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @property
    def output_size(self):
        return [self.output_dim, self.output_dim]

class RevDense(Layer):
    """Just your regular densely-connected NN layer.

    `Dense` implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).

    Note: if the input to the layer has a rank greater than 2, then
    it is flattened prior to the initial dot product with `kernel`.

    # Example

    ```python
        # as first layer in a sequential model:
        model = Sequential()
        model.add(Dense(32, input_shape=(16,)))
        # now the model will take as input arrays of shape (*, 16)
        # and output arrays of shape (*, 32)

        # after the first layer, you don't need to specify
        # the size of the input anymore:
        model.add(Dense(32))
    ```

    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).

    # Input shape
        nD tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.

    # Output shape
        nD tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """

    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(RevDense, self).__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.supports_masking = True
        self.invert = False
        # call init method of super class
        super(RevDense, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel')
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias')
        else:
            self.bias = None
        self.built = True

        super(RevDense, self).build(input_shape)

    def call(self, inputs):

        if self.invert:
            if self.use_bias:
                output = inputs + self.bias
                output = tf.einsum('ij,bj->bi', self.kernel, output, name="einsum_dense")
            else:
                output = tf.einsum('ij,bj->bi', self.kernel, inputs, name="einsum_dense")
        else:
            output = tf.einsum('ij,bj->bi', self.kernel, inputs, name="einsum_dense")
            if self.use_bias:
                output = output + self.bias

        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def invert_layer(self):
        """
        Invert the network
        """

        # compute the inverse then reassign
        self.kernel = tf.linalg.inv(self.kernel)

        self.bias = tf.multiply(-1.0, self.bias)

        # flip invert flag
        self.invert = not self.invert

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
        }
        base_config = super(RevDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))