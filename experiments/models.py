import jax.numpy as np
from jax import random
from jax.experimental.stax import elementwise, serial, Dense, Softmax
from jax.nn import sigmoid
from jax.nn.initializers import glorot_normal, kaiming_normal, orthogonal

#from utils import flatten_shallow, interleave
from utils import *
Abs = elementwise(np.abs)


def serial_with_outputs(*layers, mask):
    """Combinator for composing layers in serial.
    Args:
      *layers: a sequence of layers, each an (init_fun, apply_fun) pair.
    Returns:
      A new layer, meaning an (init_fun, apply_fun) pair, representing the serial
      composition of the given sequence of layers.
    """
    nlayers = len(layers)
    init_funs, apply_funs = zip(*layers)

    def init_fun(rng, input_shape):
        params = []
        for init_fun in init_funs:
            rng, layer_rng = random.split(rng)
            input_shape, param = init_fun(layer_rng, input_shape)
            params.append(param)
        return input_shape, params

    def apply_fun(params, inputs, **kwargs):
        outputs = []
        rng = kwargs.pop('rng', None)
        rngs = random.split(rng, nlayers) if rng is not None else (None,) * nlayers
        for i, (fun, param, rng) in enumerate(zip(apply_funs, params, rngs)):
            inputs = fun(param, inputs, rng=rng, **kwargs)
            if mask[i]:
                outputs.append(inputs)
        # Naming slightly misleading:
        # - "inputs" corresponds to the output of the nn (o)
        # - "outputs" corresponds to activations+o
        return inputs, outputs

    return init_fun, apply_fun


#def Linear(out_dim, W_init=glorot_normal()):
def Linear(out_dim, W_init=orthogonal()):
    """Layer constructor function for a linear layer without bias (i.e. a matrix)."""

    def init_fun(rng, input_shape):
        output_shape = input_shape[:-1] + (out_dim,)
        W = W_init(rng, (input_shape[-1], out_dim))
        return output_shape, W

    def apply_fun(params, inputs, **kwargs):
        return np.dot(inputs, params)

    return init_fun, apply_fun


def Accumulator():
    # TODO: do we need `rng` for compatibility (with `serial`)? Check!
    def init_fun(rng, input_shape):
        a = np.zeros(input_shape)
        return input_shape, a

    # def apply_fun(a, inputs, **kwargs):
    #  return inputs + a
    # Dirty fix below!
    def apply_fun(a, inputs, **kwargs):
        if a.shape == inputs.shape:
            return inputs + a
        else:
            return inputs

    return init_fun, apply_fun


Accumulator = Accumulator()

# Functions to deal with biases

def act_but_last(activation):
    '''Nonlinearity for affine layers'''
    def apply_but_last(x):
        y_ = activation(x[:, :-1])
        return np.hstack([y_, x[:, -1:]])
    return apply_but_last

""" TODO: generalize `activation` field."""

def G(input_shape, num_layers, activation=sigmoid, bias=False, **kwargs):
    if bias:
        activation = act_but_last(activation)
    activation = elementwise(activation)
    block = [Linear(input_shape, **kwargs), Accumulator, activation]
    layers = flatten_shallow(block for i in range(num_layers))[:-1]
    mask = flatten_shallow([False, True, False] for i in range(num_layers))[:-1]
    init_fun, apply_fun = serial_with_outputs(*layers, mask=mask)

    def init(rng, input_shape):
        output_shape, params = init_fun(rng, input_shape)
        opt_params = params[::3]
        const_params = params[1::3]
        if bias:
            opt_params = [affine_padder(p) for p in opt_params]
            const_params = [augment_dummy_params(p) for p in const_params]
        return params, opt_params, const_params  # even(params), odd(params)

    def g(opt_params, const_params, inputs):
        if bias:
            inputs = augment_data(inputs)
        o, ys = apply_fun(interleave(opt_params, const_params, (() for p in opt_params)), inputs, **kwargs)
        if bias:
            o = column_stripper(o)
            ys = [column_stripper(y) for y in ys]
        return o, ys

    return init, g


def MaxOut(filter):
    """MaxOut activation function: partitions the last dimension of the input in
    chunks of size `filter` and returns the maximum of each chunk"""

    def init_fun(rng, input_shape):
        # TODO: add an assert
        output_shape = input_shape[:-1] + (input_shape[-1] // filter,)
        return output_shape, ()

    def apply_fun(params, inputs, **kwargs):
        out_dim = inputs.shape[-1] // filter
        r = np.reshape(inputs, inputs.shape[:-1] + (out_dim, filter))
        return np.amax(r, axis=-1)

    return init_fun, apply_fun


def MaxOutBlock(d, filter):
    return serial(Dense(filter * d), MaxOut(filter))


def AbsBlock(d):
    return serial(Dense(d), Abs)


def MultinomialLogisticRegression(out_dim):
    return serial(Dense(out_dim), Softmax)


def get_dummy_params(params, data, bias=False):
    dummy = [np.zeros(data.shape) for p in params]

    if bias:
        dummy = [augment_dummy_params(p) for p in dummy]

    return dummy


def get_g_layerwise(g_dummy):
    return lambda params, data: g_dummy(params, get_dummy_params(params, data), data)


def g_with_dummies(d, num_layers, activation=sigmoid, batch_size=10, seed=1, bias=False):
    key = random.PRNGKey(seed)
    init, g_dummy = G(d, num_layers, activation=activation, bias=bias)
    _, params, dummy_params = init(key, (batch_size, d))
    g_layerwise = lambda params, data: g_dummy(params, dummy_params, data)
    g = lambda params, data: g_layerwise(params, data)[0]

    return params, g_dummy, g_layerwise, g


def incremental_forward(activations):
    def fwd(params, batch):
        o = batch

        def gen(W, activation):
            nonlocal o
            y = o @ W
            o = activation(y)
            return y

        ys = [gen(W, act) for (W, act) in zip(params, activations)]
        return o, ys

    return fwd


def g_plain(d, num_layers, activation=sigmoid, batch_size=10, seed=1):
    key = random.PRNGKey(seed)
    params = [random.normal(key, (d, d)) for i in range(num_layers)]
    activations = [activation for i in range(num_layers - 1)]
    activations.append(lambda x: x)

    # `g_layerwise` returns (o, ys) with `o` the final output (reconstructed sources) and
    # `ys` the activations at each layer
    g_layerwise = incremental_forward(activations)
    g = lambda params, data: g_layerwise(params, data)[0]

    return params, None, g_layerwise, g
