import jax.numpy as jnp

from jax                  import random
from jax.nn.initializers  import orthogonal

from custom_activations   import smooth_leaky_relu


def Dense(params, x, dummy=0):
    W, b = params
    return jnp.dot(x, W) + dummy + b

def get_dummy_params(params, x):
    return [jnp.zeros(x.shape) for p in params]

def get_model(n_features, n_layers,
              nonlinearity=smooth_leaky_relu,
              key=random.PRNGKey(0),
              W_init=orthogonal()):

    # Forward pass through the network.
    # Returns the latent configurations `z`
    # and the activations at intermediate layers `ys`.
    def forward(params, dummies, x):

        z = x

        def step(Wb, dummy):
            nonlocal z
            y = Dense(Wb, z, dummy=dummy)
            z = nonlinearity(y)
            return y

        ys = [step(Wb, dummy) for (Wb, dummy) in zip(params[:-1], dummies[:-1])]

        # last layer (no nonlinearity)
        z = Dense(params[-1], z, dummies[-1])

        return z, ys

    g_dummy = forward

    def g_layerwise(params, x):
        dummies = get_dummy_params(params, x)
        return g_dummy(params, dummies, x)

    g = lambda params, x: g_layerwise(params, x)[0]

    # parameters init
    def init_Wb(key, n_features):
        return W_init(key, (n_features, n_features)), jnp.zeros((1, n_features))

    params = [init_Wb(k, n_features) for k in random.split(key, n_layers)]

    return params, g_dummy, g_layerwise, g
