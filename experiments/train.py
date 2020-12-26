import jax.numpy as jnp

from jax                  import random, vmap, jit, grad
from jax.nn.initializers  import orthogonal
from jax.scipy.stats      import norm
from jax.experimental     import optimizers as opt

from utils                import shuffle_perm, batchify
from custom_activations   import get_activation_fn, smooth_leaky_relu
from loggers              import *
from datasets.dataloaders import load_dataset
from args                 import get_args
from itertools            import chain

import sys
import os

# SETUP ########################################################################

# getting command line args
args = get_args()
print(args)

sys.path.append('experiments/')

# abort if we completed training already
if os.path.exists(f"{args.log_dir}test_loss.npy"):
    raise FileExistsError("Training has been performed to the end already.\
                           Aborting.")

# JAX random generator key
key = random.PRNGKey(args.seed)

x, val_data, _, _ = load_dataset(args.dataset)

nonlinearity = get_activation_fn(args.nonlinearity, args.alpha)

# MODEL DEFINITION #############################################################

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

params, g_dummy, g_layerwise, g = get_model(x.shape[-1], args.num_layers,
                                    nonlinearity=nonlinearity,
                                    key=key)

# eventually reload parameters from checkpoint
current, params = resume(args.log_dir, params)

# LOSS FUNCTIONS ###############################################################
@jit
@vmap
def log_pdf_normal(s):
    """ Log-pdf for a Gaussian distribution w. mean 0 std 1"""
    return jnp.sum(norm.logpdf(s))

def inner_layerwise(sigma_prime, y):
    return jnp.log(vmap(vmap(sigma_prime))(y))

def loss_layerwise(nonlinearity, ys):
    sigma_prime = grad(nonlinearity)
    batched = vmap(inner_layerwise, in_axes=(None, 0))

    # summing individual layers contributions
    # Note: this works fine even with `len(ys)` == 2
    full_pass = jnp.sum(batched(sigma_prime, jnp.stack(ys)), axis=0)

    # summing over dimension
    return jnp.sum(full_pass, axis=1)

# Here the `dummies` argument is needed to be able to compute the `delta` terms (Appendix D in [1])
# through the JAX `grad` function.
def loss(params, dummies, x):

    z, ys = g_dummy(params, dummies, x)

    lpdf = log_pdf_normal(z)
    lwise = loss_layerwise(nonlinearity, ys)

    l = - sum(jnp.mean(li) for li in [lpdf, lwise])

    return l, (z, ys)

# Function to compute the term L^2_J of the loglikelihood
def log_abs_det(params):
    Ws = [W for (W, b) in params]
    return jnp.sum(jnp.linalg.slogdet(Ws)[1])

# Note that here we want to log the full loglikelihood;
# during training we directly optimize only the term `l1 + l2`
# and we include the gradient of the term `l3` explicitly
# (i.e. the `loss` function we derive includes only `l1 + l2`
# and the `l3` term is introduced with `add_det_grad`)
@jit
def full_loss(params, x):
    z, ys = g_layerwise(params, x)

    l1 = jnp.mean(log_pdf_normal(z))
    l2 = jnp.mean(loss_layerwise(nonlinearity, ys))
    l3 = log_abs_det(params)

    return l1 + l2 + l3

# RELATIVE GRADIENT DEFINITION #################################################

def get_relative_gradient(gradient):

    def apply_trick(Wb, z, bp_terms):
        W, b = Wb
        db, delta = bp_terms

        dW = W @ (W.T @ z.T) @ delta + W @ b.T @ db
        db = db * (1 + b @ b.T) + b @ W.T @ z.T @ delta

        return (dW, db)

    def relative_gradient(params, x):

        dummies = get_dummy_params(params, x)
        ((grads, deltas), (z, ys)) = gradient(params, dummies, x)

        dbs = (db for (dW, db) in grads)
        bp_terms = zip(dbs, deltas) # backprop terms
        zs = chain([x], (nonlinearity(y) for y in ys))

        return [apply_trick(*args) for args in zip(params, zs, bp_terms)]

    return relative_gradient

def add_det_grad(gradient):

    def det_grad(params, x):
        grad_params = gradient(params, x)
        return [(dW - W, db) for ((dW, db), (W, b)) in zip(grad_params, params)]

    return det_grad

gradient = grad(loss, argnums = (0, 1), has_aux = True)
gradient = add_det_grad(get_relative_gradient(gradient))

# LOGGING ######################################################################

log_loss, get_loss_vs_time = get_loss_logger(full_loss, val_data)
stopper = get_stopper(log_loss, args)

# TRAINING LOOP ################################################################

def get_opt_update(get_params, opt_update, gradients):

    @jit
    def update(i, opt_state, batch):
        params = get_params(opt_state)
        grads = gradients(params, batch)
        return opt_update(i, grads, opt_state)

    return update

def float_log(l):
    try:
        return [float(x) for x in l]
    except:
        return float(l)

def train(params, x, gradients,
          epochs = 100, batch_size = 10, lr = 1e-3, shuffle = True,
          start=1,
          loggers = [], log_every = 10):

    # defining and initializing the optimizer
    opt_init, opt_update, get_params = opt.adam(lr)
    update = get_opt_update(get_params, opt_update, gradients)
    opt_state = opt_init(params)

    # training loop
    for i in range(start, epochs + 1):
        # TODO: shuffle in-place to reduce memory allocations (first, copy data)
        x, _ = shuffle_perm(x) if shuffle else (x, None)
        batches = batchify(x, batch_size)
        for batch in batches:
            opt_state = update(i, opt_state, batch)

        # console logging
        if i % log_every == 0:
            params = get_params(opt_state)
            logs = [log(params, i) for log in loggers]
            print(f"Epoch {i}", end=" ")
            for log in logs:
                print(f"[{log[0]} {float_log(log[1])}]", end=" ")
            print()

    return 0

train(params, x, gradient,
      epochs = args.epochs, lr = args.lr, batch_size = args.batch_size,
      loggers = [timer(), checkpoint(args.log_dir), stopper],
      start=current, log_every = args.log_every
     )
