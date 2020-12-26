import jax.numpy as jnp
import time

from jax.nn.initializers import orthogonal, glorot_normal
from jax.scipy.stats import norm
from jax.scipy.linalg import lu, tril, triu
from jax.experimental import optimizers as opt
from jax import random, vmap, jit, grad

from utils import * #shuffle_perm, batchify
from loggers import *

from itertools import chain
from functools import reduce

# command line args
from args import get_args
args = get_args()

print(args)

#from collections import namedtuple
#Args = namedtuple("Args", ["dataset", "lr", "batch_size", "log_dir", "seed", "num_layers", "epochs", "look_ahead"])
#args = Args("MNIST", 0.0001, 100, "log/", 42, 2, 100, 30)

from datasets.dataloaders import load_dataset
x, val_data, _, _ = load_dataset(args.dataset)

key = random.PRNGKey(args.seed)

def smooth_leaky_relu(x, alpha=0.01):
  r"""Smoothed version of the leaky relu activation function
  Inspiration:
  https://stats.stackexchange.com/questions/329776/approximating-leaky-relu-with-a-differentiable-function
  """
  return alpha*x + (1 - alpha)*(jnp.logaddexp(x, 0)) # - jnp.log(2.0))

nonlinearity = smooth_leaky_relu

def DensePLU(params, x, dummy=0):
    (P, L, U), b = params
    return x @ P @ L @ U + b + dummy

def PLU_orthogonal(key, shape):
    W = orthogonal()(key, shape)
    P, L, U = lu(W)
    #return P, jnp.abs(L), jnp.abs(U)
    return P, L, jnp.eye(shape[0]) + U

def PLU_eye(key, shape):
    W = jnp.eye(shape[0])
    return lu(W)

import jax
def PLU_init(key, shape, factor=1e-3):
    k1, k2 = random.split(key)
    i_s = random.choice(k1, shape[0], (shape[0],), False)
    j_s = random.choice(k2, shape[0], (shape[0],), False)
    P = jnp.zeros((shape[0], shape[0]))

    for (i, j) in zip(i_s, j_s):
        P = jax.ops.index_update(P, jax.ops.index[i, j], 1.0) #jax.ops.index[i:i+1, j:j+1], 1.0)

    I = jnp.eye(shape[0])
    U = I + triu(random.uniform(key, shape), k=1)*factor #, k=1)
    L = I + tril(random.uniform(key, shape), k=-1)*factor
    return P, L, U

def get_dummy_params(params, x):
  return [jnp.zeros(x.shape) for p in params]

def get_model(n_features, n_layers,
              coupling=DensePLU,
              nonlinearity=smooth_leaky_relu,
              key=random.PRNGKey(0),
              W_init=orthogonal()):

    def forward(params, dummies, x):

        z = x

        def step(params, dummy):
            nonlocal z
            y = coupling(params, z, dummy=dummy)
            z = nonlinearity(y)
            return y

        ys = [step(params, dummy) for (params, dummy)
                                  in zip(params[:-1], dummies[:-1])]

        # last layer (no nonlinearity)
        z = coupling(params[-1], z, dummies[-1])
        #ys.append(z)

        return z, ys

    g_dummy = forward

    def g_layerwise(params, x):
        dummies = get_dummy_params(params, x)
        return g_dummy(params, dummies, x)

    g = lambda params, x: g_layerwise(params, x)[0]

    # parameters init
    def init_Wb(key, n_features):
        return W_init(key, (n_features, n_features)), jnp.zeros(n_features)

    params = [init_Wb(k, n_features) for k in random.split(key, n_layers)]

    return params, g_dummy, g_layerwise, g

params, g_dummy, g_layerwise, g = get_model(x.shape[-1], args.num_layers,
                                    nonlinearity=nonlinearity,
                                    W_init=PLU_init, #PLU_orthogonal,
                                    key=key)

# STOP if we completed training already
import os
if os.path.exists(f"{args.log_dir}test_loss.npy"):
    raise FileExistsError("Training has been performed to the end already. Aborting.")

current, params = resume(args.log_dir, params)

print("Epoch", current)

@jit
@vmap
def log_pdf_normal(s):
    """ Log-pdf for a Gaussian distribution w. mean 0 std 1"""
    return jnp.sum(norm.logpdf(s))

loss_pdf = log_pdf_normal

def inner_layerwise(act_prime, y):
    return jnp.log(vmap(vmap(act_prime))(y))

def loss_layerwise(nonlinearity, ys):
    act_prime = grad(nonlinearity)
    batched = vmap(inner_layerwise, in_axes=(None, 0))

    # summing individual layers contributions
    # Note: this works fine even with `len(ys)` == 2
    full_pass = jnp.sum(batched(act_prime, jnp.stack(ys)), axis=0)

    # summing over dimension
    return jnp.sum(full_pass, axis=1)

def PLU_abs_jac_det(PLU):
    P, L, U = PLU
    return jnp.log(jnp.abs(jnp.prod(jnp.diag(U) * jnp.prod(jnp.diag(L)))))

@jit
def loss(params, x):

  z, ys = g_layerwise(params, x)

  lpdf = log_pdf_normal(z)
  lwise = loss_layerwise(nonlinearity, ys)
  ljac_plu = sum(PLU_abs_jac_det(PLU) for (PLU, _) in params)

  l = - sum(jnp.mean(li) for li in [lpdf, lwise]) - ljac_plu

  return l #, (z, ys)

def project_grad(gradient):

  def project(grads):
    (dP, dL, dU), db = grads
    return (jnp.zeros(dP.shape), tril(dL, k=-1), triu(dU)), 0 #db

  def projected_grad(params, x):
    grads_list = gradient(params, x)
    return [project(grads) for grads in grads_list]

  return projected_grad

gradient = grad(loss) #, has_aux = True)
gradient = project_grad(gradient)

print(loss(params, val_data[0:100]))
P, L, U = params[0][0]
jnp.prod(jnp.diag(U))
jnp.min(jnp.abs(jnp.diag(U)))
jnp.max(U)
jnp.max(L)
jnp.abs(U)
U

# logging
from tb_logger import tensorboard_decorator
tb_scalars, tb_histograms, tb_figures, writer = tensorboard_decorator(args.log_dir)

@tb_scalars
def log_loss(params, epoch):
  return "Loglikelihood", [loss(params, val_data[0:100])]

def save_to_disk(f, params):
    return jnp.save(f, params)

def get_stopper(look_ahead):

    values = []

    def stopper(params, epoch):

        nonlocal values

        _, ll = log_loss(params, epoch)
        d = {"e": epoch, "v": ll}
        save_to_disk(f'{args.log_dir}params_{(epoch - 1) % look_ahead}.npy', params)

        if len(values) == look_ahead and values[0]["v"] < ll:
            final_val, _ = log_loss(params, epoch)
            save_to_disk(f'{args.log_dir}test_loss.npy', [final_val, args.epochs])
            raise ValueError(f"STOP: loss has been increasing for {look_ahead} epochs.")

        if len(values) == 0 or values[0]["v"] > ll:
            values = [d]
        else:
            values.append(d)

        return "Best", values[0]["v"]

    return stopper

stopper = get_stopper(args.look_ahead)

def timer():
    start = time.perf_counter()

    @tb_scalars
    def log_time(params, epoch):
        nonlocal start
        now = time.perf_counter()
        delta = now - start
        start = now
        return "Time", [delta]

    return log_time

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

    opt_init, opt_update, get_params = opt.adam(lr)
    update = get_opt_update(get_params, opt_update, gradients)
    opt_state = opt_init(params)

    for i in range(start, epochs + 1):
        # TODO: shuffle in-place to reduce memory allocations (first, copy data)
        x, _ = shuffle_perm(x) if shuffle else (x, None)
        batches = batchify(x, batch_size)
        for batch in batches:
            opt_state = update(i, opt_state, batch)
            #params = get_params(opt_state)
            #logs = [log(params, i) for log in loggers]

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
      loggers = [log_loss, checkpoint(args.log_dir), timer(), stopper],
      start=current, log_every = 1
     )

gs = gradient(params, x)
z = g(params, x)
