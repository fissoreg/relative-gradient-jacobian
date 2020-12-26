import jax.numpy as jnp

from jax   import vmap, grad, jit
from utils import *

from jax.scipy.stats import laplace, norm #, logistic
from itertools       import chain

def inner_layerwise(act_prime, y):
    return jnp.log(vmap(vmap(act_prime))(y))

def loss_layerwise(nonlinearity, ys):
    sigma_prime = grad(nonlinearity)
    batched = vmap(inner_layerwise, in_axes=(None, 0))

    # summing individual layers contributions
    # Note: this works fine even with `len(ys)` == 2
    full_pass = jnp.sum(batched(sigma_prime, jnp.stack(ys)), axis=0)

    # summing over dimension
    return jnp.sum(full_pass, axis=1)

# Function to compute the term L^2_J of the loglikelihood
def log_abs_det(params):
    Ws = [W for (W, b) in params]
    return jnp.sum(jnp.linalg.slogdet(Ws)[1])

@jit
@vmap
def log_pdf_normal(s):
    """ Log-pdf for a Gaussian distribution w. mean 0 std 1"""
    return jnp.sum(norm.logpdf(s))

# Note that here we compute the 3 terms of the loglikelihood;
# during training we directly optimize only the term `l1 + l2`
# and we include the gradient of the term `l3` explicitly
# (i.e. the `loss` function we derive includes only `l1 + l2`
# and the `l3` term is introduced with `add_det_grad`)
def get_piecewise_loss(g_layerwise, loss_pdf, nonlinearity):

    @jit
    def piecewise_loss(params, x):
        z, ys = g_layerwise(params, x)

        l1 = jnp.mean(log_pdf_normal(z))
        l2 = jnp.mean(loss_layerwise(nonlinearity, ys))
        l3 = log_abs_det(params)

        return l1, l2, l3

    return piecewise_loss
