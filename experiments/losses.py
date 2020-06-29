from jax import vmap, grad, jacfwd, jit
from models import get_dummy_params, get_g_layerwise
from itertools import chain
from jax.scipy.stats import laplace, norm #, logistic
from utils import sample
import jax.numpy as jnp
from utils import *

def inner_layerwise(act_prime, y):
    return jnp.log(vmap(vmap(act_prime))(y))

def loss_layerwise(activation, ys):
    if len(ys) == 1:
        return 0
    else:
        act_prime = grad(activation)
        batched = vmap(inner_layerwise, in_axes=(None, 0))

        # summing individual layers contributions
        # Note: this works fine even with `len(ys)` == 2
        full_pass = jnp.sum(batched(act_prime, jnp.stack(ys[:-1])), axis=0)

        # summing over dimension
        return jnp.sum(full_pass, axis=1)

def lad(params):
    return jnp.linalg.slogdet(params)[1]
    #return np.log(np.abs(np.linalg.det(params)))

def ladj(g, data):
    jac = vmap(jacfwd(g))(data)
    return lad(jac)

def loss_lad(params):
    return jnp.sum(lad(params))

def natural_trick(gradient):

    def deltaW(W, grads):
        return W @ jnp.transpose(W) @ grads

    def natural_grad(params, data):
        grads = gradient(params, data)
        return [deltaW(*xs) for xs in zip(params, grads)]

    return natural_grad

def natural_grad(loss):
    return natural_trick(grad(loss))

def det_trick(gradient, bias=False, trick=True):

    def strip_pad(W):
        W = W.T
        W = column_stripper(W)
        col = jnp.zeros((W.shape[0], 1))
        return column_padder(W, col).T

    def det_trick_grad(params, data):
        deltaWs = gradient(params, data)
        dW = lambda bias, W: strip_pad(W) if bias else W
        apply_trick = lambda trick, W: W if trick else jnp.linalg.inv(W.T)
        return [deltaW - apply_trick(trick, dW(bias, W)) for (deltaW, W) in zip(deltaWs, params)]

    return det_trick_grad

def det_trick_grad(loss):
    return det_trick(natural_grad(loss))

#################################################################################

@vmap
def log_pdf_lapl(s):
    return jnp.sum(laplace.logpdf(s))

@vmap
def log_pdf_logistic(s):
    return jnp.sum(logistic.logpdf(s))

@vmap
def log_pdf_hypsec(s):
    '''
    Log-pdf for a **Hypersecant** distribution
    The term
    -np.log(2.0)
    Is just a normalization
    '''
    return jnp.sum(-2*jnp.log(jnp.cosh(s)) -jnp.log(2.0) )

@jit
@vmap
def log_pdf_normal(s):
    """ Log-pdf for a Gaussian distribution w. mean 0 std 1"""
    return jnp.sum(norm.logpdf(s))

def get_loss_deltas(g_dummy, loss_pdf, activation):
    def loss(params, dummy_params, data):
        o, ys = g_dummy(params, dummy_params, data)

        lpdf = loss_pdf(o)
        lwise = loss_layerwise(activation, ys)

        l = - sum(jnp.mean(li) for li in [lpdf, lwise])

        return l, (o, ys)
    return loss

def natural_trick_deltas(gradient, activation, bias=False, trick=True):

    def deltaW(W, o, delta):
        # Hack to cope with biases
        # To do: cleaner solution
        if o.shape[-1]!=W.shape[0]:
            o = augment_data(o)
        # Transposed Relative Gradient:
        if trick:
            dW = W @ (jnp.transpose(W) @ jnp.transpose(o)) @ delta
        else:
            dW = o.T @ delta

        return dW
        # Relative Gradient:
        #return jnp.transpose(o) @ ((delta @ jnp.transpose(W)) @ W)
        # no trick
        #return jnp.transpose(o) @ delta

    def natural_grad_deltas(params, data, bias=bias):

        dummy = get_dummy_params(params, data, bias=bias)
        ((grads, deltas), (o, ys)) = gradient(params, dummy, data)

        os = chain([data], (activation(y) for y in ys[:-1]))

        return [deltaW(*args) for args in zip(params, os, deltas)]

    return natural_grad_deltas

def natural_grad_deltas(loss, activation, bias=False, trick=True):
    gradient = grad(loss, argnums = (0, 1), has_aux = True)
    return det_trick(
            natural_trick_deltas(gradient, activation, bias=bias, trick=trick),
            bias=bias, trick=trick)

def piecewise_loss(loss_pdf, activation, g_layerwise):

    @jit
    def inner(params, data):
        o, ys = g_layerwise(params, data)

        l1 = jnp.mean(loss_pdf(o))
        l2 = jnp.mean(loss_layerwise(activation, ys))
        l3 = loss_lad(params)

        return l1, l2, l3

    return inner

def losses_logger(loss, loss_pdf, data, activation, g_layerwise,
                  n_samples = 100):
    def logger(params, epoch):
        data_batch, _ = sample(data, n_samples)

        losses = piecewise_loss(loss_pdf, activation, g_layerwise)

        l = normalize_loss(loss)(params, data_batch)
        l1, l2, l3 = losses(params, data_batch)

        return "Loss", [-x for x in [l1, l2, l3, l1 + l2 + l3, -l]]

    return logger

def normalize_loss(loss):
    def inner(params, data):
        try:
            return loss(params, data)
        except TypeError:
            return loss(params, get_dummy_params(params, data), data)[0]
    return inner

def evaluate_data(loss_pdf, data, activation, g_layerwise):

    def evaluator(params):
        losses = piecewise_loss(loss_pdf, activation, g_layerwise)
        loss = lambda params, data: sum(losses(params, data))
        return sum(loss(params, data_batch) for data_batch in data) / len(data)

    return evaluator

def get_loss_jac(g_dummy, loss_pdf, activation = None):

    g_layerwise = get_g_layerwise(g_dummy)

    def loss_jac(params, data):
        # features, labels = split_data(data)
        o, _ = g_layerwise(params, data)
        g = lambda data: g_layerwise(params, data)[0]

        lpdf = loss_pdf(o)#, labels)
        l = - lpdf - jnp.sum(ladj(g, data))

        return l

    return loss_jac
