import jax.numpy as jnp
import models

from losses import loss_layerwise, loss_lad, log_pdf_normal
from custom_activations import smooth_leaky_relu
from datasets.dataloaders import load_dataset
from jax import jit
from args import get_args

opt = get_args()
# TODO: include the following args in `opt`
alpha_inf = 0.01
activation = lambda x: smooth_leaky_relu(x, alpha = alpha_inf)
loss_pdf = log_pdf_normal
#print(opt)

# Load dataset
train_data, val_data, test_data, transform = load_dataset(opt.dataset,
                                                          toy_name=opt.toy_name)

params = jnp.load(f'{opt.log_dir}params_0.npy')
d = params[0].shape[0]
_, _, g_layerwise, _ = models.g_with_dummies(d, len(params),
                                             activation = activation,
                                             bias=opt.bias)

def piecewise_loss_(loss_pdf, activation, g_layerwise):

    @jit
    def inner(params, data):
        o, ys = g_layerwise(params, data)

        l1 = loss_pdf(o)
        l2 = loss_layerwise(activation, ys)
        l3 = loss_lad(params)

        return l1, l2, l3

    return inner

log_test = piecewise_loss_(loss_pdf, activation, g_layerwise)
r = log_test(params, test_data)
rf = sum(r)

print(f"Test set evaluation: {jnp.mean(rf)} +- {2 * jnp.std(rf) / jnp.sqrt(rf.shape[0])}")
