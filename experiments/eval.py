import jax.numpy as jnp

from model                import get_model
from losses               import *
from custom_activations   import get_activation_fn
from datasets.dataloaders import load_dataset
from args                 import get_args

args = get_args()

nonlinearity = get_activation_fn(args.nonlinearity, args.alpha)

# Load dataset
train_data, val_data, test_data, _ = load_dataset(args.dataset,
                                                  toy_name=args.toy_name)

# we stopped with a look-ahead so the first set of parameters is the best
# (the loss hasn't improved (but might have gotten worse) in the subsequent
# epochs)
params = jnp.load(f'{args.log_dir}params_0.npy', allow_pickle=True)
params = [(p[0], p[1]) for p in params]

_, _, g_layerwise, _ = get_model(params[0][0].shape[0], args.num_layers,
                                 nonlinearity=nonlinearity)

# Note: we redefine `piecewise_loss` to get rid of the mean
@jit
def piecewise_loss(params, x):

    z, ys = g_layerwise(params, x)

    l1 = log_pdf_normal(z)
    l2 = loss_layerwise(nonlinearity, ys)
    l3 = log_abs_det(params)

    return l1, l2, l3

full_loss = lambda params, x: sum(piecewise_loss(params, x))
ll = full_loss(params, test_data)

print(f"Test set evaluation: \
        {jnp.mean(ll)} +- {2 * jnp.std(ll) / jnp.sqrt(ll.shape[0])}")
