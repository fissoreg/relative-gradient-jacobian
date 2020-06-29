import numpy as onp
import jax.numpy as jnp
import models, losses

from losses import get_loss_deltas
from training import train
from custom_activations import smooth_leaky_relu
from tb_logger import tensorboard_decorator
from loggers import *
from utils import *
from datasets.cifar import CIFAR10DataSource
from datasets.dataloaders import load_dataset
from args import get_args

opt = get_args()
# TODO: include the following args in `opt`
alpha_inf = 0.01
activation = lambda x: smooth_leaky_relu(x, alpha = alpha_inf)
loss_pdf = losses.log_pdf_normal
print(opt)

# Load dataset
train_data, val_data, test_data, transform = load_dataset(opt.dataset)

# model definition ############################################################
d = train_data.shape[-1]
params, g_dummy, g_layerwise, g = models.g_with_dummies(d, opt.num_layers,
                                             activation = activation,
                                             seed = opt.seed,
                                             batch_size = opt.batch_size,
                                             bias=opt.bias)

current, params = resume(opt.log_dir, params)

loss = losses.get_loss_deltas(g_dummy, loss_pdf, activation = activation)
grad_deltas = lambda loss: losses.natural_grad_deltas(loss, activation,
                bias=opt.bias,
                trick=opt.trick)
gradient = grad_deltas(loss)

if opt.bias:
    gradient = gradient_padding(gradient)

# logging
tb_scalars, tb_histograms, tb_figures, writer = tensorboard_decorator(opt.log_dir)

log_losses_ = losses.losses_logger(loss, loss_pdf, val_data, activation, g_layerwise)
def log_loss(p, e):
    _, v = log_losses_(p, e)
    return "Loss", [v[3]]
log_losses = tb_scalars(log_loss)

log_generate = get_generator(activation, f'{opt.log_dir}imgs/',
    every=10, bias=opt.bias, transform=transform)
log_generate = tb_figures(log_generate)

log_time = tb_scalars(timer())

log_params, get_params = params_getter(params)

def save_to_disk(f, params):
    return jnp.save(f, params)

def get_stopper(look_ahead):

    values = []

    def stopper(params, epoch):

        nonlocal values

        _, lls = log_losses_(params, epoch)
        ll = lls[3]
        d = {"e": epoch, "v": ll}
        save_to_disk(f'{opt.log_dir}params_{(epoch - 1) % look_ahead}.npy', params)

        if len(values) == look_ahead and values[0]["v"] < ll:
            final_val, _ = log_losses(params, epoch)
            save_to_disk(f'{opt.log_dir}test_loss.npy', [final_val, opt.epochs])
            raise ValueError(f"STOP: loss has been increasing for {look_ahead} epochs.")

        if len(values) == 0 or values[0]["v"] > ll:
            values = [d]
        else:
            values.append(d)

        return "Best", values[0]["v"]

    return stopper

stopper = get_stopper(opt.look_ahead)
loggers_list = [log_losses, log_time, checkpoint(opt.log_dir), stopper]
if opt.generation:
    loggers_list.append(log_generate)

train(params, train_data, gradient,
      start = current+1,
      epochs = opt.epochs+1, lr = opt.lr, batch_size = opt.batch_size,
      loggers = loggers_list,
      log_every = 1
     )

trained_params = get_params()
