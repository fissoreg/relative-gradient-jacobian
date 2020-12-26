from utils import *
from matplotlib import pyplot as plt
from jax import random

import time
import os
import jax.numpy as jnp
import numpy as onp

def timer():
    start = time.perf_counter()

    def log_time(params, epoch):
        nonlocal start
        now = time.perf_counter()
        delta = now - start
        start = now
        return "Time", [delta]

    return log_time

def checkpoint(log_dir):
    def log_checkpoint(params, epoch):
        jnp.save(f"{log_dir}checkpoint.npy", epoch)
        jnp.save(f"{log_dir}params.npy", params)

        return "Checkpoint", epoch
    return log_checkpoint

def resume(log_dir, params):
    try:
        epoch = jnp.load(f"{log_dir}checkpoint.npy")
        logged = jnp.load(f"{log_dir}params.npy")
        return int(epoch), [jnp.array(p) for (_, p) in enumerate(logged)]
    except:
        return 0, params

def params_getter(params):
    private = params
    def log_params(params, epoch):
        nonlocal private
        private = params
        return "Params mean", [onp.mean(p) for p in params]

    def get_params():
        return private

    return log_params, get_params

def get_loss_logger(loss, val_data):

    loss_vs_time = []

    def log_loss(params, epoch):
        val_loss = loss(params, val_data)
        loss_vs_time.append(val_loss)
        return "Loglikelihood", [val_loss]

    return log_loss, lambda: loss_vs_time

def save_to_disk(f, params):
    return jnp.save(f, params)

# stops training if the loss has not improved for `look_ahead` epochs
def get_stopper(log_loss, args):

    values = []

    def stopper(params, epoch):

        nonlocal values

        _, ll = log_loss(params, epoch)
        ll = -ll[0]

        if jnp.isnan(ll) or jnp.isinf(ll):
            raise ValueError("STOP: loss is inf/nan")

        d = {"e": epoch, "v": ll}
        save_to_disk(f'{args.log_dir}params_{(epoch - 1) % args.look_ahead}.npy', params)

        if len(values) == args.look_ahead and values[0]["v"] < ll:
            final_val, _ = log_loss(params, epoch)
            save_to_disk(f'{args.log_dir}test_loss.npy', [final_val, args.epochs])
            raise ValueError(f"STOP: loss has been increasing for {args.look_ahead} epochs.")

        if len(values) == 0 or values[0]["v"] > ll:
            values = [d]
        else:
            values.append(d)

        return "Best", values[0]["v"]

    return stopper

def get_log_descent(log_losses):

    precedent = 0

    def log_descent(params, epoch):
        nonlocal precedent

        _, values = log_losses(params, epoch)
        loss = values[3]
        diff = loss - precedent

        flag = 0 if loss < precedent else 1

        precedent = loss

        return "Descent", [flag, diff]

    return log_descent

def transform(x):
    return  x.reshape(len(x), 28, 28, 1)

#sampler_generation = lambda: jnp.concatenate([random.normal(k, (50, ns)), jnp.zeros((50, ns))])

def get_generator(activation, img_dir,
    every=10, seed=42, bias=False, transform=lambda x: x):
    try:
        os.mkdir(img_dir)
    except:
        pass

    k = random.PRNGKey(seed)
    sampler_generation = lambda n, ns: random.normal(k, (n, ns))

    old = plt.figure(0)

    #@tb_figures
    def log_generate(params, epoch):
        if epoch % every == 0: # and epoch > 0:
            ns = params[0].shape[0]
            if bias:
                ns = ns - 1
            sampler = lambda: sampler_generation(100, ns)
            _, img = generate(params, activation, sampler, f"{img_dir}{epoch}_",
                              transform = transform, bias=bias)
            f = img_to_fig(img)
            nonlocal old
            old = f
            return "Gen imgs", [f]
        return "Gen imgs", [old]
    return log_generate
