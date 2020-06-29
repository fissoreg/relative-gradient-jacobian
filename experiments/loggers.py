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
