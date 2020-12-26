import operator
import jax.numpy as jnp
import numpy.random as rnd

from scipy.stats import ortho_group
from scipy.stats import norm as sci_norm
from scipy.stats import gamma as sci_gamma
from jax         import grad, vmap, jit

from matplotlib import pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from functools            import reduce

def flatten_shallow(l):
    return reduce(operator.concat, l)

def interleave(*lists):
    return flatten_shallow(zip(*lists))

def partitions(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

# functions returning sublists with odd/even indices (no copies made)
def odd(l):
    return l[1::2]

def even(l):
    return l[::2]

def batchify(data, batch_size):
    return partitions(data, batch_size)

def sample(data, n):
    idxs = rnd.choice(n, size=(n,), replace=False)
    return data[idxs], idxs

def shuffle_perm(data):
    n = data.shape[0]
    return sample(data, n)

def square_ortho_mat(d, **kwargs):
  m = ortho_group.rvs(d, **kwargs)
  return jnp.dot(m, m.T)

def ortho_params(params, **kwargs):
  return [square_ortho_mat(p.shape[-1], **kwargs) for p in params]

def pt_to_jnp(data):
    return jnp.array(data).reshape(data.shape[0], -1)

def newton(f):

    f_prime = grad(f)

    def iter(x):
        return x - f(x) / f_prime(x)

    return iter

def inverse(f):

    def inv(y, x0 = 0.0, n_iter = 100):

        def fp(x):
            return f(x) - y

        iter = newton(fp)

        return reduce(lambda x0, x1: iter(x0), range(n_iter), x0)

    return inv

def get_f(params, act_inv, bias=False):

    biases = [0 for p in params]

    if bias:
        augemented_params = params
        biases = [p[:-1, -1] for p in augemented_params]
        params = [p[:-1, :-1] for p in params]

    inv_params = [jnp.linalg.inv(p) for p in params][::-1]

    @jit
    def f(s):
        x = (s - biases[0]) @ inv_params[0]
        return reduce(lambda x, Wb: (act_inv(x) - Wb[1]) @ Wb[0], zip(inv_params[1:], biases[1:]), x)

    return f

def get_g(params, activation):

    @jit
    def g(x):
        x = x @ params[0]
        return reduce(lambda x, W: activation(x) @ W, params[1:], x)

    return g

def move_and_scale(x):

    minimum = jnp.min(x)
    maximum = jnp.max(x)

    delta = maximum - minimum

    x += jnp.abs(minimum)
    x /= delta

    return x

def generate(params, activation, sampler, log_dir,
             transform=lambda x: x, bias=False):

    act_inv = inverse(activation)
    act_inv = jit(vmap(vmap(act_inv)))
    f = get_f(params, act_inv, bias=bias)

    s = sampler()

    gen = f(s)

    gen = transform(gen)

    img = save_image(gen, f"{log_dir}gen.png", nrow = 10)

    return gen, img

def img_to_fig(img, **kwargs):
    f = plt.figure(0)
    plt.axis('off')
    plt.imshow(img, **kwargs)

    return f

def sample_gaussian(n_samples, u=[(0, 1)]):
    return jnp.vstack([sci_norm.rvs(*params, n_samples) for params in u]).T

def get_gaussian_mixture(n_samples, uss=[[(1, 0)]]):
    return jnp.concatenate([sample_gaussian(n_samples, us) for us in uss])

def mixture_pdf(data, uss, base_pdf):
    pdfss = [get_nD_pdf(data, us, base_pdf) for us in uss]
    return jnp.mean(jnp.array(pdfss), axis=0)

def get_nD_pdf(data, us, pdf):
    pdfs = [pdf(data[:, i], *us[i]) for i in range(data.shape[1])]
    pdfs = reduce(operator.mul, pdfs, 1)
    return pdfs

def plot_2D_pdf(data, pdfs, ax=None):

    fig = None
    if ax is None:
        fig = plt.figure()
        ax = Axes3D(fig)
    ax.scatter(data[:, 0], data[:, 1], pdfs)

    return fig, ax

# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
    This code is created with reference to torchvision/utils.py.
    Modify: torch.tensor -> jax.numpy.DeviceArray
    If you want to know about this file in detail, please visit the original code:
        https://github.com/pytorch/vision/blob/master/torchvision/utils.py
"""
import math
from PIL import Image

import jax
import jax.numpy as jnp


def save_image(ndarray, fp, nrow=8, padding=2, pad_value=0.0, format=None):
    """Make a grid of images and Save it into an image file.
  Args:
    ndarray (array_like): 4D mini-batch images of shape (B x H x W x C)
    fp - A filename(string) or file object
    nrow (int, optional): Number of images displayed in each row of the grid.
      The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
    padding (int, optional): amount of padding. Default: ``2``.
    scale_each (bool, optional): If ``True``, scale each image in the batch of
      images separately rather than the (min, max) over all images. Default: ``False``.
    pad_value (float, optional): Value for the padded pixels. Default: ``0``.
    format(Optional):  If omitted, the format to use is determined from the filename extension.
      If a file object was used instead of a filename, this parameter should always be used.
  """
    if not (isinstance(ndarray, jnp.ndarray) or
        (isinstance(ndarray, list) and all(isinstance(t, jnp.ndarray) for t in ndarray))):
        raise TypeError('array_like of tensors expected, got {}'.format(type(ndarray)))

    ndarray = jnp.asarray(ndarray)

    if ndarray.ndim == 4 and ndarray.shape[-1] == 1:  # single-channel images
        ndarray = jnp.concatenate((ndarray, ndarray, ndarray), -1)

    # make the mini-batch of images into a grid
    nmaps = ndarray.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(ndarray.shape[1] + padding), int(ndarray.shape[2] + padding)
    num_channels = ndarray.shape[3]
    grid = jnp.full((height * ymaps + padding, width * xmaps + padding, num_channels), pad_value).astype(jnp.float32)
    k = 0
    for y in range(ymaps):
      for x in range(xmaps):
        if k >= nmaps:
          break
        grid = jax.ops.index_update(
          grid, jax.ops.index[y * height + padding:(y + 1) * height,
                              x * width + padding:(x + 1) * width],
          ndarray[k])
        k = k + 1

    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = jnp.clip(grid * 255.0 + 0.5, 0, 255).astype(jnp.uint8)
    im = Image.fromarray(ndarr.copy())
    im.save(fp, format=format)

    return im
