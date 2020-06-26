import numpy as onp

def tf_to_jnp(ds, n):
    it = iter(ds)
    convert = lambda x: jnp.array(x["image"]._numpy()).reshape(-1, 32*32*3)
    return jnp.concatenate([convert(next(it)) for i in range(n)])

def cifar_transform(x):
    return  x.reshape(len(x), 32, 32, 3)

def mnist_transform(x):
    return  x.reshape(len(x), 28, 28, 1)

def dequantize(x):
    """
    Adds noise to pixels to dequantize them.
    """
    x = x + onp.random.random_sample(x.shape) / 255
    maximum = onp.max(x)

    return x / maximum

def logit(x):
    """
    Elementwise logit (inverse logistic sigmoid).
    :param x: numpy array
    :return: numpy array
    """
    return onp.log(x / (1.0 - x))

def logit_transform(x, alpha=1e-6):
    """
    Transforms pixel values with logit to be unconstrained.
    """
    return logit(alpha + (1 - 2*alpha) * x)

def flip_horizontally(images):
    images = images.reshape(-1, 32, 32, 3)
    images = jnp.array([x[:, ::-1, :] for x in images])
    return images.reshape(-1, 32*32*3)

def flip_lr_augment(data):
    return jnp.vstack([data, flip_horizontally(data)])
