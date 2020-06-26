from datasets.data_utils import * #_dequantize
from datasets.mnist import mnist_loader
from datasets.cifar import CIFAR10DataSource

import jax.numpy as jnp

def load_by_name(name):
    exec('from datasets.{} import {}'.format(name.lower(), name))
    return locals()[name]

def load_dataset(dataset,
                 batch_size=1000,
                 seed=0,
                 toy_name="sine",
                 toy_train_size=5000,
                 toy_val_size=5000,
                 toy_test_size=5000):
    if dataset == "CIFAR":
        data_source = CIFAR10DataSource(
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            shuffle_seed=seed,
            preprocessing=False)
        train_ds = data_source.train_ds
        test_ds = data_source.eval_ds

        VAL_IMAGES = 5000
        train_data = tf_to_jnp(train_ds, data_source.TRAIN_IMAGES // batch_size)
        val_data = train_data[0:VAL_IMAGES]
        train_data = train_data[0:len(train_data) - VAL_IMAGES]
        train_data = flip_lr_augment(train_data)
        test_data = tf_to_jnp(test_ds, data_source.EVAL_IMAGES // batch_size)

        train_data = dequantize(train_data / 255)
        val_data = dequantize(val_data / 255)
        test_data = dequantize(test_data / 255)

        train_data = logit_transform(train_data, alpha=0.05)
        val_data =logit_transform(val_data, alpha=0.05)
        test_data =logit_transform(test_data, alpha=0.05)

        return train_data, val_data, test_data, cifar_transform

    elif dataset == "MNIST":

        VAL_IMAGES = 10000

        train_data, train_labels, test_data, test_labels = mnist_loader()
        val_data = train_data[-VAL_IMAGES:]
        train_data = train_data[:-VAL_IMAGES]

        train_data = dequantize(train_data)
        val_data = dequantize(val_data)
        test_data = dequantize(test_data)

        train_data = logit_transform(train_data)
        val_data =logit_transform(val_data)
        test_data =logit_transform(test_data)

        return train_data, val_data, test_data, mnist_transform
    elif dataset in ['GAS', 'POWER', 'HEPMASS', 'MINIBOONE', 'BSDS300']:  # use the constructors by MAF authors
        ds = load_by_name(dataset)("storage/")
        train_data = jnp.array(ds.trn.x)
        val_data = jnp.array(ds.val.x)
        test_data = jnp.array(ds.tst.x)

        return train_data, val_data, test_data, lambda x: x
    elif dataset in ['DENSITY', 'TOY']:  # use own constructors
        dataset = "DENSITY"
        train_data = load_by_name(dataset)(toy_name, toy_train_size)
        val_data = load_by_name(dataset)(toy_name, toy_val_size)
        test_data = load_by_name(dataset)(toy_name, toy_test_size)

        train_data = jnp.array(train_data.dataset)
        val_data = jnp.array(val_data.dataset)
        test_data = jnp.array(test_data.dataset)

        return train_data, val_data, test_data, lambda x: x
    else:
        raise ValueError(f"Wrong dataset: {dataset}")
