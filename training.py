from jax import jit
from jax.experimental import optimizers as opt
from utils import shuffle_perm, batchify

#from matplotlib import pyplot as plt
#lr = 1e-3
##schedule = opt.inverse_time_decay(lr, 1, 0.005, staircase=True)
#schedule = opt.inverse_time_decay(lr, 1, 0.005, staircase=False)
##schedule = opt.exponential_decay(lr, 1, 0.9999)
#lrs = [schedule(i) for i in range(5000)]
#plt.yscale('log')
#plt.grid(True)
#plt.plot(lrs)
#plt.show()

def get_opt_update(get_params, opt_update, gradients):

    @jit
    def update(i, opt_state, data_batch):
        params = get_params(opt_state)
        grads = gradients(params, data_batch)
        return opt_update(i, grads, opt_state)

    return update

def float_log(l):
    try:
        return [float(x) for x in l]
    except:
        return float(l)

def train(params, data, gradients,
          epochs = 100, batch_size = 10, lr = 1e-3, shuffle = True, start = 0,
          lr_decay = 0, lr_decay_steps = 1,
          loggers = [], log_every = 10):

    if lr_decay:
        schedule = opt.inverse_time_decay(lr, lr_decay_steps, lr_decay, staircase=False)
    else:
        schedule = opt.constant(lr)

    opt_init, opt_update, get_params = opt.adam(schedule)
    update = get_opt_update(get_params, opt_update, gradients)
    opt_state = opt_init(params)

    for i in range(start, epochs):
        # TODO: shuffle in-place to reduce memory allocations (first, copy data)
        data, _ = shuffle_perm(data) if shuffle else (data, None)
        batches = batchify(data, batch_size)
        for data_batch in batches:
            opt_state = update(i, opt_state, data_batch)

        if i % log_every == 0:
            params = get_params(opt_state)
            logs = [log(params, i) for log in loggers]
            print(f"Epoch {i}", end=" ")
            for log in logs:
                print(f"[{log[0]} {float_log(log[1])}]", end=" ")
            print()

    return 0
