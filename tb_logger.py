from tensorboardX import SummaryWriter
import numpy as onp

# TODO: get rid of `numpy` import? `tb_histograms` should return histograms...


def tensorboard_decorator(log_dir):
    writer = SummaryWriter(log_dir)

    def tb_scalars(logger):
        def decorated(params, epoch):
            title, values = logger(params, epoch)
            for (i, val) in enumerate(values):
                writer.add_scalar(f"{title}/{i}", onp.float32(val), epoch)
                writer.flush()
            return title, values
        return decorated

    def tb_histograms(logger):
        def decorated(params, epoch):
            title, values = logger(params, epoch)
            for (i, val) in enumerate(values):
                writer.add_histogram(f"{title}/{i}", onp.asarray(val), epoch)
                writer.flush()
            m_std = lambda x: onp.linalg.norm(x)  # (onp.mean(x), onp.std(x))
            return title, [m_std(v) for v in values]
        return decorated

    def tb_figures(logger):
        def decorated(params, epoch):
            title, values = logger(params, epoch)
            for (i, val) in enumerate(values):
                writer.add_figure(f"{title}/{i}", val, epoch)
                writer.flush()
            return title, len(values)
        return decorated

    return tb_scalars, tb_histograms, tb_figures, writer
