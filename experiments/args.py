import argparse

# boolean args:
# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse/36031646
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():

    # Command line arguments definitions
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=100, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate for the Adam optimizer")
    #parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    #parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--log_every", type=int, default=10, help="interval between logs")
    parser.add_argument("--seed", type=int, default=42, help="random generator seed")
    parser.add_argument("--dataset", type=str, default="MNIST", help="training dataset")
    parser.add_argument("--num_layers", type=int, default=2, help="define a model with `num_layers - 1` linear layers and 1 final affine layer")
    parser.add_argument("--nonlinearity", type=str, default="smooth_leaky_relu", help="nonlinear activation function")
    parser.add_argument("--alpha", type=float, default=0.01, help="angular coefficient for the left side of RELU-type activation functions")
    parser.add_argument("--look_ahead", type=int, default=30, help="stop training if no imrpovement has been observed for `look_ahead` epochs")
    parser.add_argument("--log_dir", type=str, default="log/", help="directory in which to save the model (ending in '/')")
    parser.add_argument("--bias", type=str2bool, nargs='?',
                        const=True, default=True,
                        help="whether to include the bias in the linear layers or not")
    parser.add_argument("--trick", type=str2bool, nargs='?',
                        const=True, default=True,
                        help="whether to apply the relative trick to the gradients or not")
    parser.add_argument("--generation", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="whether to perform data generation during training")

    parser.add_argument("--toy_name", type=str, default='sine', help="choice of toy distribution")

    return parser.parse_args()
