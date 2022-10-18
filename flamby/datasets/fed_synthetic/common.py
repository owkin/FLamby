import torch

from flamby.datasets.fed_synthetic.dataset import FedSynthetic
from flamby.utils import get_config_file_path, read_config

NUM_CLIENTS = 4
BATCH_SIZE = 8
NUM_EPOCHS_POOLED = 10
LR = 0.005
Optimizer = torch.optim.Adam

FedClass = FedSynthetic


def get_nb_max_rounds(num_updates, batch_size=BATCH_SIZE):
    # not clear how this should be set

    dict = read_config(get_config_file_path("fed_synthetic", False))

    return (
        (dict["n_samples"] // NUM_CLIENTS // batch_size)
        * NUM_EPOCHS_POOLED
        // num_updates
    )
