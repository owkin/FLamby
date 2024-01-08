import torch

from flamby.datasets.fed_camelyon16.dataset import FedCamelyon16

NUM_CLIENTS = 2
BATCH_SIZE = 16
NUM_EPOCHS_POOLED = 45
LR = 0.001

Optimizer = torch.optim.Adam

FedClass = FedCamelyon16


def get_nb_max_rounds(num_updates, batch_size=BATCH_SIZE):
    return (270 // NUM_CLIENTS // batch_size) * NUM_EPOCHS_POOLED // num_updates
