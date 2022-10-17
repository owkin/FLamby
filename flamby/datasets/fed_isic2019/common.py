import torch

from flamby.datasets.fed_isic2019.dataset import FedIsic2019

NUM_CLIENTS = 6
BATCH_SIZE = 64
NUM_EPOCHS_POOLED = 20
LR = 5e-4
Optimizer = torch.optim.Adam

FedClass = FedIsic2019


def get_nb_max_rounds(num_updates, batch_size=BATCH_SIZE):
    # TODO find out true number
    return (18413 // NUM_CLIENTS // batch_size) * NUM_EPOCHS_POOLED // num_updates
