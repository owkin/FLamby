import torch

from flamby.datasets.fed_heart_disease.dataset import FedHeartDisease

NUM_CLIENTS = 4
BATCH_SIZE = 4
NUM_EPOCHS_POOLED = 50
LR = 0.001
Optimizer = torch.optim.Adam

FedClass = FedHeartDisease


def get_nb_max_rounds(num_updates, batch_size=BATCH_SIZE):
    return (486 // NUM_CLIENTS // batch_size) * NUM_EPOCHS_POOLED // num_updates
