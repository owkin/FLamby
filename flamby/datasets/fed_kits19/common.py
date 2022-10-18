import torch

from flamby.datasets.fed_kits19.dataset import FedKits19

NUM_CLIENTS = 6
BATCH_SIZE = 2
NUM_EPOCHS_POOLED = 500  # 8000 gives better performance but is too long
LR = 3e-4
SEEDS = [0]
Optimizer = torch.optim.Adam

FedClass = FedKits19


def get_nb_max_rounds(num_updates, batch_size=BATCH_SIZE):
    return (72 // NUM_CLIENTS // batch_size) * NUM_EPOCHS_POOLED // num_updates
