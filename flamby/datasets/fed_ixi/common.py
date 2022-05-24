import torch

NUM_CLIENTS = 3
BATCH_SIZE = 16
SEEDS = [42]
NUM_EPOCHS_POOLED = 5


Optimizer = torch.optim.Adam


def get_nb_max_rounds(num_updates):
    return (450 // NUM_CLIENTS // BATCH_SIZE) * NUM_EPOCHS_POOLED // num_updates
