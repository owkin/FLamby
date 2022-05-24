import torch

NUM_CLIENTS = 4
NUM_EPOCHS_POOLED = 100
BATCH_SIZE = 1
LR = 1e-2
SEEDS = [42]

Optimizer = torch.optim.Adam


def get_nb_max_rounds(num_updates, batch_size=BATCH_SIZE):
    return (800 // NUM_CLIENTS // batch_size) * NUM_EPOCHS_POOLED // num_updates
