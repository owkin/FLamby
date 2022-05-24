import torch

NUM_CLIENTS = 2
BATCH_SIZE = 16
NUM_EPOCHS_POOLED = 15
LR = 0.001

Optimizer = torch.optim.Adam


def get_nb_max_rounds(num_updates, batch_size=BATCH_SIZE):
    return (270 // NUM_CLIENTS // BATCH_SIZE) * NUM_EPOCHS_POOLED // num_updates
