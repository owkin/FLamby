import torch

NUM_CLIENTS = 6
BATCH_SIZE = 2
NUM_EPOCHS_POOLED = 2000
LR = 3e-4
SEEDS = [0, 10, 20, 30, 40]
Optimizer = torch.optim.Adam


def get_nb_max_rounds(num_updates):
    return (72 // NUM_CLIENTS // BATCH_SIZE) * NUM_EPOCHS_POOLED // num_updates
