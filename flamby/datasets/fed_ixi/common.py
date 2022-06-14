import torch

NUM_CLIENTS = 3
BATCH_SIZE = 2

SEEDS = [0, 10, 20, 30, 40]
NUM_EPOCHS_POOLED = 10
LR = 0.001

Optimizer = torch.optim.AdamW


def get_nb_max_rounds(num_updates, batch_size=BATCH_SIZE):
    return (450 // NUM_CLIENTS // batch_size) * NUM_EPOCHS_POOLED // num_updates
