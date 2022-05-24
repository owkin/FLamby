import torch

NUM_CLIENTS = 3
BATCH_SIZE = 16
SEEDS = [42]
NUM_EPOCHS_POOLED = 5


Optimizer = torch.optim.AdamW


def get_nb_max_rounds(num_updates, batch_size=BATCH_SIZE):
    return (450 // NUM_CLIENTS // batch_size) * NUM_EPOCHS_POOLED // num_updates
