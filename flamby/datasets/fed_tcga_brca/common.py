import torch

NUM_CLIENTS = 6
BATCH_SIZE = 32
NUM_EPOCHS_POOLED = 30
LR = 1e-1
Optimizer = torch.optim.Adam


def get_nb_max_rounds(num_updates, batch_size=BATCH_SIZE):
    # TODO find out true number
    return (866 // BATCH_SIZE) * NUM_EPOCHS_POOLED // num_updates
