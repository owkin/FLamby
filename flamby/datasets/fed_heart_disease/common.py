import torch

NUM_CLIENTS = 4
BATCH_SIZE = 8
NUM_EPOCHS_POOLED = 50
LR = 0.001
Optimizer = torch.optim.Adam

def get_nb_max_rounds(num_updates, batch_size=BATCH_SIZE):
    return (486 // BATCH_SIZE) * NUM_EPOCHS_POOLED // num_updates
