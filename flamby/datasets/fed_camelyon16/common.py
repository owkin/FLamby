import pandas as pd

NUM_CLIENTS = 2
BATCH_SIZE = 32
NUM_EPOCHS_POOLED = 40
LR = 0.001

def get_nb_max_rounds(num_updates, batch_size=BATCH_SIZE):
    # TODO replace 300 with the actual precise number
    return (300 // BATCH_SIZE) * NUM_EPOCHS_POOLED // 50