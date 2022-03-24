NUM_CLIENTS = 2
BATCH_SIZE = 32
NUM_EPOCHS_POOLED = 100
LR = 0.001


def get_nb_max_rounds(num_updates, batch_size=BATCH_SIZE):
    return (270 // BATCH_SIZE) * NUM_EPOCHS_POOLED // num_updates
