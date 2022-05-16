NUM_CLIENTS = 4
BATCH_SIZE = 32
NUM_EPOCHS_POOLED = 50
LR = 0.05


def get_nb_max_rounds(num_updates, batch_size=BATCH_SIZE):
    return (495 // BATCH_SIZE) * NUM_EPOCHS_POOLED // num_updates
