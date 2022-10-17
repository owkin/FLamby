import torch

from flamby.datasets.fed_tcga_brca.dataset import FedTcgaBrca

NUM_CLIENTS = 6
BATCH_SIZE = 8
NUM_EPOCHS_POOLED = 30
LR = 1e-1
Optimizer = torch.optim.Adam

FedClass = FedTcgaBrca


def get_nb_max_rounds(num_updates, batch_size=BATCH_SIZE):
    # TODO find out true number
    return (866 // NUM_CLIENTS // batch_size) * NUM_EPOCHS_POOLED // num_updates
