import torch

from flamby.datasets.fed_ixi.dataset import FedIXITiny

NUM_CLIENTS = 3
BATCH_SIZE = 2

SEEDS = [0, 10, 20, 30, 40]
NUM_EPOCHS_POOLED = 10
LR = 0.001

Optimizer = torch.optim.AdamW

FedClass = FedIXITiny

# Be careful it changes frequently if the download doesn't work update this link
DATASET_URL = (
    "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1"
    ".amazonaws.com/7kd5wj7v7p-3.zip"
)
FOLDER = DATASET_URL.split("/")[-1].split(".")[0]


def get_nb_max_rounds(num_updates, batch_size=BATCH_SIZE):
    return (450 // NUM_CLIENTS // batch_size) * NUM_EPOCHS_POOLED // num_updates
