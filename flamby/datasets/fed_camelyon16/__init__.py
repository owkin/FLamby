from .common import NUM_CLIENTS, NUM_EPOCHS_POOLED, LR, BATCH_SIZE, get_nb_max_rounds
from .dataset import Camelyon16Raw, FedCamelyon16, collate_fn
from .model import Baseline
from .metric import metric
from .loss import BaselineLoss

