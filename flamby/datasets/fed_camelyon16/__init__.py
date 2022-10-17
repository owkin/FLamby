from flamby.datasets.fed_camelyon16.common import (
    NUM_CLIENTS,
    NUM_EPOCHS_POOLED,
    LR,
    BATCH_SIZE,
    get_nb_max_rounds,
    Optimizer,
    FedClass,
)
from flamby.datasets.fed_camelyon16.dataset import (
    Camelyon16Raw,
    FedCamelyon16,
    collate_fn,
)
from flamby.datasets.fed_camelyon16.model import Baseline
from flamby.datasets.fed_camelyon16.metric import metric
from flamby.datasets.fed_camelyon16.loss import BaselineLoss
