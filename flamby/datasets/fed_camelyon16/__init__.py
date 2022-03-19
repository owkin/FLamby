from .common import NUM_CLIENTS, NB_MAX_ROUNDS
from .dataset import Camelyon16Raw, FedCamelyon16, collate_fn
from .model import Baseline
from .metric import metric
from .loss import BaselineLoss