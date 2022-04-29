import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader

from flamby.datasets.fed_camelyon16 import (
    BATCH_SIZE,
    LR,
    NUM_CLIENTS,
    Baseline,
    BaselineLoss,
    FedCamelyon16,
    get_nb_max_rounds,
    metric,
)
from flamby.strategies.cyclic import Cyclic
from flamby.utils import evaluate_model_on_tests

NUM_LOCAL_EPOCHS = 2
SEED = 42
LOG_PERIOD = 10


def test_cyclic_camelyon():
    training_dls = [
        DataLoader(
            FedCamelyon16(train=True, center=i),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=10,
        )
        for i in range(NUM_CLIENTS)
    ]

    test_dls = [
        DataLoader(
            FedCamelyon16(train=False, center=i),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=10,
        )
        for i in range(NUM_CLIENTS)
    ]

    loss = BaselineLoss()
    m = Baseline()

    nrounds = get_nb_max_rounds(NUM_LOCAL_EPOCHS)

    s = Cyclic(
        training_dataloaders=training_dls,
        model=m,
        loss=loss,
        optimizer_class=optim.SGD,
        learning_rate=LR,
        num_updates=NUM_LOCAL_EPOCHS,
        nrounds=nrounds,
        log=True,
        log_period=LOG_PERIOD,
        rng=np.random.default_rng(SEED),
    )

    m = s.run()
    print(evaluate_model_on_tests(m, test_dls, metric))
