import time
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from flamby.strategies.utils import DataLoaderWithMemory, _Model


class Cyclic:
    """Cyclic Weight Transfer Strategy Class.

    Under  the  cyclical weight transfer training strategy,
    the model is transferred, in a cyclic manner, to each client more than once.

    References
    ----------
    https://pubmed.ncbi.nlm.nih.gov/29617797/

    Parameters
    ----------
    training_dataloaders: List[torch.utils.data.DataLoader]
         The list of training dataloaders from multiple training centers.

    model: torch.nn.Module
         An initialized torch model.

    loss: torch.nn.modules.loss._Loss
         The loss to minimize between the predictions of the model and the
         ground truth.

    optimizer_class: callable torch.optim.Optimizer
         The class of the torch model optimizer to use at each step.

    learning_rate: float
         The learning rate to be given to the optimizer_class.

    num_updates: int
         The number of epochs to do on each client at each round.

    nrounds: int
          The number of communication rounds to do.

    dp_target_epsilon: float
        The target epsilon for (epsilon, delta)-differential
         private guarantee. Defaults to None.

    dp_target_delta: float
        The target delta for (epsilon, delta)-differential
         private guarantee. Defaults to None.

    dp_max_grad_norm: float
        The maximum L2 norm of per-sample gradients;
         used to enforce differential privacy. Defaults to None.

    log: bool, optional
         Whether or not to store logs in tensorboard.

    log_period: int, optional

    bits_counting_function: callable, optional
         A function making sure exchanges respect the rules, this function
         can be obtained by decorating check_exchange_compliance in
         flamby.utils. Should have the signature List[Tensor] -> int

    deterministic_cycle: bool, optional
         if True, we cycle through clients in their original order,
         otherwise, the clients are reshuffled at the beginning of every cycle.

    rng: np.random._generator.Generator, optional
         used to reshuffle the clients. Defaults to None.

    logdir: str, optional
         The path where to store the logs if there are some. Defaults to ./runs.
    log_basename: str
         The basename of the created log file. Defaults to cyclic.
    """

    def __init__(
        self,
        training_dataloaders: List[torch.utils.data.DataLoader],
        model: torch.nn.Module,
        loss: torch.nn.modules.loss._Loss,
        optimizer_class: callable,
        learning_rate: float,
        num_updates: int,
        nrounds: int,
        dp_target_epsilon: float = None,
        dp_target_delta: float = None,
        dp_max_grad_norm: float = None,
        seed=None,
        log: bool = False,
        log_period: int = 100,
        bits_counting_function: callable = None,
        deterministic_cycle: bool = False,
        rng: np.random._generator.Generator = None,
        log_basename: str = "cyclic",
        logdir: str = "./runs",
    ):
        """Cf class docstring"""

        self.training_dataloaders_with_memory = [
            DataLoaderWithMemory(e) for e in training_dataloaders
        ]
        self.training_sizes = [len(e) for e in self.training_dataloaders_with_memory]
        self.total_number_of_samples = sum(self.training_sizes)

        self.dp_target_epsilon = dp_target_epsilon
        self.dp_target_delta = dp_target_delta
        self.dp_max_grad_norm = dp_max_grad_norm
        self._seed = seed

        self.log = log
        self.log_period = log_period
        self.log_basename = log_basename + f"-deterministic{deterministic_cycle}"
        self.logdir = logdir

        self.models_list = [
            _Model(
                model=model,
                optimizer_class=optimizer_class,
                lr=learning_rate,
                train_dl=_train_dl,
                dp_target_epsilon=self.dp_target_epsilon,
                dp_target_delta=self.dp_target_delta,
                dp_max_grad_norm=self.dp_max_grad_norm,
                loss=loss,
                nrounds=nrounds,
                log=self.log,
                client_id=i,
                log_period=self.log_period,
                log_basename=self.log_basename,
                logdir=self.logdir,
                seed=self._seed,
            )
            for i, _train_dl in enumerate(training_dataloaders)
        ]

        self.num_clients = len(training_dataloaders)
        self.nrounds = nrounds
        self.num_updates = num_updates

        self.bits_counting_function = bits_counting_function

        self.deterministic_cycle = deterministic_cycle

        self._rng = rng if (rng is not None) else np.random.default_rng(int(time.time()))

        self._clients = self._shuffle_clients()
        self._current_idx = -1

    def _shuffle_clients(self):
        if self.deterministic_cycle:
            _clients = np.arange(self.num_clients)

        else:
            _clients = self._rng.permutation(self.num_clients)
        return _clients

    def perform_round(self):
        self._current_idx += 1

        if self._current_idx == self.num_clients:
            self._clients = self._shuffle_clients()

            self._current_idx = 0

        current_model = self.models_list[self._clients[self._current_idx]]

        current_model._local_train(
            dataloader_with_memory=self.training_dataloaders_with_memory[
                self._clients[self._current_idx]
            ],
            num_updates=self.num_updates,
        )

        updates = [current_model._get_current_params()]

        if self.bits_counting_function is not None:
            self.bits_counting_function(updates)

    def run(self) -> List[torch.nn.Module]:
        for _ in tqdm(range(self.nrounds)):
            self.perform_round()

        return [m.model for m in self.models_list]
