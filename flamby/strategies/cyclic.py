import copy
import time
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from .utils import DataLoaderWithMemory, _Model


class Cyclic:
    """Cyclic Weight Transfer Strategy Class.

    Under  the  cyclical weight transfer training strategy,
    the model is transferred, in a cyclic manner, to each client more than once.

    References
    ----------
    https://pubmed.ncbi.nlm.nih.gov/29617797/

    Attributes
    ----------
    training_dataloaders : List
            The list of training dataloaders from multiple training centers.

    model : torch.nn.Module
        An initialized torch model.

    loss : torch.nn.modules.loss._Loss
        The loss to minimize between the predictions of the model and the
        ground truth.

    optimizer_class : torch.optim.Optimizer
        The class of the torch model optimizer to use at each step.

    learning_rate : float
        The learning rate to be given to the optimizer_class.

    num_updates : int
        The number of updates to do on each client at each round.

    nrounds : int
        The number of communication rounds to do.

    log: bool
        Whether or not to store logs in tensorboard.

    bits_counting_function : callable
        A function making sure exchanges respect the rules, this function
        can be obtained by decorating check_exchange_compliance in
        flamby.utils. Should have the signature List[Tensor] -> int

    Methods
    ----------
    perform_round: Perform a single round of cyclic weight transfer

    run: Performs `self.nrounds` rounds of averaging and returns the final model.

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
        log: bool = False,
        log_period: int = 100,
        bits_counting_function: callable = None,
        rng: np.random._generator.Generator = None,
    ):
        """_summary_

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

        log: bool
            Whether or not to store logs in tensorboard.

        log_period: int

        bits_counting_function: callable
            A function making sure exchanges respect the rules, this function
            can be obtained by decorating check_exchange_compliance in
            flamby.utils. Should have the signature List[Tensor] -> int

        """

        self.training_dataloaders_with_memory = [
            DataLoaderWithMemory(e) for e in training_dataloaders
        ]
        self.training_sizes = [len(e) for e in self.training_dataloaders_with_memory]
        self.total_number_of_samples = sum(self.training_sizes)

        self.log = log
        self.log_period = log_period

        self.model = _Model(
            model=copy.deepcopy(model),
            optimizer_class=optimizer_class,
            lr=learning_rate,
            loss=loss,
            log=self.log,
            client_id=-1,
            log_period=self.log_period,
        )

        self.num_clients = len(training_dataloaders)
        self.nrounds = nrounds
        self.num_updates = num_updates

        self.bits_counting_function = bits_counting_function

        self.__rng = (
            rng if (rng is not None) else np.random.default_rng(int(time.time()))
        )
        self.__clients = self.__rng.permutation(self.num_clients)

        self.__current_idx = -1

    def perform_round(self):
        self.__current_idx += 1

        if self.__current_idx == self.num_clients:
            self.__clients = self.__rng.permutation(self.num_clients)
            self.__current_idx = 0

        self.model._local_train(
            dataloader_with_memory=self.training_dataloaders_with_memory[
                self.__clients[self.__current_idx]
            ],
            num_updates=self.num_updates,
        )

        updates = [self.model._get_current_params()]

        if self.bits_counting_function is not None:
            self.bits_counting_function(updates)

    def run(self) -> List[torch.nn.Module]:
        for _ in tqdm(range(self.nrounds)):
            self.perform_round()

        return [self.model.model]
