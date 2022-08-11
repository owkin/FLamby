from typing import List

import torch
from tqdm import tqdm

from flamby.strategies import FedAvg


class FedAvgFineTuning(FedAvg):
    """Federated Averaging with Fine-Tuning Strategy class.

    Federated Averaging with fine tuning is the most simple personalized FL strategy.
    First, all clients collaborate to learn a global model using FedAvg, then each
    client, independently, fine-tunes the parameters of the global model through
    few stochastic gradient descent steps using it local dataset.

    References
    ----------
    - https://arxiv.org/abs/1909.12488, 2019

    Parameters
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
    num_fine_tuning_steps: int
        The number of SGD fine-tuning updates to be performed on the
         model at the personalization step.
    log: bool, optional
        Whether or not to store logs in tensorboard. Defaults to False.
    log_period: int, optional
        If log is True then log the loss every log_period batch updates.
        Defauts to 100.
    bits_counting_function : Union[callable, None], optional
        A function making sure exchanges respect the rules, this function
        can be obtained by decorating check_exchange_compliance in
        flamby.utils. Should have the signature List[Tensor] -> int.
        Defaults to None.
    logdir: str, optional
        Where logs are stored. Defaults to ./runs.
    log_basename: str, optional
        The basename of the created log_file. Defaults to fed_avg.
    """

    def __init__(
        self,
        training_dataloaders: List,
        model: torch.nn.Module,
        loss: torch.nn.modules.loss._Loss,
        optimizer_class: torch.optim.Optimizer,
        learning_rate: float,
        num_updates: int,
        nrounds: int,
        num_fine_tuning_steps: int,
        log: bool = False,
        log_period: int = 100,
        bits_counting_function: callable = None,
        logdir: str = "./runs",
        log_basename: str = "fed_avg_ft",
    ):
        super().__init__(
            training_dataloaders,
            model,
            loss,
            optimizer_class,
            learning_rate,
            num_updates,
            nrounds,
            log,
            log_period,
            bits_counting_function,
            log_basename=log_basename,
            logdir=logdir,
        )

        self.num_fine_tuning_steps = num_fine_tuning_steps

    def run(self):
        """This method performs self.nrounds rounds of averaging
        and returns the list of models.
        """
        for _ in tqdm(range(self.nrounds)):
            self.perform_round()

        for _model, dataloader_with_memory in zip(
            self.models_list, self.training_dataloaders_with_memory
        ):
            _model._local_train(dataloader_with_memory, self.num_fine_tuning_steps)

        return [m.model for m in self.models_list]
