from typing import List

import torch

from flamby.strategies import FedAvg
from flamby.strategies.utils import _Model


class FedProx(FedAvg):
    """FedProx Strategy class.

    The FedProx strategy is a generalization and re-parametrization of FedAvg that adds
    a proximal term. Each client first trains his version of a global model locally using
    a proximal term, the states of the model of each client are then weighted-averaged
    and returned to each client for further training.

    References
    ----------
    - https://arxiv.org/abs/1812.06127
    - https://github.com/litian96/FedProx

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
    dp_target_epsilon: float
        The target epsilon for (epsilon, delta)-differential
         private guarantee. Defaults to None.
    dp_target_delta: float
        The target delta for (epsilon, delta)-differential
         private guarantee. Defaults to None.
    dp_max_grad_norm: float
        The maximum L2 norm of per-sample gradients;
        used to enforce differential privacy. Defaults to None.
    mu: float
        The mu parameter involved in the proximal term. If mu = 0, then FedProx
        is reduced to FedAvg. Need to be tuned, there are no default mu values
        that would work for all settings.
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
    log_basename: str, optional
        The basename of the created log file. Defaults to fed_prox.
    logdir: str, optional
        The directory where to store the logs. Defaults to ./runs.

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
        mu: float,
        dp_target_epsilon: float = None,
        dp_target_delta: float = None,
        dp_max_grad_norm: float = None,
        seed=None,
        log: bool = False,
        log_period: int = 100,
        bits_counting_function: callable = None,
        log_basename: str = "fed_prox",
        logdir: str = "./runs",
    ):
        """Cf class docstring"""
        super().__init__(
            training_dataloaders,
            model,
            loss,
            optimizer_class,
            learning_rate,
            num_updates,
            nrounds,
            dp_target_epsilon,
            dp_target_delta,
            dp_max_grad_norm,
            log,
            log_period,
            bits_counting_function,
            log_basename=log_basename,
            logdir=logdir,
            seed=seed,
        )
        self.mu = mu

    def _local_optimization(self, _model: _Model, dataloader_with_memory):
        """Carry out the local optimization step.

        Parameters
        ----------
        _model: _Model
            The model on the local device used by the optimization step.
        dataloader_with_memory : dataloaderwithmemory
            A dataloader that can be called infinitely using its get_samples()
            method.
        """
        _model._prox_local_train(dataloader_with_memory, self.num_updates, self.mu)
