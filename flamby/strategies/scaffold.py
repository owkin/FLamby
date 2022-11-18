import warnings
from typing import List

import torch

from flamby.strategies.fed_avg import FedAvg
from flamby.strategies.utils import _Model


class Scaffold(FedAvg):
    """SCAFFOLD Strategy class

    SCAFFOLD is a stateful algorithm which modifies the local update steps of FedAvg
    in order to provably correct for data heterogeneity across clients. If the data
    on each client is very different, their local updates via FedAvg will move in
    different directions. Each client maintains a 'correction' which estimates
    this difference between client updates and global average update. This correction
    is added to every local update on the client.
    This is a more efficient implementation of Scaffold whose communication and
    computation requirement exactly matches that of FedAvg.
    The current implementation assumes that SGD is the local optimizer, and that all
    clients participate every round.

    References
    ----------
    https://arxiv.org/abs/1910.06378

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
        It has to be SGD.
    learning_rate : float
        The learning rate to be given to the clients optimizer_class.
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
    server_learning_rate : float
        The learning rate with which the server's updates are aggregated.
        Defaults to 1.
    log: bool
        Whether or not to store logs in tensorboard. Defaults to False.
    log_period: int
        If log is True then log the loss every log_period batch updates.
        Defauts to 100.
    bits_counting_function : Union[callable, None]
        A function making sure exchanges respect the rules, this function
        can be obtained by decorating check_exchange_compliance in
        flamby.utils. Should have the signature List[Tensor] -> int.
        Defaults to None.
    logdir: str
        Where to store the logs. Defaulst to ./runs.
    log_basename: str
        The basename of the created logfile. Defaulst to scaffold.
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
        dp_target_epsilon: float = None,
        dp_target_delta: float = None,
        dp_max_grad_norm: float = None,
        server_learning_rate: float = 1,
        log: bool = False,
        log_period: int = 100,
        bits_counting_function: callable = None,
        logdir: str = "./runs",
        log_basename: str = "scaffold",
    ):
        """Cf class docstring"""

        assert (
            optimizer_class == torch.optim.SGD
        ), "Only SGD for client optimizer with Scaffold"

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
        )

        # Add a warning if user wants to make DP
        if dp_target_epsilon is not None:
            warnings.warn("Warning, the DP bounds passed are not valid for Scaffold.")

        # initialize the previous state of each client
        self.previous_client_state_list = [
            _model._get_current_params() for _model in self.models_list
        ]
        # initialize the corrections used by each client to 0s.
        self.client_corrections_state_list = [
            [torch.zeros_like(torch.from_numpy(p)) for p in _model._get_current_params()]
            for _model in self.models_list
        ]
        self.client_lr = learning_rate
        self.server_lr = server_learning_rate

    def _local_optimization(
        self, _model: _Model, dataloader_with_memory, correction_state: List
    ):
        """Carry out the local optimization step.

        Parameters
        ----------
        _model: _Model
            The model on the local device used by the optimization step.
        dataloader_with_memory : dataloaderwithmemory
            A dataloader that can be called infinitely using its get_samples()
            method.
        correction_state: List
            Correction to be applied to the model state during every local update.
        """
        _model._local_train_with_correction(
            dataloader_with_memory, self.num_updates, correction_state
        )

    def perform_round(self):
        """Does a single federated averaging round. The following steps will be
        performed:

        - each model will be trained locally for num_updates batches.
        - the parameter updates will be collected and averaged. Averages will be
          weighted by the number of samples in each client
        - the averaged updates willl be used to update the local model
        """
        local_updates = list()
        new_client_state_list = list()
        new_correction_state_list = list()
        for (
            _model,
            dataloader_with_memory,
            size,
            _previous_client_state,
            _prev_correction_state,
        ) in zip(
            self.models_list,
            self.training_dataloaders_with_memory,
            self.training_sizes,
            self.previous_client_state_list,
            self.client_corrections_state_list,
        ):
            # Update as correction += (server_state - previous_state) / lr*num_updates.
            _server_state = _model._get_current_params()
            _new_correction_state = [
                c
                + torch.from_numpy(
                    (p - q) / (self.server_lr * self.client_lr * self.num_updates)
                )
                for c, p, q in zip(
                    _prev_correction_state, _server_state, _previous_client_state
                )
            ]
            new_correction_state_list.append(_new_correction_state)
            del _previous_client_state
            del _prev_correction_state

            # Local Optimization
            self._local_optimization(
                _model, dataloader_with_memory, _new_correction_state
            )
            _local_next_state = _model._get_current_params()

            # Scale local parameters by server_lr
            _local_next_state = [
                self.server_lr * new + (1 - self.server_lr) * old
                for new, old in zip(_local_next_state, _server_state)
            ]
            new_client_state_list.append(_local_next_state)

            # Recovering updates
            updates = [new - old for new, old in zip(_local_next_state, _server_state)]
            del _local_next_state

            # Reset local model
            for p_new, p_old in zip(_model.model.parameters(), _server_state):
                p_new.data = torch.from_numpy(p_old).to(p_new.device)
            del _server_state

            if self.bits_counting_function is not None:
                self.bits_counting_function(updates)

            local_updates.append({"updates": updates, "n_samples": size})

        # update previous client states
        self.previous_client_state_list = new_client_state_list
        self.client_corrections_state_list = new_correction_state_list

        # Aggregation step
        aggregated_delta_weights = [
            None for _ in range(len(local_updates[0]["updates"]))
        ]
        for idx_weight in range(len(local_updates[0]["updates"])):
            aggregated_delta_weights[idx_weight] = sum(
                [
                    local_updates[idx_client]["updates"][idx_weight]
                    * local_updates[idx_client]["n_samples"]
                    for idx_client in range(self.num_clients)
                ]
            )
            aggregated_delta_weights[idx_weight] /= float(self.total_number_of_samples)

        # Update models
        for _model in self.models_list:
            _model._update_params(aggregated_delta_weights)
