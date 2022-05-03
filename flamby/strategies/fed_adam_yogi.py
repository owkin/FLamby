from typing import List

import numpy as np
import torch

from flamby.strategies.fed_avg import FedAvg


class FedAdamYogi(FedAvg):
    """FedAdam and FedYogi Strategy class

    References
    ----------
    https://arxiv.org/abs/2003.00295

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
        log: bool,
        log_period: int = 100,
        bits_counting_function: callable = None,
        tau: float = 1e-3,
        server_learning_rate: float = 1e-2,
        yogi: bool = False,
    ):
        """_summary_

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
            This is the client optimizer, it has to be SGD is FedAdam is chosen
            for the server optimizer. The adaptive logic sits with the server
            optimizer and is coded below with the aggregation.
        learning_rate : float
            The learning rate to be given to the client optimizer_class.
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
        tau: float
            adaptivity hyperparameter for the Adam/Yogi optimizer.
        server_learning_rate : float
            The learning rate used by the server optimizer.
        yogi: bool
            False dy default. The optimizer is FedAdam if False, FedYogi if True.
        """

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
        )

        assert (
            optimizer_class == torch.optim.SGD
        ), "Only SGD for client optimizer with FedAdam or FedYogi"

        self.beta1 = 0.9  # momentum parameter
        self.beta2 = 0.999  # second moment parameter
        self.m = [
            np.zeros_like(param) for param in self.models_list[0]._get_current_params()
        ]  # momentum
        self.v = [
            np.zeros_like(param) for param in self.models_list[0]._get_current_params()
        ]  # second moment
        self.updates = [
            np.zeros_like(param) for param in self.models_list[0]._get_current_params()
        ]  # param update to be applied by the server optimizer
        self.tauarray = [
            np.ones_like(param) * tau
            for param in self.models_list[0]._get_current_params()
        ]  # adaptivity HP for Adam and Yogi
        self.server_learning_rate = server_learning_rate
        self.yogi = yogi

    def perform_round(self):
        """Does a single federated round. The following steps will be
        performed:

        - each model will be trained locally for num_updates batches.
        - the parameter updates will be collected and averaged. Averages will be
            weighted by the number of samples in each client.
        - the averaged updates will be processed the same way as Adam or Yogi
            algorithms do in a non-federated setting.
        - the averaged updates will be used to update the local models.
        """
        local_updates = list()
        for _model, dataloader_with_memory, size in zip(
            self.models_list, self.training_dataloaders_with_memory, self.training_sizes
        ):
            # Local Optimization
            _local_previous_state = _model._get_current_params()
            _model._local_train(dataloader_with_memory, self.num_updates)
            _local_next_state = _model._get_current_params()
            # Recovering updates
            updates = [
                new - old for new, old in zip(_local_next_state, _local_previous_state)
            ]
            del _local_next_state
            # Reset local model
            for p_new, p_old in zip(_model.model.parameters(), _local_previous_state):
                p_new.data = torch.from_numpy(p_old).to(p_new.device)
            del _local_previous_state

            if self.bits_counting_function is not None:
                self.bits_counting_function(updates)

            local_updates.append({"updates": updates, "n_samples": size})

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

        # Update momentum and second moment, calculate parameter updates
        for param_idx in range(len(self.m)):
            self.m[param_idx] = (
                self.beta1 * self.m[param_idx]
                + (1 - self.beta1) * aggregated_delta_weights[param_idx]
            )
        for param_idx in range(len(self.v)):
            if self.yogi:
                sign = np.sign(
                    self.v[param_idx]
                    - aggregated_delta_weights[param_idx]
                    * aggregated_delta_weights[param_idx]
                )
                self.v[param_idx] = (
                    self.v[param_idx]
                    - (1 - self.beta2)
                    * aggregated_delta_weights[param_idx]
                    * aggregated_delta_weights[param_idx]
                    * sign
                )
            else:
                self.v[param_idx] = (
                    self.beta2 * self.v[param_idx]
                    + (1 - self.beta2)
                    * aggregated_delta_weights[param_idx]
                    * aggregated_delta_weights[param_idx]
                )
        for param_idx in range(len(self.updates)):
            self.updates[param_idx] = (
                self.server_learning_rate
                * self.m[param_idx]
                / (np.sqrt(self.v[param_idx]) + self.tauarray[param_idx])
            )

        # Update models
        for _model in self.models_list:
            _model._update_params(self.updates)
