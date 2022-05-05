from typing import List

import torch
from tqdm import tqdm

from flamby.strategies.utils import DataLoaderWithMemory, _Model


class FedAvg:
    """Federated Averaging Strategy class.

    The Federated Averaging strategy is the most simple centralized FL strategy.
    Each client first trains his version of a global model locally on its data,
    the states of the model of each client are then weighted-averaged and returned
    to each client for further training.

    References
    ----------
    - https://arxiv.org/abs/1602.05629

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
        log: bool = False,
        log_period: int = 100,
        bits_counting_function: callable = None,
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
            The class of the torch model optimizer to use at each step.
        learning_rate : float
            The learning rate to be given to the optimizer_class.
        num_updates : int
            The number of updates to do on each client at each round.
        nrounds : int
            The number of communication rounds to do.
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
        """
        self.training_dataloaders_with_memory = [
            DataLoaderWithMemory(e) for e in training_dataloaders
        ]
        self.training_sizes = [len(e) for e in self.training_dataloaders_with_memory]
        self.total_number_of_samples = sum(self.training_sizes)
        self.log = log
        self.log_period = log_period
        self.models_list = [
            _Model(
                model=model,
                optimizer_class=optimizer_class,
                lr=learning_rate,
                loss=loss,
                log=self.log,
                client_id=i,
                log_period=self.log_period,
            )
            for i in range(len(training_dataloaders))
        ]
        self.nrounds = nrounds
        self.num_updates = num_updates
        self.num_clients = len(self.training_sizes)
        self.bits_counting_function = bits_counting_function

    def perform_round(self):
        """Does a single federated averaging round. The following steps will be
        performed:

        - each model will be trained locally for num_updates batches.
        - the parameter updates will be collected and averaged. Averages will be
            weighted by the number of samples in each client
        - the averaged updates willl be used to update the local model
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

        print(aggregated_delta_weights[0])
        print(type(aggregated_delta_weights[0]))
        print(aggregated_delta_weights[0].is_cuda)

        # Update models
        for _model in self.models_list:
            _model._update_params(aggregated_delta_weights)

    def run(self):
        """This method performs self.nrounds rounds of averaging
        and returns the list of models.
        """
        for _ in tqdm(range(self.nrounds)):
            self.perform_round()
        return [m.model for m in self.models_list]
