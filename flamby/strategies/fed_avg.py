from typing import List

import torch
from tqdm import tqdm

from flamby.strategies.utils import DataLoaderWithMemory, _Model


class FedAvg:
    """Federated Averaging Strategy class.

    The Federated Averaging strategy is the most simple centralized strategy.
    Each client first trains his version of a global model locally on its data,
    the states of the model of each client are then weighted-averaged and returned
    to each client for further training.

    References
    ----------
    - https://arxiv.org/abs/1602.05629

    """

    def __init__(
        self,
        training_dataloaders: List[torch.utils.data.dataloader],
        model: torch.nn.Module,
        loss: torch.nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer,
        num_updates: int,
        nrounds: int,
    ):
        self.training_dataloaders_with_memory = [
            DataLoaderWithMemory(e) for e in training_dataloaders
        ]
        self.training_sizes = [len(e) for e in self.training_dataloaders_with_memory]
        self.total_number_of_samples = sum(self.training_sizes)
        self.models_list = [
            _Model(model=model, optimizer=optimizer, loss=loss)
            for _ in range(len(training_dataloaders))
        ]
        self.nrounds = nrounds
        self.num_updates = num_updates
        self.num_clients = len(self.training_sizes)

    def perform_round(self):
        """a single federated averaging round. The following steps will be performed:

        - each model will be trained locally for num_updates batches.
        - the parameters will be collected and averaged. Average will be
            weighted by the number of samples used by each model
        - the averaged parameters will be returned and set at each model
        """
        local_updates = list()
        for _model, dataloader_with_memory, size in zip(
            self.models_list, self.training_dataloaders_with_memory, self.training_sizes
        ):
            # Local Optimization
            _local_previous_state = _model._get_current_params()
            _model._local_train(dataloader_with_memory)
            _local_next_state = _model._get_current_params()
            # Recovering updates
            updates = [
                new - old for new, old in zip(_local_next_state, _local_previous_state)
            ]
            del _local_next_state
            # Reset local model
            for p_new, p_old in zip(_model.parameters(), _local_previous_state):
                p_new.data = p_old
            del _local_previous_state
            local_updates.append({"updates": updates, "n_samples": size})

        # Aggregation step
        aggregated_delta_weights = [None for _ in range(len(local_updates[0]))]
        for idx_weight in range(len(local_updates[0])):
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

    def run(self):
        """This method performs self.nrounds rounds of averaging
        and returns the list of models.
        """
        for _ in tqdm(range(self.nrounds)):
            self.perform_round()
        return [m.model for m in self.models_list]
