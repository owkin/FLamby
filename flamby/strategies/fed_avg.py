import copy
from typing import List

import numpy as np
import torch
from tqdm import tqdm


class FedAvgTorch:
    """Federated Average Strategy for Torch.

    Federated Average strategy is the most classic of strategies.
    Each client has different set of the data on which a
    the training round(s) are performed. The state of the model of each client
    is then weighted-averaged and return to each client for further training.
    """

    def __init__(
        self,
        training_dataloaders: List,  # [torch.utils.data.dataset],
        model: torch.nn.Module,
        loss: torch.nn.modules.loss._Loss,
        nrounds: int,
        optimizer: torch.optim.Optimizer,
    ):
        self.training_dataloaders = training_dataloaders
        self.models_list = [
            _Model(model=model, optimizer=optimizer, loss=loss)
            for _ in range(len(training_dataloaders))
        ]
        self.nrounds = nrounds

    def perform_round(self):
        """a single federated averaging round. The following steps will be performed:

        - each model will be trained locally with its full training dataset
        - the parameters will be collected and averaged. Average will be
            weighted by the number of samples used by each model
        - the averaged parameters will be returned and set at each model
        """
        local_states = list()
        all_samples = 0
        for _model, dataloader in zip(self.models_list, self.training_dataloaders):
            # make one training step of the model
            _model._local_train(dataloader)  # training mode

            _local_state, n_samples = _model._get_current_params()
            all_samples += n_samples
            local_states.append({"state": _local_state, "n_samples": n_samples})

        averaged = list()
        for idx in range(len(local_states[0]["state"])):
            states = list()
            for state in local_states:
                states.append(state["state"][idx] * (state["n_samples"] / all_samples))
            averaged.append(np.sum(states, axis=0))

        for _model in self.models_list[:2]:
            for _model in self.models_list:

                # update the model to the averaged params
                _model._update_params(averaged)

    def run(self):
        for _ in tqdm(range(self.nrounds)):
            self.perform_round()
        return [m.model for m in self.models_list]


class _Model:
    def __init__(self, model, optimizer, loss):
        self.model = copy.deepcopy(model)
        self._optimizer = copy.deepcopy(optimizer)
        self._loss = copy.deepcopy(loss)
        init_seed = 42
        torch.manual_seed(init_seed)
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self._device)
        self.print_progress = True

    def _local_train(self, dataloader):
        # Local train
        self.n_samples = 0
        _size = len(dataloader.dataset)
        for _batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self._device), y.to(self._device)

            # Compute prediction and loss
            _pred = self.model(X)
            _loss = self._loss(_pred, y)

            # Backpropagation
            self._optimizer.zero_grad()
            _loss.backward()
            self._optimizer.step()
            self.n_samples += len(X)

            # print progress # TODO: this might be removed
            if _batch % 100 == 0:
                _loss, _current = _loss.item(), _batch * len(X)
                if self.print_progress:
                    print(f"loss: {_loss:>7f}  [{_current:>5d}/{_size:>5d}]")

    @torch.inference_mode()
    def _get_current_params(self):
        return [
            param.detach().numpy() for param in self.model.parameters()
        ], self.n_samples

    @torch.inference_mode()
    def _update_params(self, new_params):
        model_params = self.model.parameters()
        assert len(new_params) == len(list(model_params))

        # update all the parameters
        for old_param, new_param in zip(model_params, new_params):
            old_param.data = new_param.data
