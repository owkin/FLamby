import copy

import torch


class DataLoaderWithMemory:
    def __init__(self, dataloader):
        self._dataloader = dataloader

        self._iterator = iter(self._dataloader)

    def _reset_iterator(self):
        self._iterator = iter(self._dataloader)

    def __len__(self):
        return len(self._dataloader.dataset)

    def get_samples(self):
        try:
            X, y = self._iterator.next()
        except StopIteration:
            self._reset_iterator()
            X, y = self._iterator.next()
        return X, y


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
        self.num_batches_seen = 0

    def _local_train(self, dataloader_with_memory, num_updates):
        # Local train
        _size = len(dataloader_with_memory)
        self.model = self.model.train()
        for _batch in range(num_updates):
            X, y = dataloader_with_memory.get_samples()
            X, y = X.to(self._device), y.to(self._device)
            # Compute prediction and loss
            _pred = self.model(X)
            _loss = self._loss(_pred, y)

            # Backpropagation
            self._optimizer.zero_grad()
            _loss.backward()
            self._optimizer.step()
            self.num_batches_seen += 1

            # print progress # TODO: this might be removed
            if _batch % 100 == 0:
                _loss, _current_epoch = _loss.item(), _size // (
                    self.num_batches_seen * X.shape[0]
                )
                if self.print_progress:
                    print(
                        f"loss: {_loss:>7f} after {self.num_batches_seen:>5d} "
                        f"batches of data amounting to {_current_epoch:>5d}"
                        "epochs."
                    )

    @torch.inference_mode()
    def _get_current_params(self):
        return [param.detach().numpy() for param in self.model.parameters()]

    @torch.inference_mode()
    def _update_params(self, new_params):
        model_params = self.model.parameters()
        assert len(new_params) == len(list(model_params))

        # update all the parameters
        for old_param, new_param in zip(model_params, new_params):
            old_param.data += new_param
