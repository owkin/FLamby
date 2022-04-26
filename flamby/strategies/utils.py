import copy

import torch


class DataLoaderWithMemory:
    """This class allows to iterate the dataloader infinitely batch by batch.
    When there are no more batches the iterator is reset silently.
    This class allows to keep the memory of the state of the iterator hence its
    name.
    """

    def __init__(self, dataloader):
        """This initialization takes a dataloader and creates an iterator object
        from it.

        Parameters
        ----------
        dataloader : torch.utils.data.dataloader
            A dataloader object built from one of the datasets of this repository.
        """
        self._dataloader = dataloader

        self._iterator = iter(self._dataloader)

    def _reset_iterator(self):
        self._iterator = iter(self._dataloader)

    def __len__(self):
        return len(self._dataloader.dataset)

    def get_samples(self):
        """This method generates the next batch from the iterator or resets it
        if needed. It can be called an infinite amount of times.

        Returns
        -------
        tuple
            a batch from the iterator
        """
        try:
            X, y = self._iterator.next()
        except StopIteration:
            self._reset_iterator()
            X, y = self._iterator.next()
        return X, y


class _Model:
    """This is a helper class allowing to train a copy of a given model for
    num_updates steps by instantiating the user-provided optimizer.
    This class posesses method to retrieve current parameters set in np.ndarrays
    and to update the weights with a numpy list of the same size as the
    parameters of the model.
    """

    def __init__(self, model, optimizer_class, lr, loss):
        """_summary_

        Parameters
        ----------
        model : torch.nn.Module
            _description_
        optimizer_class : torch.optim
            A torch optimizer class that will be instantiated by calling:
            optimizer_class(self.model.parameters(), lr)
        lr : float
            The learning rate to use with th optimizer class.
        loss : torch.nn.modules.loss._loss
            an instantiated torch loss.
        """
        self.model = copy.deepcopy(model)
        self._optimizer = optimizer_class(self.model.parameters(), lr)
        self._loss = copy.deepcopy(loss)
        init_seed = 42
        torch.manual_seed(init_seed)
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self._device)
        self.print_progress = True
        self.num_batches_seen = 0

    def _local_train(self, dataloader_with_memory, num_updates):
        """This method trains the model using the dataloader_with_memory given
        for num_updates steps.

        Parameters
        ----------
        dataloader_with_memory : dataloaderwithmemory
            A dataloader that can be called infinitely using its get_samples()
            method.
        num_updates : int
            The number of batches to train on.
        """
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
            _loss.backward()
            self._optimizer.step()
            self._optimizer.zero_grad()
            self.num_batches_seen += 1

            # print progress # TODO: this might be removed
            if _batch % 100 == 0:
                _loss, _current_epoch = _loss.item(), self.num_batches_seen // (
                    _size // X.shape[0]
                )
                if self.print_progress:
                    print(
                        f"loss: {_loss:>7f} after {self.num_batches_seen:>5d}"
                        f" batches of data amounting to {_current_epoch:>5d}"
                        " epochs."
                    )

    @torch.inference_mode()
    def _get_current_params(self):
        """Returns the current weights of the pytorch model.

        Returns
        -------
        list[np.ndarray]
            A list of numpy versions of the weights.
        """
        return [param.cpu().detach().numpy() for param in self.model.parameters()]

    @torch.inference_mode()
    def _update_params(self, new_params):
        """Update in place the weights of the pytorch model by adding the
        new_params llist of the same size to it.
        """
        # update all the parameters
        for old_param, new_param in zip(self.model.parameters(), new_params):
            old_param.data += torch.from_numpy(new_param).to(old_param.device)
