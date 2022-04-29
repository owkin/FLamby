import copy
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


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

    def __init__(
        self, model, optimizer_class, lr, loss, client_id=0, log=False, log_period=100
    ):
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
        log: bool
            Whether or not to log quantities with tensorboard. Defaults to False.
        client_id: int
            The id of the client for logging purposes. Default to 0.
        log_period: int
            The period at which to log quantities. Defaults to 100.
        """
        self.model = copy.deepcopy(model)
        self._optimizer = optimizer_class(self.model.parameters(), lr)
        self._loss = copy.deepcopy(loss)
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self._device)
        self.num_batches_seen = 0
        self.log = log
        self.log_period = log_period
        self.client_id = client_id
        if self.log:
            date_now = str(datetime.now())
            self.writer = SummaryWriter(log_dir=f"./runs/fed_avg-{date_now}")
        self.current_epoch = 0
        self.batch_size = None

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
        for idx, _batch in enumerate(range(num_updates)):
            X, y = dataloader_with_memory.get_samples()
            X, y = X.to(self._device), y.to(self._device)
            if idx == 0:
                # Initialize the batch-size using the first batch to avoid
                # edge cases with drop_last=False
                self.batch_size = X.shape[0]
            # Compute prediction and loss
            _pred = self.model(X)
            _loss = self._loss(_pred, y)

            # Backpropagation
            _loss.backward()
            self._optimizer.step()
            self._optimizer.zero_grad()
            self.num_batches_seen += 1

            if self.log:
                if _batch % self.log_period == 0:
                    _loss, _current_epoch = _loss.item(), self.num_batches_seen // (
                        _size // self.batch_size
                    )
                    if _current_epoch > self.current_epoch:
                        # At each epoch we look at the histograms of all the
                        # network's parameters
                        for name, p in self.model.named_parameters():
                            self.writer.add_histogram(
                                f"client{self.client_id}/name", p, _current_epoch
                            )

                    print(
                        f"loss: {_loss:>7f} after {self.num_batches_seen:>5d}"
                        f" batches of data amounting to {_current_epoch:>5d}"
                        " epochs."
                    )
                    self.writer.add_scalar(
                        f"client{self.client_id}/train/Loss",
                        _loss,
                        self.num_batches_seen,
                    )
            self.current_epoch = _current_epoch

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


def check_exchange_compliance(tensors_list, max_bytes, units="bytes"):
    """
    Check that for each round the quantities exchanged are below the dataset
    specific limit.
    Parameters
    ----------
    tensors_list: List[Union[torch.Tensor, np.ndarray]]
        The list of quantities sent by the client.
    max_bytes: int
        The number of bytes max to exchange pper round per client.
    units: str
        The units in which to return the result. Default to bytes.$
    Returns
    -------
    int
        Returns the number of bits exchanged in the pprovided unit or raises an
        error if it went above the limit.
    """
    assert units in ["bytes", "bits", "megabytes", "gigabytes"]
    assert isinstance(tensors_list, list), "You should provide a list of tensors."
    assert all(
        [
            (isinstance(t, np.ndarray) or isinstance(t, torch.Tensor))
            for t in tensors_list
        ]
    )
    bytes_count = 0
    for t in tensors_list:
        if isinstance(t, np.ndarray):
            bytes_count += t.nbytes
        else:
            bytes_count += t.shape.numel() * torch.finfo(t.dtype).bits // 8
        if bytes_count > max_bytes:
            raise ValueError(
                f"You cannot send more than {max_bytes} bytes, this "
                f"round. You tried sending more than {bytes_count} bytes already"
            )
    if units == "bytes":
        res = bytes_count
    elif units == "bits":
        res = bytes_count * 8
    elif units == "megabytes":
        res = 1e-6 * bytes_count
    elif units == "gigabytes":
        res = 1e-9 * bytes_count

    return res
