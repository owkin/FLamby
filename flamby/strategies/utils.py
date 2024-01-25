import copy
import os
import time
from datetime import datetime

import numpy as np
import torch
from opacus import PrivacyEngine
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
            X, y = next(self._iterator)
        except StopIteration:
            self._reset_iterator()
            X, y = next(self._iterator)
        return X, y


class _Model:
    """This is a helper class allowing to train a copy of a given model for
    num_updates steps by instantiating the user-provided optimizer.
    This class posesses method to retrieve current parameters set in np.ndarrays
    and to update the weights with a numpy list of the same size as the
    parameters of the model.
    """

    def __init__(
        self,
        model,
        train_dl,
        optimizer_class,
        lr,
        loss,
        nrounds,
        client_id=0,
        dp_target_epsilon=None,
        dp_target_delta=None,
        dp_max_grad_norm=None,
        log=False,
        log_period=100,
        log_basename="local_model",
        logdir="./runs",
        seed=None,
    ):
        """_summary_

        Parameters
        ----------
        model : torch.nn.Module
            _description_
        train_dl : torch.utils.data.DataLoader
            _description_
        optimizer_class : torch.optim
            A torch optimizer class that will be instantiated by calling:
            optimizer_class(self.model.parameters(), lr)
        lr : float
            The learning rate to use with th optimizer class.
        loss : torch.nn.modules.loss._loss
            an instantiated torch loss.
        nrounds: int
            The number of communication rounds to do.
        log: bool
            Whether or not to log quantities with tensorboard. Defaults to False.
        client_id: int
            The id of the client for logging purposes. Default to 0.
        dp_target_epsilon: float
            The target epsilon for (epsilon, delta)-differential
             private guarantee. Defaults to None.
        dp_target_delta: float
            The target delta for (epsilon, delta)-differential
             private guarantee. Defaults to None.
        dp_max_grad_norm: float
            The maximum L2 norm of per-sample gradients;
             used to enforce differential privacy. Defaults to None.
        log_period: int
            The period at which to log quantities. Defaults to 100.
        log_basename: str
            The basename of the created log file if log=True. Defaults to fed_avg.
        logdir: str
            Where to create the log file. Defaults to ./runs.
        seed: int
            Seed provided to torch.Generator. Defaults to None.
        """
        self.model = copy.deepcopy(model)

        self._train_dl = train_dl
        self._optimizer = optimizer_class(self.model.parameters(), lr)
        self._loss = copy.deepcopy(loss)
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self._device)
        self.num_batches_seen = 0
        self.log = log
        self.log_period = log_period
        self.client_id = client_id

        self.dp_target_epsilon = dp_target_epsilon
        self.dp_target_delta = dp_target_delta
        self.dp_max_grad_norm = dp_max_grad_norm

        if self.log:
            os.makedirs(logdir, exist_ok=True)
            date_now = str(datetime.now())
            self.writer = SummaryWriter(
                log_dir=os.path.join(logdir, f"{log_basename}-{date_now}")
            )

        self._apply_dp = (
            (self.dp_target_epsilon is not None)
            and (self.dp_max_grad_norm is not None)
            and (self.dp_target_delta is not None)
        )

        if self._apply_dp:
            seed = seed if seed is not None else int(time.time())

            privacy_engine = PrivacyEngine()

            (
                self.model,
                self._optimizer,
                self._train_dl,
            ) = privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=self._optimizer,
                data_loader=self._train_dl,
                epochs=nrounds,
                target_epsilon=dp_target_epsilon,
                target_delta=dp_target_delta,
                max_grad_norm=dp_max_grad_norm,
                noise_generator=torch.Generator(self._device).manual_seed(seed),
            )

        self.current_epoch = 0
        self.batch_size = None
        self.num_batches_per_epoch = None

    def _local_train(self, dataloader_with_memory, num_updates):
        """This method trains the model using the dataloader_with_memory given
        for num_updates steps.

        Parameters
        ----------
        dataloader_with_memory : DataLoaderWithMemory
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
            if _batch == 0:
                # Initialize the batch-size using the first batch to avoid
                # edge cases with drop_last=False
                _batch_size = X.shape[0]
                _num_batches_per_epoch = (_size // _batch_size) + int(
                    (_size % _batch_size) != 0
                )
            # Compute prediction and loss
            _pred = self.model(X)
            _loss = self._loss(_pred, y)

            # Backpropagation
            _loss.backward()
            self._optimizer.step()
            self._optimizer.zero_grad()
            self.num_batches_seen += 1
            _loss, _current_epoch = (
                _loss.item(),
                self.num_batches_seen // _num_batches_per_epoch,
            )

            if self.log:
                if _batch % self.log_period == 0:
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

                if _current_epoch > self.current_epoch:
                    # At each epoch we look at the histograms of all the
                    # network's parameters
                    for name, p in self.model.named_parameters():
                        self.writer.add_histogram(
                            f"client{self.client_id}/{name}", p, _current_epoch
                        )

            self.current_epoch = _current_epoch

    def _prox_local_train(self, dataloader_with_memory, num_updates, mu):
        """This method trains the model using the dataloader_with_memory given
        for num_updates steps.

        Parameters
        ----------
        dataloader_with_memory : dataloaderwithmemory
            A dataloader that can be called infinitely using its get_samples()
            method.
        num_updates : int
            The number of batches to train on.
        mu: float
            The mu parameter involved in the proximal term.
        """
        # Model used for FedProx for regularization at every optimization round
        model_initial = copy.deepcopy(self.model)

        # Local train
        _size = len(dataloader_with_memory)
        self.model = self.model.train()
        for idx, _batch in enumerate(range(num_updates)):
            X, y = dataloader_with_memory.get_samples()
            X, y = X.to(self._device), y.to(self._device)
            if idx == 0:
                # Initialize the batch-size using the first batch to avoid
                # edge cases with drop_last=False
                _batch_size = X.shape[0]
                _num_batches_per_epoch = (_size // _batch_size) + int(
                    (_size % _batch_size) != 0
                )
            # Compute prediction and loss
            _pred = self.model(X)
            _prox_loss = self._loss(_pred, y)
            # We preserve the true loss before adding the proximal term
            # and doing the backward step on the sum.
            _loss = _prox_loss.detach()

            if mu > 0.0:
                squared_norm = compute_model_diff_squared_norm(
                    model_initial, self.model
                )
                _prox_loss += mu / 2 * squared_norm

            # Backpropagation
            _prox_loss.backward()
            self._optimizer.step()
            self._optimizer.zero_grad()
            self.num_batches_seen += 1
            _loss, _current_epoch = (
                _loss.item(),
                self.num_batches_seen // _num_batches_per_epoch,
            )

            if self.log:
                if _batch % self.log_period == 0:
                    if _current_epoch > self.current_epoch:
                        # At each epoch we look at the histograms of all the
                        # network's parameters
                        for name, p in self.model.named_parameters():
                            self.writer.add_histogram(
                                f"client{self.client_id}/{name}", p, _current_epoch
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

    def _local_train_with_correction(
        self, dataloader_with_memory, num_updates, correction_state
    ):
        """This method trains the model using the dataloader_with_memory given
        for num_updates steps while applying a correction during every update.

        Parameters
        ----------
        dataloader_with_memory : dataloaderwithmemory
            A dataloader that can be called infinitely using its get_samples()
            method.
        num_updates : int
            The number of batches to train on.
        correction_state: List
            Correction to be applied to the model state during every local update.
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
                _batch_size = X.shape[0]
                _num_batches_per_epoch = (_size // _batch_size) + int(
                    (_size % _batch_size) != 0
                )
            # We will implement correction by modifying loss as
            # corrected_loss = loss - correction @ model_params.
            # Then, we have corrected gradient = gradient - correction.

            # Compute prediction and loss
            _pred = self.model(X)
            _corrected_loss = self._loss(_pred, y)
            # We preserve the true loss before adding the correction term
            # and doing the backward step on the sum.
            _loss = _corrected_loss.detach()
            _corrected_loss -= compute_dot_product(self.model, correction_state)

            # Backpropagation
            _corrected_loss.backward()
            self._optimizer.step()
            self._optimizer.zero_grad()
            self.num_batches_seen += 1
            _loss, _current_epoch = (
                _loss.item(),
                self.num_batches_seen // _num_batches_per_epoch,
            )

            if self.log:
                if _batch % self.log_period == 0:
                    if _current_epoch > self.current_epoch:
                        # At each epoch we look at the histograms of all the
                        # network's parameters
                        for name, p in self.model.named_parameters():
                            self.writer.add_histogram(
                                f"client{self.client_id}/{name}", p, _current_epoch
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

    @torch.no_grad()
    def _get_current_params(self):
        """Returns the current weights of the pytorch model.

        Returns
        -------
        list[np.ndarray]
            A list of numpy versions of the weights.
        """
        return [
            param.cpu().detach().clone().numpy() for param in self.model.parameters()
        ]

    @torch.no_grad()
    def _update_params(self, new_params):
        """Update in place the weights of the pytorch model by adding the
        new_params list of the same size to it.

        """
        # update all the parameters
        for old_param, new_param in zip(self.model.parameters(), new_params):
            old_param.data += torch.from_numpy(new_param).to(old_param.device)


def compute_model_diff_squared_norm(model1: torch.nn.Module, model2: torch.nn.Module):
    """Compute the squared norm of the difference between two models.

    Parameters
    ----------
    model1 : torch.nn.Module
    model2 : torch.nn.Module
    """
    tensor1 = list(model1.parameters())
    tensor2 = list(model2.parameters())
    norm = sum([torch.sum((tensor1[i] - tensor2[i]) ** 2) for i in range(len(tensor1))])

    return norm


def compute_dot_product(model: torch.nn.Module, params):
    """Compute the dot prodcut between model and input parameters.

    Parameters
    ----------
    model : torch.nn.Module
    params : List containing model parameters
    """
    model_p = list(model.parameters())
    device = model_p[0].device
    dot_prod = sum([torch.sum(m * p.to(device)) for m, p in zip(model_p, params)])
    return dot_prod


def check_exchange_compliance(tensors_list, max_bytes, units="bytes"):
    """
    Check that for each round the quantities exchanged are below the dataset
    specific limit.
    Parameters
    ----------
    tensors_list: List[Union[torch.Tensor, np.ndarray]]
        The list of quantities sent by the client.
    max_bytes: int
        The number of bytes max to exchange per round per client.
    units: str
        The units in which to return the result. Default to bytes.$
    Returns
    -------
    int
        Returns the number of bits exchanged in the provided unit or raises an
        error if it went above the limit.
    """
    assert units in ["bytes", "bits", "megabytes", "gigabytes"]
    assert isinstance(tensors_list, list), "You should provide a list of tensors."
    assert all([
        (isinstance(t, np.ndarray) or isinstance(t, torch.Tensor)) for t in tensors_list
    ])
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
    else:
        raise NotImplementedError(f"{units} is not a possible unit")

    return res
