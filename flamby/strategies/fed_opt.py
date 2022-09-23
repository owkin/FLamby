from typing import List

import numpy as np
import torch
from tqdm import tqdm

from flamby.strategies.utils import DataLoaderWithMemory, _Model


class FedOpt:
    """FedOpt Strategy class

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
        dp_target_epsilon: float = None,
        dp_target_delta: float = None,
        dp_max_grad_norm: float = None,
        seed=None,
        log: bool = False,
        log_period: int = 100,
        bits_counting_function: callable = None,
        tau: float = 1e-8,
        server_learning_rate: float = 1e-2,
        beta1=0.9,
        beta2=0.999,
        logdir="./runs",
        log_basename="fed_opt",
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
        dp_target_epsilon: float
            The target epsilon for (epsilon, delta)-differential
            private guarantee. Defaults to None.
        dp_target_delta: float
            The target delta for (epsilon, delta)-differential
             private guarantee. Defaults to None.
        dp_max_grad_norm: float
            The maximum L2 norm of per-sample gradients; used to
            enforce differential privacy. Defaults to None.
         seed: int
            Seed to use for differential privacy. Defaults to None
         log: bool, optional
             Whether or not to store logs in tensorboard. Defaults to False.
        log_period: int, optional
            If log is True then log the loss every log_period batch updates.
            Defauts to 100.
         bits_counting_function : callable, optional
             A function making sure exchanges respect the rules, this function
             can be obtained by decorating check_exchange_compliance in
             flamby.utils. Should have the signature List[Tensor] -> int.
             Defaults to None.
         tau: float, optional
             adaptivity hyperparameter for the Adam/Yogi optimizer. Defaults to 1e-8.
         server_learning_rate : float, optional
             The learning rate used by the server optimizer. Defaults to 1.
         beta1: float, optional
             between 0 and 1, momentum parameter. Defaults to 0.9.
         beta2: float, optional
             between 0 and 1, second moment parameter. Defaults to 0.999.
         logdir: str, optional
             The path where to store the logs. Defaults to ./runs.
         log_basename: str, optional
             The basename of the logs that are created. Defaults to fed_opt.
        """

        assert (
            optimizer_class == torch.optim.SGD
        ), "Only SGD for client optimizer with FedOpt"

        self.training_dataloaders_with_memory = [
            DataLoaderWithMemory(e) for e in training_dataloaders
        ]
        self.training_sizes = [len(e) for e in self.training_dataloaders_with_memory]
        self.total_number_of_samples = sum(self.training_sizes)

        self.dp_target_epsilon = dp_target_epsilon
        self.dp_target_delta = dp_target_delta
        self.dp_max_grad_norm = dp_max_grad_norm
        self._seed = seed

        self.log = log
        self.log_period = log_period
        self.log_basename = log_basename
        self.logdir = logdir

        self.models_list = [
            _Model(
                model=model,
                optimizer_class=optimizer_class,
                lr=learning_rate,
                train_dl=_train_dl,
                dp_target_epsilon=self.dp_target_epsilon,
                dp_target_delta=self.dp_target_delta,
                dp_max_grad_norm=self.dp_max_grad_norm,
                loss=loss,
                nrounds=nrounds,
                log=self.log,
                client_id=i,
                log_period=self.log_period,
                log_basename=self.log_basename,
                logdir=self.logdir,
                seed=self._seed,
            )
            for i, _train_dl in enumerate(training_dataloaders)
        ]
        self.nrounds = nrounds
        self.num_updates = num_updates

        self.num_clients = len(self.training_sizes)
        self.bits_counting_function = bits_counting_function
        self.tauarray = [
            np.ones_like(param) * tau
            for param in self.models_list[0]._get_current_params()
        ]  # adaptivity HP for Adam and Yogi
        self.server_learning_rate = server_learning_rate
        self.beta1 = beta1  # momentum parameter
        self.beta2 = beta2  # second moment parameter
        self.m = [
            np.zeros_like(param) for param in self.models_list[0]._get_current_params()
        ]  # momentum
        self.v = [
            np.zeros_like(param) for param in self.models_list[0]._get_current_params()
        ]  # second moment
        self.updates = [
            np.zeros_like(param) for param in self.models_list[0]._get_current_params()
        ]  # param update to be applied by the server optimizer

    def run(self):
        """This method performs self.nrounds rounds of averaging
        and returns the list of models.
        """
        for _ in tqdm(range(self.nrounds)):
            self.perform_round()
        return [m.model for m in self.models_list]

    def calc_aggregated_delta_weights(self):
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

        return aggregated_delta_weights


class FedAdam(FedOpt):
    """FedAdam Strategy class

    References
    ----------
    https://arxiv.org/abs/2003.00295


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
    dp_target_epsilon: float
        The target epsilon for (epsilon, delta)-differential
         private guarantee. Defaults to None.
    dp_target_delta: float
        The target delta for (epsilon, delta)-differential
        private guarantee. Defaults to None.
    dp_max_grad_norm: float
        The maximum L2 norm of per-sample gradients; used to enforce
        differential privacy. Defaults to None.
    seed: int
        Seed to use for differential privacy. Defaults to None
    log: bool, optional
        Whether or not to store logs in tensorboard. Defaults to False.
    log_period: int, optional
       If log is True then log the loss every log_period batch updates.
       Defauts to 100.
    bits_counting_function : callable, optional
        A function making sure exchanges respect the rules, this function
        can be obtained by decorating check_exchange_compliance in
        flamby.utils. Should have the signature List[Tensor] -> int.
        Defaults to None.
    tau: float, optional
        adaptivity hyperparameter for the Adam optimizer. Defaults to 1e-3.
    server_learning_rate : float, optional
        The learning rate used by the server optimizer. Defaults to 1e-2.
    beta1: float, optional
        between 0 and 1, momentum parameter. Defaults to 0.9.
    beta2: float, optional
        between 0 and 1, second moment parameter. Defaults to 0.999.
    logdir: str, optional
        The path where to store the logs. Defaults to ./runs.
    log_basename: str, optional
        The basename of the logs that are created. Defaults to fed_adam.
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
        seed: int = None,
        log: bool = False,
        log_period: int = 100,
        bits_counting_function: callable = None,
        tau: float = 1e-3,
        server_learning_rate: float = 1e-2,
        beta1=0.9,
        beta2=0.999,
        logdir="./runs",
        log_basename="fed_adam",
    ):
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
            seed,
            log,
            log_period,
            bits_counting_function,
            tau,
            server_learning_rate,
            beta1,
            beta2,
            log_basename=log_basename,
            logdir=logdir,
        )

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

        aggregated_delta_weights = self.calc_aggregated_delta_weights()

        # Update momentum and second moment, calculate parameter updates
        for param_idx in range(len(self.m)):
            self.m[param_idx] = (
                self.beta1 * self.m[param_idx]
                + (1 - self.beta1) * aggregated_delta_weights[param_idx]
            )
        for param_idx in range(len(self.v)):
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


class FedYogi(FedOpt):
    """FedYogi Strategy class

    References
    ----------
    https://arxiv.org/abs/2003.00295

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
    dp_target_epsilon: float
        The target epsilon for (epsilon, delta)-differential
        private guarantee. Defaults to None.
    dp_target_delta: float
        The target delta for (epsilon, delta)-differential
        private guarantee. Defaults to None.
    dp_max_grad_norm: float
        The maximum L2 norm of per-sample gradients;
        used to enforce differential privacy. Defaults to None.
    seed: int
        Seed to use for differential privacy. Defaults to None
    log: bool, optional
        Whether or not to store logs in tensorboard. Defaults to False.
    log_period: int, optional
       If log is True then log the loss every log_period batch updates.
       Defauts to 100.
    bits_counting_function : callable, optional
        A function making sure exchanges respect the rules, this function
        can be obtained by decorating check_exchange_compliance in
        flamby.utils. Should have the signature List[Tensor] -> int.
        Defaults to None.
    tau: float, optional
        adaptivity hyperparameter for the Adam optimizer. Defaults to 1e-3.
    server_learning_rate : float, optional
        The learning rate used by the server optimizer. Defaults to 1e-2.
    beta1: float, optional
        between 0 and 1, momentum parameter. Defaults to 0.9.
    beta2: float, optional
        between 0 and 1, second moment parameter. Defaults to 0.999.
    logdir: str, optional
        The path where to store the logs. Defaults to ./runs.
    log_basename: str, optional
        The basename of the logs that are created. Defaults to fed_yogi.
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
        seed=None,
        log: bool = False,
        log_period: int = 100,
        bits_counting_function: callable = None,
        tau: float = 1e-3,
        server_learning_rate: float = 1e-2,
        beta1: float = 0.9,
        beta2: float = 0.999,
        logdir: str = "./runs",
        log_basename: str = "fed_yogi",
    ):

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
            seed,
            log,
            log_period,
            bits_counting_function,
            tau,
            server_learning_rate,
            beta1,
            beta2,
            log_basename=log_basename,
            logdir=logdir,
        )

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
        aggregated_delta_weights = self.calc_aggregated_delta_weights()

        # Update momentum and second moment, calculate parameter updates
        for param_idx in range(len(self.m)):
            self.m[param_idx] = (
                self.beta1 * self.m[param_idx]
                + (1 - self.beta1) * aggregated_delta_weights[param_idx]
            )

        for param_idx in range(len(self.v)):
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

        for param_idx in range(len(self.updates)):
            self.updates[param_idx] = (
                self.server_learning_rate
                * self.m[param_idx]
                / (np.sqrt(self.v[param_idx]) + self.tauarray[param_idx])
            )

        # Update models
        for _model in self.models_list:
            _model._update_params(self.updates)


class FedAdagrad(FedOpt):
    """FedYogi Strategy class

    References
    ----------
    https://arxiv.org/abs/2003.00295

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
    dp_target_epsilon: float
        The target epsilon for (epsilon, delta)-differential
         private guarantee. Defaults to None.
    dp_target_delta: float
        The target delta for (epsilon, delta)-differential
        private guarantee. Defaults to None.
    dp_max_grad_norm: float
        The maximum L2 norm of per-sample gradients;
        used to enforce differential privacy. Defaults to None.
    seed: int
        Seed to use for differential privacy. Defaults to None
    log: bool, optional
         Whether or not to store logs in tensorboard. Defaults to False.
    log_period: int, optional
        If log is True then log the loss every log_period batch updates.
        Defauts to 100.
    bits_counting_function : callable, optional
         A function making sure exchanges respect the rules, this function
         can be obtained by decorating check_exchange_compliance in
         flamby.utils. Should have the signature List[Tensor] -> int.
         Defaults to None.
    tau: float, optional
         adaptivity hyperparameter for the Adam optimizer. Defaults to 1e-3.
    server_learning_rate : float, optional
         The learning rate used by the server optimizer. Defaults to 1e-2.
    beta1: float, optional
         between 0 and 1, momentum parameter. Defaults to 0.9.
    beta2: float, optional
         between 0 and 1, second moment parameter. Defaults to 0.999.
    logdir: str, optional
         The path where to store the logs. Defaults to ./runs.
    log_basename: str, optional
         The basename of the logs that are created. Defaults to fed_adagrad.
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
        seed: int = None,
        log: bool = False,
        log_period: int = 100,
        bits_counting_function: callable = None,
        tau: float = 1e-3,
        server_learning_rate: float = 1e-2,
        beta1: float = 0.9,
        beta2: float = 0.999,
        logdir: str = "./runs",
        log_basename: str = "fed_adagrad",
    ):

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
            seed,
            log,
            log_period,
            bits_counting_function,
            tau,
            server_learning_rate,
            beta1,
            beta2,
            logdir=logdir,
            log_basename=log_basename,
        )

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
        aggregated_delta_weights = self.calc_aggregated_delta_weights()

        # Update momentum and second moment, calculate parameter updates
        for param_idx in range(len(self.m)):
            self.m[param_idx] = (
                self.beta1 * self.m[param_idx]
                + (1 - self.beta1) * aggregated_delta_weights[param_idx]
            )

        for param_idx in range(len(self.v)):
            self.v[param_idx] = (
                self.v[param_idx]
                + aggregated_delta_weights[param_idx]
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
