
from substrafl.strategies import FedAvg
import torch
from substrafl.model_loading import load_algo
from substrafl.model_loading import download_algo_files
import matplotlib.pyplot as plt
import pandas as pd
from substrafl.experiment import execute_experiment
from substrafl.evaluation_strategy import EvaluationStrategy
from substrafl.nodes import TestDataNode
from substrafl.nodes import AggregationNode
from substrafl.nodes import TrainDataNode
from substrafl.algorithms.pytorch import TorchFedAvgAlgo
from substrafl.index_generator import NpIndexGenerator
from substrafl.remote.register import add_metric
from substrafl.dependency import Dependency
from torch.utils import data
import numpy as np
from substra.sdk.schemas import Permissions
from substra.sdk.schemas import DataSampleSpec
from substra.sdk.schemas import DatasetSpec
import pathlib
import substra
from substra import Client
from flamby.datasets import fed_heart_disease


MODE = substra.BackendType.LOCAL_SUBPROCESS

# Create the substra clients
data_provider_clients = [Client(backend_type=MODE)
                         for _ in range(fed_heart_disease.NUM_CLIENTS)]

data_provider_clients = {client.organization_info(
).organization_id: client for client in data_provider_clients}

algo_provider_client = Client(backend_type=MODE)

# Store their IDs
DATA_PROVIDER_ORGS_ID = list(data_provider_clients.keys())

# The org id on which your computation tasks are registered
ALGO_ORG_ID = algo_provider_client.organization_info().organization_id


assets_directory = pathlib.Path.cwd() / "assets"

print(assets_directory)
empty_path = assets_directory / "empty_datasamples"

permissions_dataset = Permissions(public=False, authorized_ids=[ALGO_ORG_ID])

train_dataset_keys = {}
test_dataset_keys = {}

train_datasample_keys = {}
test_datasample_keys = {}


for ind, org_id in enumerate(DATA_PROVIDER_ORGS_ID):
    client = data_provider_clients[org_id]

    # DatasetSpec is the specification of a dataset. It makes sure every field
    # is well defined, and that our dataset is ready to be registered.
    # The real dataset object is created in the add_dataset method.

    dataset = DatasetSpec(
        name="FLamby",
        type="torchDataset",
        data_opener=assets_directory / "dataset" / f"opener_train_{org_id}.py",
        description=assets_directory / "dataset" / "description.md",
        permissions=permissions_dataset,
        logs_permission=permissions_dataset,
    )

    # Add the dataset to the client to provide access to the opener in each organization.
    train_dataset_key = client.add_dataset(dataset)
    assert train_dataset_key, "Missing data manager key"

    train_dataset_keys[org_id] = train_dataset_key

    # Add the training data on each organization.
    data_sample = DataSampleSpec(
        data_manager_keys=[train_dataset_key],
        test_only=False,
        path=empty_path,
    )
    train_datasample_key = client.add_data_sample(
        data_sample,
        local=True,
    )

    train_datasample_keys[org_id] = train_datasample_key

    # Add the testing data.

    test_dataset_key = client.add_dataset(
        DatasetSpec(
            name="FLamby",
            type="torchDataset",
            data_opener=assets_directory / "dataset" / f"opener_test_{org_id}.py",
            description=assets_directory / "dataset" / "description.md",
            permissions=permissions_dataset,
            logs_permission=permissions_dataset,
        )
    )
    assert test_dataset_key, "Missing data manager key"
    test_dataset_keys[org_id] = test_dataset_key

    data_sample = DataSampleSpec(
        data_manager_keys=[test_dataset_key],
        test_only=True,
        path=empty_path,
    )
    test_datasample_key = client.add_data_sample(
        data_sample,
        local=True,
    )

    test_datasample_keys[org_id] = test_datasample_key


# def fed_heart_disease_metric(datasamples, predictions_path):

#     config = datasamples

#     dataset = fed_heart_disease.FedHeartDisease(**config)
#     dataloader = data.DataLoader(dataset, batch_size=len(dataset))

#     y_true = next(iter(dataloader))[1]
#     y_pred = np.load(predictions_path)

#     return float(fed_heart_disease.metric(y_true, y_pred))


# # The Dependency object is instantiated in order to install the right libraries in
# # the Python environment of each organization.
# # The local dependencies are local packages to be installed using the command `pip install -e .`.
# # Flamby is a local dependency. We put as argument the path to the `setup.py` file.
# metric_deps = Dependency(pypi_dependencies=["torch==1.11.0", "numpy==1.23.1"],
#                          # Flamby dependency
#                          local_dependencies=[pathlib.Path.cwd().parent.parent],
#                          )
# permissions_metric = Permissions(
#     public=False, authorized_ids=DATA_PROVIDER_ORGS_ID + [ALGO_ORG_ID])

# metric_key = add_metric(
#     client=algo_provider_client,
#     metric_function=fed_heart_disease_metric,
#     permissions=permissions_metric,
#     dependencies=metric_deps,
# )


# NUM_UPDATES = 16
# SEED = 42

# index_generator = NpIndexGenerator(
#     batch_size=fed_heart_disease.BATCH_SIZE,
#     num_updates=NUM_UPDATES,
# )


# class TorchDataset(fed_heart_disease.FedHeartDisease):

#     def __init__(self, datasamples, is_inference):
#         config = datasamples
#         super().__init__(**config)


# model = fed_heart_disease.Baseline()


# class MyAlgo(TorchFedAvgAlgo):
#     def __init__(self):
#         super().__init__(
#             model=model,
#             criterion=fed_heart_disease.BaselineLoss(),
#             optimizer=fed_heart_disease.Optimizer(
#                 model.parameters(), lr=fed_heart_disease.LR),
#             index_generator=index_generator,
#             dataset=TorchDataset,
#             seed=SEED,
#         )

#     def _local_predict(self, predict_dataset: torch.utils.data.Dataset, predictions_path):

#         batch_size = self._index_generator.batch_size
#         predict_loader = torch.utils.data.DataLoader(
#             predict_dataset, batch_size=batch_size)

#         self._model.eval()

#         # The output dimension of the model is of size (1,)
#         predictions = torch.zeros((len(predict_dataset), 1))

#         with torch.inference_mode():
#             for i, (x, _) in enumerate(predict_loader):
#                 x = x.to(self._device)
#                 predictions[i * batch_size: (i+1) * batch_size] = self._model(x)

#         predictions = predictions.cpu().detach()
#         self._save_predictions(predictions, predictions_path)


# strategy = FedAvg()


# aggregation_node = AggregationNode(ALGO_ORG_ID)

# train_data_nodes = list()

# for org_id in DATA_PROVIDER_ORGS_ID:

#     # Create the Train Data Node (or training task) and save it in a list
#     train_data_node = TrainDataNode(
#         organization_id=org_id,
#         data_manager_key=train_dataset_keys[org_id],
#         data_sample_keys=[train_datasample_keys[org_id]],
#     )
#     train_data_nodes.append(train_data_node)


# test_data_nodes = list()

# for org_id in DATA_PROVIDER_ORGS_ID:

#     # Create the Test Data Node (or testing task) and save it in a list
#     test_data_node = TestDataNode(
#         organization_id=org_id,
#         data_manager_key=test_dataset_keys[org_id],
#         test_data_sample_keys=[test_datasample_keys[org_id]],
#         metric_keys=[metric_key],
#     )
#     test_data_nodes.append(test_data_node)

# # Test at the end of every round
# my_eval_strategy = EvaluationStrategy(test_data_nodes=test_data_nodes, rounds=1)


# # Number of time to apply the compute plan.
# NUM_ROUNDS = 3

# # The Dependency object is instantiated in order to install the right libraries in
# # the Python environment of each organization.
# # The local dependencies are local packages to be installed using the command `pip install -e .`.
# # Flamby is a local dependency. We put as argument the path to the `setup.py` file.
# algo_deps = Dependency(pypi_dependencies=["torch==1.11.0"], local_dependencies=[
#     pathlib.Path.cwd().parent.parent])

# compute_plan = execute_experiment(
#     client=algo_provider_client,
#     algo=MyAlgo(),
#     strategy=strategy,
#     train_data_nodes=train_data_nodes,
#     evaluation_strategy=my_eval_strategy,
#     aggregation_node=aggregation_node,
#     num_rounds=NUM_ROUNDS,
#     experiment_folder=str(pathlib.Path.cwd() / "experiment_summaries"),
#     dependencies=algo_deps,
# )


# plt.title("Performance evolution on each center of the baseline on Fed-TCGA-BRCA with Federated Averaging training")
# plt.xlabel("Rounds")
# plt.ylabel("Metric")

# performance_df = pd.DataFrame(client.get_performances(compute_plan.key).dict())

# for i, id in enumerate(DATA_PROVIDER_ORGS_ID):
#     df = performance_df.query(f"worker == '{id}'")
#     plt.plot(df["round_idx"], df["performance"], label=f"Client {i} ({id})")

# plt.legend(loc=(1.1, 0.3), title="Test set")
# plt.show()


# client_to_dowload_from = DATA_PROVIDER_ORGS_ID[0]
# round_idx = None

# folder = str(pathlib.Path.cwd() / "experiment_summaries" /
#              compute_plan.key / ALGO_ORG_ID / (round_idx or "last"))

# download_algo_files(
#     client=data_provider_clients[client_to_dowload_from],
#     compute_plan_key=compute_plan.key,
#     round_idx=round_idx,
#     dest_folder=folder,
# )

# model = load_algo(input_folder=folder)._model

# print(model)
# print([p for p in model.parameters()])
