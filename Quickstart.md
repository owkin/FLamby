## Quickstart

We will start with the Fed-TCGA-BRCA dataset because it requires no downloading or preprocessing and is very lightweight.
Therefore it provides a good introduction to navigating Flamby.
This tutorial does not require the use of a GPU.

### Dataset example

We do provide users with two dataset abstractions: RawDataset and FedDataset.
The recommended one is FedDataset: it is compatible with the rest of the repository's code.
This class allows to instantiate either the single-centric version of the dataset using the argument `pooled = True`, or the different local datasets belonging to each client by providing the client index in the arguments (e.g. `center = 2, pooled = False`).
The arguments `train = True` and `train = False` allow to instantiate train or test sets (in both pooled or local cases).
We also provide RawDataset objects which are less easy to work with but that should provide all metadata required for power users that find the abstraction in FLamby hard to use for their use-cases.

To instantiate the raw TCGA-BRCA or the Fed-TCGA-BRCA dataset, follow those examples:
```python

from flamby.datasets.fed_tcga_brca import TcgaBrcaRaw, FedTcgaBrca

# Raw dataset
mydataset = TcgaBrcaRaw()

# Pooled test dataset
mydataset = FedTcgaBrca(train=False, pooled=True)

# Center 2 train dataset
mydataset = FedTcgaBrca(center=2, train=True, pooled=False)

```

### Local training example

Below is an example of how to train the chosen baseline model with default settings on one local train set and evaluate it on all the local test sets:
```python

import torch
from flamby.utils import evaluate_model_on_tests

# 2 lines of code to change to switch to another dataset
from flamby.datasets.fed_tcga_brca import (
    BATCH_SIZE,
    LR,
    NUM_EPOCHS_POOLED,
    Baseline,
    BaselineLoss,
    metric,
    NUM_CLIENTS,
    Optimizer,
)
from flamby.datasets.fed_tcga_brca import FedTcgaBrca as FedDataset


train_dataset = FedDataset(center=0, train=True, pooled=False)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
lossfunc = BaselineLoss()
model = Baseline()
optimizer = Optimizer(model.parameters(), lr=LR)

# Traditional pytorch training loop
for epoch in range(0, NUM_EPOCHS_POOLED):
    for idx, (X, y) in enumerate(train_dataloader):
        optimizer.zero_grad()
        outputs = model(X)
        loss = lossfunc(outputs, y)
        loss.backward()
        optimizer.step()

# Evaluation
test_dataloaders = [
            torch.utils.data.DataLoader(
                FedDataset(center=i, train=False, pooled=False),
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0,
            )
            for i in range(NUM_CLIENTS)
        ]
dict_cindex = evaluate_model_on_tests(model, test_dataloaders, metric)
print(dict_cindex)

```

### Federated Learning example

See below an example of how to train a baseline model on the Fed-TCGA-BRCA dataset in a federated way using the FedAvg strategy and evaluate it on all the pooled test set:
```python

import torch
from flamby.utils import evaluate_model_on_tests

# 2 lines of code to change to switch to another dataset
from flamby.datasets.fed_tcga_brca import (
    BATCH_SIZE,
    LR,
    NUM_EPOCHS_POOLED,
    Baseline,
    BaselineLoss,
    metric,
    NUM_CLIENTS,
    get_nb_max_rounds
)
from flamby.datasets.fed_tcga_brca import FedTcgaBrca as FedDataset

# 1st line of code to change to switch to another strategy
from flamby.strategies.fed_avg import FedAvg as strat

# We loop on all the clients of the distributed dataset and instantiate associated dataloaders
train_dataloaders = [
            torch.utils.data.DataLoader(
                FedDataset(center = i, train = True, pooled = False),
                batch_size = BATCH_SIZE,
                shuffle = True,
                num_workers = 0
            )
            for i in range(NUM_CLIENTS)
        ]
lossfunc = BaselineLoss()
m = Baseline()

# Federated Learning loop
# 2nd line of code to change to switch to another strategy (feed the FL strategy the right HPs)
args = {
            "training_dataloaders": train_dataloaders,
            "model": m,
            "loss": lossfunc,
            "optimizer_class": torch.optim.SGD,
            "learning_rate": LR / 10.0,
            "num_updates": 100,
            "nrounds": get_nb_max_rounds(100),
        }
s = strat(**args)
m = s.run()[0]

# Evaluation
test_dataloaders = [
            torch.utils.data.DataLoader(
                FedDataset(train = False, pooled = True),
                batch_size = BATCH_SIZE,
                shuffle = False,
                num_workers = 0,
            )
        ]
dict_cindex = evaluate_model_on_tests(m, test_dataloaders, metric)
print(dict_cindex)

```

### Downloading a dataset

We will follow up on how to download a dataset that is not hosted in this repository.
We will use Fed-Heart-Disease as the download process is simple and it requires no preprocessing.
Please run:

```
cd flamby/datasets/fed_heart_disease/dataset_creation_scripts
python download.py --output-folder ./heart_disease_dataset
```

### Training and evaluation in a pooled setting

To train and evaluate a model for the pooled Heart Disease dataset, run:
```
cd flamby/datasets/fed_heart_disease
python benchmark.py --num-workers-torch 0
```

### Training and evaluation in a federated learning setting

In order to train and evaluate models trained on the pooled dataset, on all local datasets as well as in a federated way for all FL strategies, you can run the following command.
The config file is present in the repository and holds all necesssary HPs for FL strategies. The results are stored in the csv file given in the command.

```
cd flamby/benchmarks
python fed_benchmark.py --config-file-path ../heart_disease_config.json --results-file-path ./test_res.csv --seed 0
```

In order to train and evaluate a model with a specific FL strategy and new hyperparameters one can run:

```
python fed_benchmark.py --strategy FedProx --mu 0.001 --learning_rate 3. --config-file-path ../heart_disease_config.json --results-file-path ./test_res.csv
```
