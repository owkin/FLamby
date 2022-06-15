## Quickstart

The Fed-TCGA-BRCA dataset requires no downloading or preprocessing.

### Dataset example

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

Below is an example of how to train a baseline model on one local train set and evaluate it on all the local test sets:
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
    NUM_CLIENTS
)
from flamby.datasets.fed_tcga_brca import FedTcgaBrca as FedDataset


train_dataset = FedDataset(center=0, train=True, pooled=False)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
lossfunc = BaselineLoss()
model = Baseline()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 5, 7, 9, 11, 13, 15, 17], gamma=0.5)

# Training loop
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

