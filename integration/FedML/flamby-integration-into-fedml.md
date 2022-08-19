# Using FLamby with FedML

This document highlights how to interface FLamby with [FedML](https://github.com/FedML-AI/FedML).

## FedML installation

The first step is to install FedML in the FLamby environment.

```bash
conda activate flamby
pip install fedml==0.7.290
# necessary for the tutorial
pip install easydict
pip install jupyter
```
## Launching FedML in RAM  

The important thing to know is that the procedure to launch FedML trainings with all FLamby datasets is similar. Regardless of the dataset, the same components will have to be setup each time.
We will take the example of the Fed-heart disease dataset, we assume it is already downloaded.
The easiest way is to copy-paste the code blocks one after another in a jupyter notebook with flamby.

FedML has three important arguments: the dataset, the trainer and the args namespace.
Let's instantiate the dataset first, the dataset is simply a list with handles towards dict containing either links directly to the dataloaders or to the size of the dataset from which they originate. There are also handles towards the total size of the train and test.
We will create it using straightforward FLamby code:
```python
# We import lots of FLamby specific quantities that we will use either now or subsequently
from flamby.datasets.fed_heart_disease import (
    BATCH_SIZE,
    LR,
    NUM_EPOCHS_POOLED,
    Baseline,
    BaselineLoss,
    FedHeartDisease,
    metric,
    get_nb_max_rounds,
    NUM_CLIENTS,
)
from torch.utils.data import DataLoader as dl

train_data_local_dict = {}
test_data_local_dict = {}
train_data_local_num_dict = {}
test_data_local_num_dict = {}

for client_idx in range(NUM_CLIENTS):
    # We create the traditional FLamby datasets
    train_dataset = FedHeartDisease(center=client_idx, train=True)
    train_dataloader = dl(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=10)
    test_dataset = FedHeartDisease(center=client_idx, train=False)
    test_dataloader = dl(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=10, drop_last=True)
    train_data_local_num_dict[client_idx] = len(train_dataset)
    train_data_local_dict[client_idx] = train_dataloader
    test_data_local_dict[client_idx] = test_dataloader

train_data_num = sum({v for _, v in train_data_local_num_dict.items()})
test_data_num = sum({v for _, v in test_data_local_num_dict.items()})
train_data_global = None
test_data_global = None
class_num = 2

# We instantiate the dataset FedML object, the order is very important
dataset = [train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ]
```
We will now instantiate the FedML trainer, which is simply a cllass with  `get/set_model_params` methods and a `train` method doing the local updates, we'll use FLamby's abstraction: `DataLoaderWithMemory` in order to more closely follow FLamby's specifications:  

```python
from fedml.core.alg_frame.client_trainer import ClientTrainer
from flamby.strategies.utils import DataLoaderWithMemory
from torch.optim import SGD
import logging

# We will implement a class
class HeartDiseaseTrainer(ClientTrainer):
    def __init__(self, model, num_updates, args=None):
        super().__init__(model, args)
        # We have to count in epochs
        self.epochs = 10
        self._optimizer = SGD(model.parameters(), lr=LR)
        self._loss = BaselineLoss()
        self.num_updates = num_updates

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        logging.info("set_model_params")
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        logging.info("Start training on Trainer {}".format(self.id))
        if not(hasattr(self, "dl_with_memory")):
            self.dl_with_memory = DataLoaderWithMemory(train_data)
        # We almost copy-paste verbatim FLamby's code but one could modify
        # it to use a different optimizer or do epochs instead of updates

        self.model = self.model.train()
        for idx, _batch in enumerate(range(self.num_updates)):
            X, y = self.dl_with_memory.get_samples()
            # Compute prediction and loss
            _pred = self.model(X)
            _loss = self._loss(_pred, y)

            # Backpropagation
            _loss.backward()
            self._optimizer.step()
            self._optimizer.zero_grad()

```
The last object to create is the args namespace:
```python
from easydict import EasyDict as edict

# client_num_per_rounds is relative to client subsampling, 
# we compute the number of rounds to do 100 updates
NB_ROUNDS = get_nb_max_rounds(100)
args = edict({"client_num_per_round": NUM_CLIENTS , "comm_round": NB_ROUNDS, "frequency_of_the_test": NB_ROUNDS // 10, "client_num_in_total": NUM_CLIENTS, "dataset": ""})

```


Now we can put everything together: we first instantiate a trainer and we'll use the args and datasets alongside the `FedAvgAPI` abstraction:

```python
from fedml.simulation.sp.fedavg.fedavg_api import FedAvgAPI
from flamby.utils import evaluate_model_on_tests

m = Baseline()
trainer = HeartDiseaseTrainer(model=m, num_updates=100, args=args)
# We instantiate a FedML FedAvg API
s = FedAvgAPI(args, "cpu", dataset, m)
# As we cannot control the trainer used, we'll set it ourselves
s.model_trainer = trainer
# WE reinitialize the client list with the new trainer manually
s.client_list = []
s._setup_clients(
            s.train_data_local_num_dict, s.train_data_local_dict, s.test_data_local_dict, s.model_trainer,
        )
# We will kill the validation as it is buggy
s._local_test_on_all_clients = lambda x: None
s.train()

# we retrieve the final global model
final_model = s.model_trainer.model
print(evaluate_model_on_tests(final_model, [dataset[6][i] for i in range(NUM_CLIENTS)], metric))
```

## FedML: FL communications outside of RAM thanks to the MPI backend

**WARNING: This part is not functional yet and is a Work In Progress !**
### Setup the MPI backend

Install MPI by following instructions [here](https://www.open-mpi.org/faq/?category=building#easy-build).
This is quite a long process, for which we provide detailed Mac instructions below:
```
wget https://download.open-mpi.org/release/open-mpi/v2.0/openmpi-2.0.4.tar.gz
tar -xf openmpi-2.0.4.tar.gz
cd openmpi-2.0.4/
./configure --prefix=$HOME/opt/usr/local
make all
make install
# add to your bashrc/zshrc
alias mpirun=$HOME/opt/usr/local/bin/mpirun
mpirun --version
```
Then we install [mpi4py](https://mpi4py.readthedocs.io/en/stable/).
```
conda install mpi4py
```
### Start the simulation

We launch the `launch_client.py` script for the first time launching
`num_workers + 1` processes with MPI.

```bash
hostname > mpi_host_file
(which mpirun) -np 5 python launch_client.py --cf config/fedml_config.yaml 
```

Then we up the server and the clients using the same script:


```bash
python launch_client.py --cf config/fedml_config.yaml --run_id heart_disease --rank 0 --role server
````

```
# in a new terminal window
python launch_client.py --cf config/fedml_config.yaml --run_id heart_disease --rank 1 --role client
# in a new terminal window
python launch_client.py --cf config/fedml_config.yaml --run_id heart_disease --rank 2 --role client
# in a new terminal window
python launch_client.py --cf config/fedml_config.yaml --run_id heart_disease --rank 3 --role client
# in a new terminal window
python launch_client.py --cf config/fedml_config.yaml --run_id heart_disease --rank 4 --role client
```
Then the training will occur.


## Citation:

```bash
@article{he2021fedcv,
  title={Fedcv: a federated learning framework for diverse computer vision tasks},
  author={He, Chaoyang and Shah, Alay Dilipbhai and Tang, Zhenheng and Sivashunmugam, Di Fan1Adarshan Naiynar and Bhogaraju, Keerti and Shimpi, Mita and Shen, Li and Chu, Xiaowen and Soltanolkotabi, Mahdi and Avestimehr, Salman},
  journal={arXiv preprint arXiv:2111.11066},
  year={2021}
}
@misc{he2020fedml,
      title={FedML: A Research Library and Benchmark for Federated Machine Learning},
      author={Chaoyang He and Songze Li and Jinhyun So and Xiao Zeng and Mi Zhang and Hongyi Wang and Xiaoyang Wang and Praneeth Vepakomma and Abhishek Singh and Hang Qiu and Xinghua Zhu and Jianzong Wang and Li Shen and Peilin Zhao and Yan Kang and Yang Liu and Ramesh Raskar and Qiang Yang and Murali Annavaram and Salman Avestimehr},
      year={2020},
      eprint={2007.13518},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
