
Quickstart
----------

We will start with the :any:`fed_tcga_brca` dataset because it requires no downloading or preprocessing and is very lightweight.
Therefore it provides a good introduction to navigating Flamby.
This tutorial does not require the use of a GPU.

Dataset example
^^^^^^^^^^^^^^^

We do provide users with two dataset abstractions: ``RawDataset`` and ``FedDataset``.
The recommended one is ``FedDataset`` as it is compatible with the rest of the repository's code.
This class allows to instantiate either the single-centric version of the dataset using the argument ``pooled = True``\ , or the different local datasets belonging to each client by providing the client index in the arguments (e.g. ``center = 2, pooled = False``\ ).
The arguments ``train = True`` and ``train = False`` allow to instantiate train or test sets (in both pooled or local cases).
It is important to understand that ``FedDataset`` is simply a wrapper around a `a map-style pytorch's dataset <https://pytorch.org/docs/stable/data.html#map-style-datasets>`_ and thus data can be accessed
the usual way by doing ``fed_dataset[i]`` where ``i`` belongs to ``[0, len(fed_dataset) - 1]``.
We also provide ``RawDataset`` objects which are less easy to work with but that should provide all metadata required for power users that find the ``FedDataset`` abstraction not flexible enough for their specific use-cases.

To instantiate the raw TCGA-BRCA or the Fed-TCGA-BRCA dataset, install FLamby (see :any:`installation`) and execute the following lines either in the python console, a notebook or a python script:

.. code-block:: python


   from flamby.datasets.fed_tcga_brca import TcgaBrcaRaw, FedTcgaBrca

   # Raw dataset
   mydataset_raw = TcgaBrcaRaw()

   # Pooled test dataset
   mydataset_pooled = FedTcgaBrca(train=False, pooled=True)

   # Center 2 train dataset
   mydataset_local2= FedTcgaBrca(center=2, train=True, pooled=False)

   # Computing the length of mydataset_local2
   N = len(my_dataset_local2)

   # Accessing individual samples
   X, y = mydataset_local2[N // 2]

Local training example
^^^^^^^^^^^^^^^^^^^^^^

Below is an example of how to train the chosen baseline model with default settings on one local train set and evaluate it on all the local test sets.
The code is identical to the ones would use on any pytorch dataset.

.. code-block:: python


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

   # Instantiation of local train set (and data loader)), baseline loss function, baseline model, default optimizer
   train_dataset = FedDataset(center=0, train=True, pooled=False)
   train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
   # This is the dataset's loss implementing in torch.nn's style (https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html#BCELoss)
   # therefore it needs to be instantiated
   lossfunc = BaselineLoss()
   # This is simply a pytorch model
   model = Baseline()
   # This is simply a pytorch optimizer
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
   # Instantiation of a list of the local test sets
   test_dataloaders = [
               torch.utils.data.DataLoader(
                   FedDataset(center=i, train=False, pooled=False),
                   batch_size=BATCH_SIZE,
                   shuffle=False,
                   num_workers=0,
               )
               for i in range(NUM_CLIENTS)
           ]
   # Helper function performing the evaluation on a list of dataloaders
   # it can also be done manually
   dict_cindex = evaluate_model_on_tests(model, test_dataloaders, metric)
   print(dict_cindex)


Federated Learning training example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

See below an example of how to train a baseline model on the Fed-TCGA-BRCA dataset in a federated way using the FedAvg strategy and evaluate it on the pooled test set:

.. code-block:: python


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

   # We loop on all the clients of the distributed dataset and instantiate associated data loaders
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
   # This helper function returns the number of rounds necessary to perform approximately as many
   # epochs on each local dataset as with the pooled training
               "nrounds": get_nb_max_rounds(100),
           }
   s = strat(**args)
   m = s.run()[0]

   # Evaluation
   # We only instantiate one test set in this particular case: the pooled one
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

Note that other models and loss functions compatible with the dataset can be used as long as they inherit from torch.nn.Module.

Using other FLamby's datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We will follow up on how to download datasets that are not hosted on this repository.
We will use the example of :any:`fed_heart` as its download process is simple and it requires no preprocessing.
Note that to use each new dataset if you had chosen the lightweight install you might need to install additional requirements
by rerunning ``pip install -e`` using different options (``[cam16, heart, isic2019, ixi, kits19, lidc, tcga]``).
In this case if you had done ``pip install -e .[tcga]`` then run ``pip install -e .[heart]``
Then please run:


.. code-block::

   cd flamby/datasets/fed_heart_disease/dataset_creation_scripts
   python download.py --output-folder ./heart_disease_dataset

You can instantiate this dataset as you did ``FedTcgaBrca`` by executing:

.. code-block:: python

   from flamby.datasets.fed_heart_disease import HeartDiseaseRaw, FedHeartDisease
   # Raw dataset
   mydataset_raw = HeartDiseaseRaw()
   # Pooled train dataset
   mydataset_pooled = FedHeartDisease(train=True, pooled=True)
   # Center 1 train dataset
   mydataset_local1= FedHeartDisease(center=1, train=True, pooled=False)

Other datasets downloads and instantiations follow a similar pattern, please find instructions for each of the dataset in their corresponding sections.
Note however that you certainly do not have to download them all as each takes some non negligible disk space. 

* :any:`fed_ixi`.
* :any:`fed_isic`.
* :any:`fed_camelyon`.
* :any:`fed_lidc`.
* :any:`fed_kits19`.


Training and evaluation in a pooled setting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To train and evaluate the baseline model for the pooled Heart Disease dataset using a helper script, run:

.. code-block::

   cd flamby/datasets/fed_heart_disease
   python benchmark.py --num-workers-torch 0

Benchmarking FL strategies
^^^^^^^^^^^^^^^^^^^^^^^^^^

The command below allows to reproduce the article's results for a given seed aka:

* train a model on the pooled dataset and evaluate it on all test sets (local and pooled).
* train models on all local datasets and evaluate them on all test sets (local and pooled).
* train models in a federated way for all FL strategies with associated hyperparameters in corresponding config files
  and evaluate them on all test sets (local and pooled).


The config files given in the repository (\ ``flamby/config_*.json``\ ) hold the different HPs sets used in the companion 
article for the FL strategies on the different datasets.
The results are stored in the csv file specified either in the config file or with the --results-file-path option.

.. code-block::

   cd flamby/benchmarks
   python fed_benchmark.py --config-file-path ../config_heart_disease.json --results-file-path ./test_res_0.csv --seed 0

Note that 1. this script might take a long time for large datasets 2. the communication budget (the number of rounds used)
might be insufficient for full convergence. For tighter control over the parameters return to subsection `Federated Learning training example`_.
and follow instructions.  

For more details about how to reproduce results in the article go to :any:`reproducing`

FL training and evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to train and evaluate the baseline model with a specific FL strategy and associated hyperparameters, one can run the following command:

.. code-block::

   python fed_benchmark.py --strategy FedProx --mu 1.0 --learning_rate 0.05 --config-file-path ../config_heart_disease.json \
    --results-file-path ./test_res1.csv --seed 1

In this case the strategy specific HPs in the config file are ignored and the HPs used are given by the user or take the default values given in this script.

Going further
^^^^^^^^^^^^^
If you made it here please consider contributing to FLamby by either opening `issues <https://github.com/owkin/FLamby/issues>`_  
on pain-points you might have encountered or on things you do not understand after having consulted the :any:`faq`.
If you think you can fix the issue yourself or want to add new distributed datasets with natural splits follow the steps
outlined in :any:`contributing`