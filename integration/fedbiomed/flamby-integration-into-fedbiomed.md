# Using FLamby with Fed-BioMed

This document highlights how to interface FLamby with [Fed-BioMed](https://gitlab.inria.fr/fedbiomed/fedbiomed), the FL framework by Inria Sophia.

## Fed-BioMed installation

Before running the examples in the notebook, it is necessary to have Fed-BioMed installed on your machine. An easy-to-follow guide is available here to help you with the process: https://fedbiomed.gitlabpages.inria.fr/latest/tutorials/installation/0-basic-software-installation/.

## Launching Fed-BioMed components (FLamby datasets configuration)
 
An important thing to know is that the procedure to configure all FLamby datasets in Fed-BioMed is similar. Regardless of the dataset, the same components will be set up each time (the network, a certain number of nodes, and the researcher).

The first step is the network launching, by entering the following command on your terminal : `${FEDBIOMED_DIR}/scripts/fedbiomed_run network`

Now, you will have to create as many nodes as there are centers in your specific FLamby dataset .
For instance, in the case of IXI, there are **3** centers, meaning that **3** nodes have to be created.

Enter this command to create a **1st** node : `${FEDBIOMED_DIR}/scripts/fedbiomed_run node add`

Select FLamby option and the IXI dataset through the CLI menu.
The name and the description can be filled as you want. The tag has to be set the same as it will be defined in the example notebook (let's enter `ixi`).
A center id will also be asked. Center ids for IXI are ranged between **0** and **2**. Enter **0** for this 1st node.

You can now launch this node with the following command: `${FEDBIOMED_DIR}/scripts/fedbiomed_run node start`

To complete the procedure, a **2nd** and a **3rd** node have to be created. Open a new terminal for each of them.
The only thing that will differ is the specification of a different (non-default) configuration file, to tell that we want to perform operations on a different node:

`${FEDBIOMED_DIR}/scripts/fedbiomed_run node config config2.ini add`

`${FEDBIOMED_DIR}/scripts/fedbiomed_run node config config3.ini add`

Center ids need to be set as **1** and **2**, respectively to the **2nd** and **3rd** node. The tag has to be defined identically to the 1st node, `ixi`.

To start these two nodes, simply execute:

`${FEDBIOMED_DIR}/scripts/fedbiomed_run node config config2.ini start`

`${FEDBIOMED_DIR}/scripts/fedbiomed_run node config config3.ini start`

The only thing remaining is to launch the researcher (jupyter notebook console). Open a new terminal and execute the following command: `${FEDBIOMED_DIR}/scripts/fedbiomed_run researcher`

Congratulations, the IXI example in the researcher notebook can now be run! Feel free to follow the same steps to configure all the others!

## Notebook (researcher side)

The first step is to create a class inheriting from Fed-BioMed's `TorchTrainingPlan`.
This class should have different attributes: 
- model
- loss
- optimizer

This is very similar in spirit to FLamby's strategies. We, then, should implement the forward of the model in the method `training_step`.
For this, we just have to write the class using one of FLamby's dataset's set of arguments.
The only difficulty is to also take care of the serialization process (what happens when you pickle a class, send it to a distant computer and reload it into a different runtime).
In order to do that we just have to copy the FLamby's import and to give it to the `add_dependency` method from the parent class.

```python
# We import the TorchTrainingPlan class from fedbiomed
from fedbiomed.common.training_plans import TorchTrainingPlan
# We import all flamby utilities, the model, loss, batch_size and optimizer details
from flamby.datasets.fed_ixi import (Baseline, BaselineLoss, Optimizer BATCH_SIZE, LR, get_nb_max_rounds)
# We create a class inheriting from TorchTrainingPlan reimplementing
# the training step method and filling the necesssary class attributes from FLamby
# information
class FLambyTrainingPlan(TorchTrainingPlan):
    # Init of UNetTrainingPlan
    def __init__(self, model_args: dict = {}):
        super().__init__(model_args)
        
        self.model = Baseline() # UNet model
        self.loss = BaselineLoss() # Dice loss
        # As we will be serializing the instance, we need the distant machine
        # to have made the same imports as we did in order to deserialize the 
        # object without error.
        # This will be the caqe for TorchTrainingPlan but not for FLamby
        # so we add it below
        deps = ['from flamby.datasets.fed_ixi import (Baseline,\
                BaselineLoss,\
                Optimizer)',]
        self.add_dependency(deps)
        
        self.optimizer = Optimizer(self.parameters())
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, img, target):
        # We implement the forward and return the loss
        output = self.forward(img)
        loss = self.loss(output, target)
        return loss
```

Fed-BioMed requires to pass the `batch_size`, `learning rate` and number of local `epochs` directly into the experiment abstraction. In FLamby the strategies abstraction fuse the TrainingPlan and the experiment.
```python
# We create a dictionnary of FedBioMed kwargs with the: batch_size, learning-rate and epochs.
# Be careful, Fed-BioMed doesn't count count local training steps in batch-updates but in local epochs !
# In this examplen we do 10 epochs on each of the local datasets (which amounts to different
# number of batch-updates per client) so cannot be reproduced using FLamby internal strategies
training_args = {
    'batch_size': BATCH_SIZE,
    'lr': LR,
    'epochs': 10,
}
```

We are now ready to launch an experiment doing 50 rounds of FederatedAveraging.
We will the experiment abstraction and pass to it, the class from above, the `training_args` and the `FedAverage` aggregator.

```python
from fedbiomed.researcher.experiment import Experiment
from fedbiomed.researcher.aggregators.fedavg import FedAverage
# This is just a handle to retrieve models and performances, later
tags =  ['ixi']
num_rounds = 50
exp = Experiment(tags=tags,
                 model_class=FLambyTrainingPlan,
                 training_args=training_args,
                 round_limit=num_rounds,
                 aggregator=FedAverage(),
                )
# This command runs the experiment, the model and results are stored
exp.run()
```
We can now load the resulting model and evaluate its performances:

- Loading

```python
fed_model_ixi = exp.model_instance()
fed_model_ixi.load_state_dict(exp.aggregated_params()[num_rounds - 1]['params'])
```

- Evaluation

```python
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from flamby.datasets.fed_ixi import (metric as ixi_metric,
                                     FedClass as ixi_fed)
                                     
def compute_test_performance(model, data_loader, metric, batch_size):
    print("Test set size:", len(data_loader))
    model.eval()
    computed_performances = []
    device = "cpu"
    
    with torch.no_grad():
        for idx, (data, target) in tqdm(enumerate(DataLoader(dataset=data_loader,batch_size=batch_size))):
            data, target = data.to(device), target.to(device)    
            output = model(data)        
            target, output = target.numpy(), output.numpy()
            performance = metric(target, output)
            computed_performances.append(performance)
          
    print("Test mean performance:", torch.mean(torch.tensor(computed_performances)))
    
test_dataloader_ixi = ixi_fed(train=False, pooled=True)
compute_test_performance(fed_model_ixi, test_dataloader_ixi, ixi_metric, ixi_batch_size)
```