# Using FLamby with Fed-BioMed

This document highlights how to interface FLamby with [Fed-BioMed](https://gitlab.inria.fr/fedbiomed/fedbiomed), the FL framework by Inria Sophia.

TO FILL: how to set up each dataset in Fed-BioMed

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
    
    def training_step(self, img, target):
        # We implement the forward and return the loss
        output = self.model(img)
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
TO FILL