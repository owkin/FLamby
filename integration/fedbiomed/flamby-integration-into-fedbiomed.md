# Using FLamby with Fed-BioMed

This document highlights how to interface FLamby with [Fed-BioMed](https://gitlab.inria.fr/fedbiomed/fedbiomed), the FL framework by Inria Sophia.

## Fed-BioMed installation

Before running the examples below, it is necessary to have Fed-BioMed installed on your machine. An easy-to-follow guide is available [here](https://fedbiomed.gitlabpages.inria.fr/latest/tutorials/installation/0-basic-software-installation/) to help you with the process.
Please use the branch `poc/flamby`:
```
git checkout poc/flamby
```

In addition, it is also necessary to have a local version of the FLamby package on your computer already installed in editable mode and to have downloaded
the required dataset.s (see Getting started section).

Once Fed-BioMed is installed alongside FLamby, we need to first update all fedbiomed's conda environments' yaml config files.
Go to the fedbiomed directory and cd into the conda environments folder:
```
cd /path/where/is/installed/fedbiomed/envs/development/conda/
```
Open both fedbiomed-node(-macosx).yaml & fedbiomed-researcher(-macosx).yaml into a text editor or IDE and add the following line under pip:
```
      - -e /path/towards/your/editable/installation/of/FLamby
```
Then run:
```
${FEDBIOMED_DIR}/scripts/fedbiomed_environment clean
${FEDBIOMED_DIR}/scripts/configure_conda
 ```
 At the end you should have access to 3 new conda environments.
 You can test your installation by doing:
 ```
conda activate fedbiomed-researcher
ipython
import flamby
exit()
```
If the import goes through the installation is working.
## Launching Fed-BioMed components (FLamby datasets configuration)
 
An important thing to know is that the procedure to configure all FLamby datasets in Fed-BioMed is similar. Regardless of the dataset, the same components will be set up each time (the network, a certain number of nodes, and the researcher).

Firstable, make sure you have downloaded and installed the FLamby dataset you want to use: there should be a `dataset_location.yaml` file
inside the FLamby repository in the `dataset_creation_scripts` folder of the corresponding dataset.
```
cat /path/towards/flamby/as/given/to/conda/FLamby/flamby/datasets/fed_ixi/dataset_creation_scripts
````
Should display:
```
dataset_path: /path/towards/downloaded/ixi
download_complete: true
preprocessing_complete: true
```

The first step is to launch the network, by entering the following command on your terminal: 
```
${FEDBIOMED_DIR}/scripts/fedbiomed_run network
```

Once the network is deployed, you will have to create as many nodes as there are centers in the FLamby dataset you want to use.
In this example we'll use IXI that has **3** centers, meaning that **3** nodes have to be created.

### First node

Enter this command to create the **1st** node : 
```
${FEDBIOMED_DIR}/scripts/fedbiomed_run node add
```

Select FLamby option (6) and the IXI dataset (5) through the CLI menu.
Then enter a name and a description. You are free to write anything that goes through your mind except for the **tag**.
**All nodes need to share the same tag** and this tag needs to be passed to the experiment.
In this example we will choose `ixi` for the tag, so write ixi in the CLI and enter.
The last step is to input the center id of the center corresponding to the node, as this is the first node we'll write 0 as FLamby uses python indexing. So write 0 and enter.
Once everything is validated you can now launch this node with the following command: 
```
${FEDBIOMED_DIR}/scripts/fedbiomed_run node start
```
We will now do the same for the second and 3rd node, however it is of the utmost importance to **use different shells for each node**.
### Second node

Open a new terminal and run:
```
${FEDBIOMED_DIR}/scripts/fedbiomed_run node config config2.ini add
```
As with the first node select FLamby and IXI dataset, enter a name and a description. Enter `ixi` for the tag and select 1 for the center id to deploy the second node.
Then similarly run:
```
${FEDBIOMED_DIR}/scripts/fedbiomed_run node config config2.ini start
```
### 3rd node
Open a new terminal and run:
```
${FEDBIOMED_DIR}/scripts/fedbiomed_run node config config3.ini add
```
As with the first node select FLamby and IXI dataset, enter a name and a description. Enter `ixi` for the tag and select 2 for the center id to deploy the third and final node.
Then similarly run:
```
${FEDBIOMED_DIR}/scripts/fedbiomed_run node config config3.ini start
```

Congratulations, the IXI example in the researcher notebook can now be run!  
Feel free to follow the same steps to configure other FLamby datasets.  

## Notebook (researcher side)

We need to send commands to the nodes that are already up we will use the researcher environment.
You can now activate the conda environment of the researcher by running:
```
conda activate fedbiomed-researcher
```
Then the simplest way is to use ipython or jupyter to copy-paste the commands below:

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
from flamby.datasets.fed_ixi import (Baseline, BaselineLoss, Optimizer, BATCH_SIZE, LR, get_nb_max_rounds)
# We create a class inheriting from TorchTrainingPlan reimplementing
# the training step method and filling the necessary class attributes from FLamby
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
        # This will be the case for TorchTrainingPlan but not for FLamby
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
You can go and copy-paste the code above in ipython.

Fed-BioMed requires to pass the `batch_size`, `learning rate` and number of local `epochs` directly into the experiment abstraction. In FLamby the strategies abstraction fuse the TrainingPlan and the experiment.
Copy-paste the code below in your notebook or interactive shell:
```python
# We create a dictionary of FedBioMed kwargs with the: batch_size, learning-rate and epochs.
# Fed-BioMed also counts local training steps in batch-updates (change currently made in the following branch of Fed-BioMed):
# https://gitlab.inria.fr/fedbiomed/fedbiomed/-/tree/poc/flamby (will be integrated to master very soon)
# In this example we set this parameter to 100, as in the FLamby benchmarks
training_args = {
    'batch_size': BATCH_SIZE,
    'lr': LR,
    'num_updates': 100,
}
```

We are now ready to launch an experiment doing 5 rounds of FederatedAveraging.
We will instantiate the experiment abstraction and give it: the class from above, the `training_args` and the `FedAverage` aggregator.
Copy-paste and execute the code below:

```python
from fedbiomed.researcher.experiment import Experiment
from fedbiomed.researcher.aggregators.fedavg import FedAverage
# This is just a handle to access the available nodes that we tagged with ixi
tags =  ['ixi']
num_rounds = 5
exp = Experiment(tags=tags,
                 model_class=FLambyTrainingPlan,
                 training_args=training_args,
                 round_limit=num_rounds,
                 aggregator=FedAverage(),
                )

```
Now the last step to launch the simulated Federated Learning between the different nodes is to run the experiment:
```
exp.run()
```
Once the nodes have finished training the model, we can load it in the researcher runtime and evaluate its performances:


```python
# We retrieve the model architecture from the experiment:
fed_model_ixi = exp.model_instance()
# We load the weights corresponding to the last round
fed_model_ixi.load_state_dict(exp.aggregated_params()[num_rounds - 1]['params'])
```

We can then evaluate it using traditional FLamby's evaluation scripts:

```python
from torch.utils.data import DataLoader
from flamby.utils import evaluate_model_on_tests
from flamby.datasets.fed_ixi import metric, BATCH_SIZE, FedIXITiny

# We load the test datasets
test_dataloader_ixi_client0 = DataLoader(dataset=FedIXITiny(center=0, train=False),batch_size=BATCH_SIZE)
test_dataloader_ixi_client1 = DataLoader(dataset=FedIXITiny(center=1, train=False),batch_size=BATCH_SIZE)
test_dataloader_ixi_client2 = DataLoader(dataset=FedIXITiny(center=2, train=False),batch_size=BATCH_SIZE)

# We evaluate the model on them
evaluate_model_on_tests(fed_model_ixi,
                        [test_dataloader_ixi_client0,
                         test_dataloader_ixi_client1,
                         test_dataloader_ixi_client2],
                        metric)
```
One can change the number of round or the number of updates and continue launching jobs. One can also use clean the environment and get ready to use other datasets.
