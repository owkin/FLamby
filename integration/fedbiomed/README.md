# FLamby integration into Fed-BioMed

## Installation

Before running the examples in the notebook, it is necessary to have Fed-BioMed installed on your machine. You can follow this tutorial to know detailed steps: https://fedbiomed.gitlabpages.inria.fr/latest/tutorials/installation/0-basic-software-installation/.

## Launching Fed-BioMed components
Once Fed-BioMed is installed, launch the network with the following command on your terminal : `${FEDBIOMED_DIR}/scripts/fedbiomed_run network`

Now, you will have to create as many nodes as there are centers in your specific FLamby dataset.
For instance, in the case of IXI, there are **3** centers, meaning that **3** nodes have to be created.

Enter this command to create a **1st** node : `${FEDBIOMED_DIR}/scripts/fedbiomed_run node add`

Select flamby option and the IXI dataset through the CLI menu.
The name and the description can be filled as you want. The tag has to be set the same as it will be defined in the example notebook (enter `ixi`).
A center id will also be asked. Center ids for IXI are ranged between **0** and **2**. Enter **0** for this 1st node.

You can now launch this node with the following command: `${FEDBIOMED_DIR}/scripts/fedbiomed_run node start`

To complete the procedure, a **2nd** and a **3rd** node have to be created. Open a new terminal for each of them.
The only thing that will differ is the specification of a different (non-default) configuration file, to tell that we want to perform operations on a different node:

`${FEDBIOMED_DIR}/scripts/fedbiomed_run node config config2.ini add`

`${FEDBIOMED_DIR}/scripts/fedbiomed_run node config config3.ini add`

Center id will be set as **1** and **2**, respectively to the **2nd** and **3rd** node. The tag has to be defined identically to the 1st node, `ixi`.

To start these two nodes, simply execute:

`${FEDBIOMED_DIR}/scripts/fedbiomed_run node config config2.ini start`

`${FEDBIOMED_DIR}/scripts/fedbiomed_run node config config3.ini start`

The only thing remaining is to launch the researcher (jupyter notebook console). Open a new terminal and execute the following command: `${FEDBIOMED_DIR}/scripts/fedbiomed_run researcher`

Congratulations, the examples in the researcher notebook can now be run!
