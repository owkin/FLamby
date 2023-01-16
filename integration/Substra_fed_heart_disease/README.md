# Substra

[Substra](https://docs.substra.org/en/0.21.0/) is an open source federated learning framework. It provides a flexible python interface and a web app to run federated learning training at scale.

Substra main usage is a production one. It has already been deployed and used by hospitals and biotech companies (see the [MELLODDY](https://www.melloddy.eu/) project for instance). Yet Substra can also be used on a single machine on a virtually splitted dataset for two use cases:

- debugging code before launching experiments on a real network
- performing FL simulations

Substra was created by [Owkin](https://owkin.com/) and is now hosted by the [Linux Foundation for AI and Data](https://lfaidata.foundation/).

# SubstraFL example on FLamby TCGA BRCA dataset

This example illustrate the basic usage of SubstraFL, and propose a model training by Federated Learning using the Federated Averaging strategy on the TCGA BRCA dataset.

The objective of this example is to launch a *federated learning* experiment on the six centers provided on Flamby (called organizations in Substra), using the **FedAvg strategy** on the **baseline model**.

This example runs in local mode, no need to deploy a Substra platform to run it.

This example is designed to work in local subprocess mode, and not in local Docker mode. As FLamby is a dependency for both the metric and the algorithm, it would need to be installed in editable mode in each container, which has not been added to the example yet.

## Requirements

Please ensure to have, in addition to flamby, all the libraries needed for this example installed, a *requirements.txt* file is included in the example, this way you can run the command: `pip install -r requirements.txt` to install them.

## Objective

This example runs a federated training on all 6 centers.

This example shows how to interface Substra with Flamby. The modifications to use this example on other dataset are minimal and have to be made on the openers files, the metric file and the main script.
This example runs in local mode, simulating a **federated learning** experiment.

## Run the example

To run the example, launch `jupyter notebook` and copy paste and run the cells of the `flamby-integration-into-substra.md` file (you can also do it in classical Python console with your favorite IDE)..
