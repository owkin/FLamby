# FLamby TCGA BRCA

This dataset is [the FLamby TCGA BRCA dataset](https://github.com/owkin/FLamby/tree/main/flamby/datasets/fed_tcga_brca).

## Opener usage

The opener exposes 2 methods:

* `get_X` returns a config dictionary to instantiate a FLamby dataset.
* `get_y` Not used in training, and return all test samples on testing.
