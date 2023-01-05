Benchmarks
==========

The results are stored in ``flamby/results`` in corresponding subfolders
``results_benchmark_fed_dataset`` for each dataset. These results can be
plotted using:

::

    python plot_results.py

which produces the plot found at the end of the main article.

In order to re-run each of the benchmark on your machine, first download
the dataset you are interested in and then run the following command
replacing ``config_dataset.json`` by one of the listed config files
(``config_camelyon16.json``, ``config_heart_disease.json``,
``config_isic2019.json``, ``config_ixi.json``, ``config_kits19.json``,
``config_lidc_idri.json``, ``config_tcga_brca.json``):

::

    cd flamby/benchmarks
    python fed_benchmark.py --seed 42 -cfp ../config_dataset.json
    python fed_benchmark.py --seed 43 -cfp ../config_dataset.json
    python fed_benchmark.py --seed 44 -cfp ../config_dataset.json
    python fed_benchmark.py --seed 45 -cfp ../config_dataset.json
    python fed_benchmark.py --seed 46 -cfp ../config_dataset.json

The config lists all hyperparameters used for each FL strategy.


We have observed that results vary from machine to machine and are
sensitive to GPU randomness. However you should be able to reproduce the
results up to some variance and results on the same machine should be
perfecty reproducible. Please open an issue if it is not the case. The
script ``extract_config.py`` allows to go from a results file to a
``config.py``. 
Note that the communication budget in terms of rounds might be insufficient
for full convergence of the model. A quick fix would be simply to use more rounds,
(see the :any:`quickstart` section to learn how to change parameters).
More involved modifications such as using learning rate schedulers might be needed to
obtain optimal results but it would require to slightly modify the strategy code.
