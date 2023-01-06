Reproduction instructions
=========================

Heterogeneity plots
^^^^^^^^^^^^^^^^^^^

Most plots from the article can be reproduced using the following
commands after having downloaded the corresponding datasets:

-  :any:`fed_tcga_brca`

::

    cd flamby/datasets/fed_tcga_brca
    python plot_kms.py

-  :any:`fed_lidc`

   ::

       cd flamby/datasets/fed_lidc_idri
       python lidc_heterogeneity_plot.py

-  :any:`fed_isic`

**In order to exactly reproduce the plot in the article**, one needs to
first deactivate color constancy normalization when preprocessing the
dataset (change ``cc`` to ``False`` in ``resize_images.py``) while
following download and preprocessing instructions (in :any:`fed_isic`). 
Hence one might have to download the dataset a second time, if it was already
downloaded, and therefore to potentially update
``dataset_location.yaml`` files accordingly.

::

    cd flamby/datasets/fed_isic2019
    python heterogeneity_pic.py

-  :any:`fed_ixi`

   ::

       cd flamby/datasets/fed_ixi
       python ixi_plotting.py

-  :any:`fed_kits19`

   ::

       cd flamby/datasets/fed_kits19/dataset_creation_scripts
       python kits19_heterogenity_plot.py

-  :any:`fed_heart`

   ::

       cd flamby/datasets/fed_heart_disease
       python heterogeneity_plot.py

-  :any:`fed_camelyon`

First concatenate as many 224x224 image patches extracted from regions
on the slides containing matter from Hospital 0 and Hospital 1 (see what
is done in the `tiling
script <https://github.com/owkin/FLamby/blob/main/flamby/datasets/fed_camelyon16/dataset_creation_scripts/tiling_slides.py>`__
to collect image patches) as can be fit in the RAM. Then compute both
histograms **per-color-channel** using 256 equally sized bins with the
``np.histogram`` function with ``density=True``. Then save the results
respectively as: histogram\_0.npy, histogram\_1.npy and bins\_0.npy.
Once this is done run in the current directory:

::

    cp -t flamby/datasets/fed_camelyon16 histograms_{0, 1}.npy bins_0.npy
    cd flamby/datasets/fed_camelyon16
    python plot_camelyon16_histogram.py

Results plots
^^^^^^^^^^^^^

The results are stored in ``flamby/results`` in corresponding subfolders
``results_benchmark_fed_dataset`` for each dataset. These results can be
plotted using:

::

    python plot_results.py

which produces the plot found at the end of the main article.

In order to re-run each of the benchmark on your machine, first download
the dataset you are interested in (be mindful that you might have to specify
another option to ``pip install -e`` to install additional requirements 
if you had chosen a lightweight installation).
and then run the following command replacing ``config_dataset.json`` by one of the listed config files
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
Note that this can be excessively long for some datasets.


We have observed that results vary from machine to machine and are
sensitive to GPU randomness. However you should be able to reproduce the
results up to some variance and results on the same machine should be
perfecty reproducible. Please open an issue if it is not the case. The
script ``extract_config.py`` allows to go from a results file to a
``config.py``.
To fo further into reproducibility you can try the :any:`docker` section.  

Note that the communication budget in terms of rounds might be insufficient
for full convergence of the model. A quick fix would be simply to use more rounds,
(see the :any:`quickstart` section to learn how to change parameters).
Otherwise try different parameters such as learning rates !
All strategy-specific HP can be found in the :any:`strategies` API doc.  

More involved modifications such as using learning rate schedulers might be needed to
obtain optimal results but it would require to slightly modify the strategy code.
