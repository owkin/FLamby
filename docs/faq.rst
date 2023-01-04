FAQ
===

How can I do a clean slate?
---------------------------

To clean the environment you must execute (after being inside the FLamby
folder ``cd FLamby/``):

.. code:: bash

    conda deactivate
    make clean

I get an error when installing Flamby
-------------------------------------

error: [Errno 2] No such file or directory: 'pip'
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Try running:

.. code:: bash

    conda deactivate
    make clean
    pip3 install --upgrade pip

and try running your make installation option again.

I am installing Flamby on a machine equipped with macOS and an intel processor
------------------------------------------------------------------------------

In that case, you should use

.. code:: bash

    make install-mac

instead of the standard installation. If you have already installed the
flamby environment, just run

.. code:: bash

    conda deactivate 
    make clean

before running the install-mac installation again. This is to avoid the
following error, which will appear when running scripts. #### error :
OMP: Error #15

I or someone else already downloaded a dataset using another copy of the flamby repository, my copy of flamby cannot find it and I don't want to download it again, what can I do ?
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

There are two options. The safest one is to cd to the flamby directory
and run:

::

    python create_dataset_config.py --dataset-name fed_camelyon16 OR fed_heart_disease OR ... --path /path/where/the/dataset/is/located

This will create the required ``dataset_location.yaml`` file in your
copy of the repository allowing FLamby to find it.

One can also directly pass the ``data_path`` argument when instantiating
the dataset but this is not recommended.

.. code:: python

    from flamby.datasets.fed_heart_disease import FedHeartDisease
    center0 = FedHeartDisease(center=0, train=True, data_path="/path/where/the/dataset/is/located")

Collaborative work on FLamby: I am working with FLamby on a server with other users, how can we share the datasets efficiently ?
--------------------------------------------------------------------------------------------------------------------------------

The basic answer is to use the answer just above to recreate the config
file in every copy of the repository.

It can possibly become more seamless in the future if we introduce
checks for environment variables in FLamby, which would allow to setup a
general server-wise config so that all users of the server have access
to all needed paths. In the meantime one can fill/comment the following
bash script after downloading the dataset and share it with all users of
the server:

.. code:: bash

    python create_dataset_config.py --dataset-name fed_camelyon16 --path TOFILL
    python create_dataset_config.py --dataset-name fed_heart_disease --path TOFILL
    python create_dataset_config.py --dataset-name fed_lidc_idri --path TOFILL
    python create_dataset_config.py --dataset-name fed_kits19 --path TOFILL
    python create_dataset_config.py --dataset-name fed_isic2019 --path TOFILL
    python create_dataset_config.py --dataset-name fed_ixi --path TOFILL

Which allows users to set all necessary paths in their local copies.

Can I run clients in different threads with FLamby? How does it run under the hood?
-----------------------------------------------------------------------------------

FLamby is a lightweight and simple solution, designed to allow
researchers to quickly use cleaned datasets with a standard API. As a
consequence, the benchmark code performing the FL simulation is
minimalistic. All clients run sequentially in the same python
environment, without multithreading. Datasets are assigned to clients as
different python objects.

Does FLamby support GPU acceleration?
-------------------------------------

FLamby supports GPU acceleration thanks to the underlying deep learning
backend (pytorch for now).
