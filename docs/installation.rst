Installation
------------

We recommend using anaconda and pip. You can install anaconda by downloading and executing appropriate installers from the `Anaconda website <https://www.anaconda.com/products/distribution>`_\ , pip often comes included with python otherwise check `the following instructions <https://pip.pypa.io/en/stable/installation/>`_. We support all Python version starting from **3.7**.

You may need ``make`` for simplification. The following command will install all packages used by all datasets within FLamby. If you already know you will only need a fraction of the datasets inside the suite you can do a partial installation and update it along the way using the options described below.
Create and launch the environment using:

.. code-block:: bash

   git clone https://github.com/owkin/FLamby.git
   cd FLamby
   make install
   conda activate flamby

To limit the number of installed packages you can use the ``enable`` argument to specify which dataset(s)
you want to build required dependencies for and if you will need to execute the tests (tests) and build the documentation (docs):

.. code-block:: bash

   git clone https://github.com/owkin/FLamby.git
   cd FLamby
   make enable=option_name install
   conda activate flamby

where ``option_name`` can be one of the following:
cam16, heart, isic2019, ixi, kits19, lidc, tcga, docs, tests

if you want to use more than one option you can do it using comma
(\ **WARNING:** there should be no space after ``,``\ ), eg:

.. code-block:: bash

   git clone https://github.com/owkin/FLamby.git
   cd FLamby
   make enable=cam16,kits19,tests install
   conda activate flamby

Be careful, each command tries to create a conda environment named flamby therefore make install will fail if executed
numerous times as the flamby environment will already exist. Use make update as explained in the next section if you decide to
use more datasets than intended originally.

Update environment
^^^^^^^^^^^^^^^^^^

Use the following command if new dependencies have been added, and you want to update the environment for additional datasets:

.. code-block:: bash

   make update

or you can use ``enable`` option:

.. code-block:: bash

   make enable=cam16 update

In case you don't have the ``make`` command (e.g. Windows users)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can install the environment by running:

.. code-block:: bash

   git clone https://github.com/owkin/FLamby.git
   cd FLamby
   conda env create -f environment.yml
   conda activate flamby
   pip install -e .[all_extra]

or if you wish to install the environment for only one or more datasets, tests or documentation:

.. code-block:: bash

   git clone https://github.com/owkin/FLamby.git
   cd FLamby
   conda env create -f environment.yml
   conda activate flamby
   pip install -e .[option_name]

where ``option_name`` can be one of the following:
cam16, heart, isic2019, ixi, kits19, lidc, tcga, docs, tests. If you want to use more than one option you can do it
using comma (',') (no space), eg:

.. code-block:: bash

   pip install -e .[cam16,ixi]

Datasets
^^^^^^^^

All datasets but one have to be downloaded and preprocessed to be accessible. Refer either to the :any:`quickstart` or
directly to the datasets' sections:

* :any:`fed_heart`.
* :any:`fed_ixi`.
* :any:`fed_isic`.
* :any:`fed_camelyon`.
* :any:`fed_lidc`.
* :any:`fed_kits19`.