.. FLamby documentation master file, created by
   sphinx-quickstart on Wed May 11 17:24:35 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to FLamby's documentation!
==================================

FLamby
######


.. image:: logo.png
   :scale: 50 %
   :align: center
  



FLamby [#flamby]_ is a repository regrouping several distributed datasets with natural splits \
whose aim is to facilitate benchmarking new cross-silo FL strategies on realistic problems.

You can find a link to the companion paper published at NeurIPS 2022 in the 
Dataset and Benchmark track `here <https://arxiv.org/abs/2210.04620>`_

If you use FLamby please consider citing it::

    @article{duterrail2022flamby,
      title={FLamby: Datasets and Benchmarks for Cross-Silo Federated Learning in Realistic Healthcare Settings},
      author={Ogier du Terrail, Jean and Ayed, Samy-Safwan and Cyffers, Edwige and Grimberg, Felix and He, Chaoyang and Loeb, Regis and Mangold, Paul and Marchand, Tanguy and Marfoq, Othmane and Mushtaq, Erum and others},
      journal={arXiv preprint arXiv:2210.04620},
      year={2022}
    }

Before jumping to the :any:`quickstart` , make sure FLamby is properly installed
following the steps highlighted in :any:`installation`

.. rubric:: Footnotes

.. [#flamby] `Federated Learning AMple Benchmark of Your cross-silo strategies`



.. toctree::
   :maxdepth: 0
   :caption: Installation
   
   installation


.. toctree::
   :maxdepth: 0
   :caption: Getting Started Instructions
   
   quickstart




.. toctree::
   :maxdepth: 0
   :caption: Code Documentation

   strategies
   datasets





Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
