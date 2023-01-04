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

Team
----

This repository was created thanks to the contributions of many researchers and engineers. 
We list them in the order of the companion article, following the `CREDIT
framework <https://credit.niso.org/>`__: `Jean Ogier du
Terrail <https://github.com/jeandut>`__, `Samy-Safwan
Ayed <https://github.com/AyedSamy>`__, `Edwige
Cyffers <https://github.com/totilas>`__, `Felix
Grimberg <https://github.com/Grim-bot>`__, `Chaoyang
He <https://github.com/chaoyanghe>`__, `Régis
Loeb <https://github.com/regloeb>`__, `Paul
Mangold <https://github.com/pmangold>`__, `Tanguy
Marchand <https://github.com/tanguy-marchand>`__, `Othmane
Marfoq <https://github.com/omarfoq/>`__, `Erum
Mushtaq <https://github.com/ErumMushtaq>`__, `Boris
Muzellec <https://github.com/BorisMuzellec>`__, `Constantin
Philippenko <https://github.com/philipco>`__, `Santiago
Silva <https://github.com/sssilvar>`__, `Maria
Telenczuk <https://github.com/maikia>`__, `Shadi
Albarqouni <https://albarqouni.github.io/>`__, `Salman
Avestimehr <https://www.avestimehr.com/>`__, `Aurélien
Bellet <http://researchers.lille.inria.fr/abellet/>`__, `Aymeric
Dieuleveut <http://www.cmap.polytechnique.fr/~aymeric.dieuleveut/>`__,
`Martin Jaggi <https://people.epfl.ch/martin.jaggi>`__, `Sai Praneeth
Karimireddy <https://github.com/Saipraneet>`__, `Marco
Lorenzi <https://marcolorenzi.github.io/publications.html>`__, `Giovanni
Neglia <http://www-sop.inria.fr/members/Giovanni.Neglia/publications.htm>`__,
`Marc
Tommasi <http://researchers.lille.inria.fr/tommasi/#publications>`__,
`Mathieu Andreux <https://github.com/mandreux-owkin>`__.

Acknowledgements
----------------

FLamby's initiative was made possible thanks to the support of the following
institutions: - `Owkin <https://www.owkin.com>`__ -
`Inria <https://www.inria.fr>`__ - `Ecole
polytechnique <https://www.polytechnique.edu>`__ - `University of
California - Berkeley <https://www.berkeley.edu/>`__ - `University of
Southern California (USC) <https://www.usc.edu/>`__ -
`EPFL <https://www.epfl.ch>`__ - `Universitätsklinikum
Bonn <https://www.ukbonn.de/patient_innen/international/english/>`__

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
   :caption: Datasets informations
   
   fed_tcga_brca
   fed_heart
   fed_ixi
   fed_isic
   fed_camelyon
   fed_lidc
   fed_kits19

.. toctree::
   :maxdepth: 0
   :caption: Integration with FL-frameworks
   
   substra
   fedbiomed
   fedml

.. toctree::
   :maxdepth: 0
   :caption: FAQ

   faq


.. toctree::
   :maxdepth: 0
   :caption: Extending FLamby

   contributing


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
