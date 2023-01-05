Containerized execution
=======================

A good step towards float-perfect reproducibility in your future
benchmarks is to use docker. We give a base docker image and examples
containing dataset download and benchmarking. For
:any:`fed_heart`,
``cd`` to the flamby dockers folder, replace ``myusername`` and
``mypassword`` with your git credentials (OAuth token) in the command
below and run:

::

    docker build -t flamby-heart -f Dockerfile.base --build-arg DATASET_PREFIX="heart" --build-arg GIT_USER="myusername" --build-arg GIT_PWD="mypassword" .
    docker build -t flamby-heart-benchmark -f Dockerfile.heart .
    docker run -it flamby-heart-benchmark

If you are convinced you will use many datasets with docker, build the
base image using ``all_extra`` option for flamby's install, you will be
able to reuse it for all datasets with multi-stage build:

::

    docker build -t flamby-all -f Dockerfile.base --build-arg DATASET_PREFIX="all_extra" --build-arg GIT_USER="myusername" --build-arg GIT_PWD="mypassword" .
    # modify Dockerfile.* line 1 to FROM flamby-all by replacing * with the dataset name of the dataset you are interested in
    # Then run the following command replacing * similarly
    #docker build -t flamby-* -f Dockerfile.* .
    #docker run -it flamby-*-benchmark

Checkout ``Dockerfile.tcga``. Similar dockerfiles can be theoretically
easily built for the other datasets as well by replicating instructions
found in each dataset folder following the model of
``Dockerfile.heart``. Note that for bigger datasets execution can be
prohibitively slow and docker can run out of time/memory.
