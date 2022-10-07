# this base image works in both CPU and GPU enabled environments
FROM ghcr.io/substra/substra-tools:0.16.0-nvidiacuda11.6.0-base-ubuntu20.04-python3.9

# install dependencies
RUN pip3 install numpy==1.23.1 flamby torch==1.11.0

# add your metrics script to docker image
ADD metrics.py .

# define how script is run
ENTRYPOINT ["python3", "metrics.py", "--method-name", "score"]
