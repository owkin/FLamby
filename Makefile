.ONESHELL:

CONDAPATH = $$(conda info --base)

# set default value
enable ?= all_extra
install:
	conda env create -f environment.yml
	${CONDAPATH}/envs/flamby/bin/pip install -e .[${enable}]

update:
	conda env update --prune -f environment.yml
	${CONDAPATH}/envs/flamby/bin/pip install -U .[${enable}]

clean:
	conda env remove --name flamby
