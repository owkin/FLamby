install:
	conda env create -f environment.yml

update:
	conda env update --prune -f environment.yml

clean:
	conda env remove --name flamby