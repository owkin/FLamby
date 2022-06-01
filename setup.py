from subprocess import call, check_call

from setuptools import find_packages, setup
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info
from setuptools.command.install import install

# Histolab has a dependency that requires options
histolab_dep_commands = [
    "pip",
    "install",
    "large-image-source-openslide",
    "--find-links",
    "https://girder.github.io/large_image_wheels",
]


class CustomInstallCommand(install):
    def run(self):
        install.run(self)
        command = call(histolab_dep_commands)
        assert command == 0


class CustomDevelopCommand(develop):
    def run(self):
        develop.run(self)
        command = call(histolab_dep_commands)
        assert command == 0


class CustomEggInfoCommand(egg_info):
    def run(self):
        egg_info.run(self)
        command = check_call(histolab_dep_commands)
        assert command == 0


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name="flamby",
    version="0.0.1",
    python_requires=">=3.7.0",
    license="MIT",
    classifiers=[
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    install_requires=[
        "albumentations",
        "argparse",
        "batchgenerators",
        "dask",
        "dicom-numpy",
        "dicom-numpy",
        "efficientnet-pytorch",
        "google-api-python-client",
        "google-auth-httplib2",
        "google-auth-oauthlib",
        "histolab",
        "lifelines",
        "matplotlib",
        "monai",
        "nibabel",
        "nibabel",
        "nnunet==1.7.0",
        "numpy",
        "openslide-python",
        "pandas",
        "pre-commit",
        "pydicom",
        "pydrive",
        "pytest",
        "requests",
        "scikit-learn",
        "scipy",
        "setuptools==59.5.0",
        "sphinx",
        "tensorboard",
        "torch",
        "torchvision",
        "tqdm",
        "wget",
        "xlrd",
        ],
    extras_require={},
    description="FLamby: A cross-silo Federated Learning Benchmark.",
    long_description=long_description,
    author="FL-datasets team",
    author_email="jean.du-terrail@owkin.com",
    packages=find_packages(),
    include_package_data=True,
    cmdclass={
        "install": CustomInstallCommand,
        "develop": CustomDevelopCommand,
        "egg_info": CustomEggInfoCommand,
    },
)
