from setuptools import find_packages, setup


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name="flamby",
    version="0.0.1",
    python_requires=">=3.7.0",
    license="MIT",
    classifiers=[
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Science/Research',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    install_requires=[
        "argparse",
        "matplotlib",
        "numpy",
        "pandas",
        "pre-commit",
        "scikit-learn",
        "scipy",
        "torch>=1.5",
        "tqdm",
        "pydrive",
        "openslide-python",
        "pydicom",
        "dask",
        "requests",
        "google-api-python-client",
        "google-auth-httplib2",
        "google-auth-oauthlib",

    ],
    extras_require={},
    description="FLamby: A cross-silo Federated Learning Benchmark.",
    long_description=long_description,
    author="FL-datasets team",
    author_email="jean.du-terrail@owkin.com",
    packages=find_packages(),
    include_package_data=True,
)