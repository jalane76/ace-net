# ace-net

## Overview

The purpose of this repo is twofold:

1. To reproduce some of the results from the paper [Neural Network Attributions: A Causal Perspective](https://arxiv.org/abs/1902.02302) and [repository](https://github.com/Piyushi-0/ACE).
2. To further explore using *average causal effect* (ACE) to analyze neural networks, particularly in adversarial settings.

## Project Organization

    ├── LICENSE
    ├── Makefile                    <- Makefile with commands like `make data` or `make train`
    ├── README.md                   <- The top-level README for developers using this project.
    ├── data (HIDDEN)               <- Hidden from Git, but files are in a public Google Drive (see below)
    │   ├── models                  <- Trained and serialized models, model predictions, or model summaries
    │   ├── processed               <- The final, canonical data sets for modeling.
    │   ├── raw                     <- The original, immutable data dump.
    │   ├── results                 <- Intermediate results files.
    │   └── viz                     <- Images generated for visualization.
    │ 
    ├── docker                      <- A Dockerfile and scripts for development in a container
    │   ├── apt-requirements.txt    <- Apt packages requirements file for building the docker image
    │   └── requirements.txt        <- The requirements file for reproducing the analysis environment
    │   └── requirements-freeze.txt <- The detailed requirements file generated with `pip freeze > requirements-freeze.txt`
    │
    ├── docs                        <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── references                  <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports                     <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures                 <- Generated graphics and figures to be used in reporting
    │
    │
    ├── setup.py                    <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                         <- Source code for use in this project.
    │   ├── __init__.py             <- Makes src a Python module
    │   │
    │   ├── attacks                 <- Scripts to generate attacks using ACE
    │   │
    │   ├── data                    <- Scripts to download or generate data
    │   │
    │   ├── model_analysis          <- Scripts to analyze trained models
    │   │
    │   ├── models                  <- Scripts to train models and then use trained models to make predictions
    │   │
    │   ├── stages                  <- DVC stage files for defining reproducible experiments
    │   │
    │   ├── tests                   <- Scripts to test utilities and algorithms
    │   │
    │   ├── utils                   <- Scripts for everything else
    │   │
    │   └── visualization           <- Scripts to create exploratory and results oriented visualizations
    │
    └── tox.ini                     <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

--------

## Installation and Usage
```diff
- TODO: Add manual installation instructions.
````
### Prerequisites

* [Git](https://git-scm.com/downloads)
* [Docker](https://www.docker.com/products/docker-desktop)

For GPU acceleration:

* Use a Linux host for the Docker container.
* [Nvidia Container Toolkit](https://github.com/NVIDIA/nvidia-docker)

### Clone, Pull, and Run

Carefully consider where you will clone the repository since it will be bound as the container volume and so must be available to Docker to bind.

```
$ git clone https://github.com/jalane76/ace-net.git 
$ export ACE_NET_HOME=/absolute/path/to/ace-net

$ cd ${ACE_NET_HOME}/docker
$ docker pull jalane76/ace-net
$ ./run-container
```

The run script assumes that the host has been set up with GPU support.  Running CPU-only is as simple as editing [run-container.sh](docker/run-container.sh) to comment out the GPU support line and uncomment the no GPU support line.