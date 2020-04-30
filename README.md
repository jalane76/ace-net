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
    │   ├── results                 <- Intermediate results files.
    │   ├── external                <- Data from third party sources.
    │   ├── interim                 <- Intermediate data that has been transformed.
    │   ├── processed               <- The final, canonical data sets for modeling.
    │   └── raw                     <- The original, immutable data dump.
    │
    ├── docker                      <- A Dockerfile and scripts for development in a container
    │   ├── apt-requirements.txt    <- Apt packages requirements file for building the docker image
    │   └── requirements.txt        <- The requirements file for reproducing the analysis environment, e.g.
    │                                  generated with `pip freeze > requirements.txt`
    │
    ├── docs                        <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── notebooks                   <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                                  the creator's initials, and a short `-` delimited description, e.g.
    │                                  `1.0-jqp-initial-data-exploration`.
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
    │   ├── data                    <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features                <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models                  <- Scripts to train models and then use trained models to make
    │   │   │                          predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization           <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini                     <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

--------

## Installation and Usage
 [ ] Add manual installation instructions.

### Prerequisites

* [Git](https://git-scm.com/downloads)
* [Docker](https://www.docker.com/products/docker-desktop)

For GPU acceleration:

* Use a Linux host for the Docker container.
* [Nvidia Container Toolkit](https://github.com/NVIDIA/nvidia-docker)

### Clone, Pull, and Run

Carefully consider where you will clone the repository since it will be bound as the container volume and so must be available to Docker to bind.

```
$ git clone https://github.com/jalane76/ace-net.git <path/to/ace-net/repo>
$ export ACE_NET_HOME=<path/to/ace-net/repo>

$ cd ${ACE_NET_HOME}/docker
$ docker pull jalane76/ace-net
$ chmod +x run-container.sh
$ ./run-container
```

The run script assumes that the host has been set up with GPU support.  Running CPU-only is as simple as editing [run-container.sh](docker/run-container.sh) to comment out the GPU support line and uncomment the no GPU support line.