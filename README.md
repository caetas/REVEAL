# REVEAL

[![Python](https://img.shields.io/badge/python-3.10+-informational.svg)]()
[![documentation](https://img.shields.io/badge/docs-mkdocs%20material-blue.svg?style=flat)](https://mkdocstrings.github.io)
[![wandb](https://img.shields.io/badge/tracking-wandb-blue)](https://wandb.ai/site)

A short description of the project. No quotes.

## Prerequisites

You will need:

- `python` (see `pyproject.toml` for full version)
- `Git`
- `uv`
- a `.secrets` file with the required secrets and credentials
- load environment variables from `.env`
- `Weights & Biases` account

## Installation

Clone this repository (requires git ssh keys)

    git clone --recursive <ssh link>
    cd reveal

### Using uv

Create the environment and install the dependencies:

    uv sync --python 3.12

#### Activate the environment on Linux

You can activate the environment with:

    source .venv/bin/activate

You might be required to run the following command once to setup the automatic activation of the conda environment and the virtualenv:

    direnv allow

Feel free to edit the [`.envrc`](.envrc) file if you prefer to activate the environments manually.

#### Activate the environment on Windows

You can activate the environment with:

    .venv-dev/Scripts/Activate.ps1

### Using Docker or Apptainer

Create a `.secrets` file and add your Weights & Biases API Key:

    WANDB_API_KEY = <your-wandb-api-key>

#### Docker

Create the image using the provided [`Dockerfile`](Dockerfile)

    docker build --tag reveal .

Or download it from the Hub:

    docker pull docker://ocaetas/reveal

Then run the script [`job_docker.sh`](scripts/job_docker.sh) that will execute [`main.sh`](scripts/main.sh):

    cd scripts
    bash job_docker.sh

To access the shell, please run:

    docker run --rm -it --gpus all --ipc=host --env-file .env -v $(pwd)/:/app/ reveal bash

#### Apptainer

Convert the Docker Image to a `.sif` file:

    apptainer pull reveal.sif docker://ocaetas/reveal

Then run the script [`job_apptainer.sh`](scripts/job_apptainer.sh) that will execute [`main.sh`](scripts/main.sh):
    
    cd scripts
    bash job_apptainer.sh

To access the shell, please run:

    apptainer shell --nv --env-file .env --bind $(pwd)/:/app/ reveal.sif

**Add the flag `--nvccli` if you are using WSL.**

**Note: Edit the [`main.sh`](scripts/main.sh) script if you want to train a different model.**

## Documentation

Full documentation is available here: [`docs/`](docs).

## License

This project is licensed under the terms of the `MIT` license.
See [LICENSE](LICENSE) for more details.

## Citation

If you publish work that uses REVEAL, please cite REVEAL as follows:

```bibtex
@misc{REVEAL,
  author = {TU/e},
  title = {A short description of the project. No quotes.},
  year = {2026},
}
```
