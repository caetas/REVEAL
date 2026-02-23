FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
	git \
	python3 \
	python3-pip \
	&& rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir uv

WORKDIR /app/
ENV UV_PROJECT_ENVIRONMENT="/opt/venv"

COPY pyproject.toml uv.lock /app/
RUN uv sync --frozen --no-dev --no-install-project --python 3.12
RUN mkdir /app/data
RUN mkdir /app/src
RUN mkdir /app/models

ENV PYTHONPATH="${PYTHONPATH}:/app/src/reveal"