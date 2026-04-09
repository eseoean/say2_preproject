FROM continuumio/miniconda3:latest

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

# Install rdkit via conda-forge + pip packages
RUN conda install -y -c conda-forge python=3.11 rdkit=2023.09 \
    && conda clean -afy

COPY requirements-fe.txt .
RUN pip install --no-cache-dir -r requirements-fe.txt

# AWS CLI via pip (avoids binary compatibility issues on cross-platform builds)
RUN pip install --no-cache-dir awscli \
    && ln -sf "$(which aws)" /usr/local/bin/aws

COPY . /workspace/nextflow/
