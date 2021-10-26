FROM continuumio/miniconda3

ADD env.yaml /tmp/env.yaml

RUN conda env create -f /tmp/env.yaml