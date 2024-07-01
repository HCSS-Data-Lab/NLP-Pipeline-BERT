########################################################################
# Base
########################################################################
FROM mcr.microsoft.com/vscode/devcontainers/python:3.9 as base

RUN sudo apt-get update
RUN sudo apt-get install python3-distutils -y
RUN sudo apt-get install python3-dev -y

RUN curl -O https://bootstrap.pypa.io/get-pip.py
RUN python3.9 get-pip.py
RUN pip install --upgrade pip

########################################################################
# Development
########################################################################
FROM base as dev

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

########################################################################
# Jupyter Notebook
########################################################################
FROM jupyter/base-notebook:python-3.9.13 as hcss-jupyternotebook-bert

USER root
RUN apt-get update
RUN apt-get install htop -y
RUN apt-get install python3-distutils -y
RUN apt-get install python3-dev -y
RUN apt-get install gcc -y
RUN apt-get install curl -y

USER jovyan
RUN curl -O https://bootstrap.pypa.io/get-pip.py
RUN python3.9 get-pip.py
RUN pip install --upgrade pip

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt
