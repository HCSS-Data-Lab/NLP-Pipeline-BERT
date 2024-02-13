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
