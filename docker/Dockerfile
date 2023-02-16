FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

RUN python3 -m pip install --upgrade pip
RUN pip3 install pyOpenSSL --upgrade
RUN pip3 install sagemaker-training

# Copies the training code inside the container
COPY ./code /opt/ml/code
WORKDIR /opt/ml/code

# Defines train.py as script entrypoint
ENV SAGEMAKER_PROGRAM cifar10.py
ENV PYTHONUNBUFFERED TRUE