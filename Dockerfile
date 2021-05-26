FROM python:3.6.13-slim-buster

RUN apt update && apt-get install -y gfortran libopenblas-dev liblapack-dev tk

WORKDIR /halomorp

COPY requirements.txt /halomorp/requirements.txt

RUN pip install -r requirements.txt

EXPOSE 8888

COPY . /halomorp
