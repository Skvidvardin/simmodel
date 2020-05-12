# pull official base image
FROM python:3.6.10-slim-buster

# set working dir
WORKDIR /home/ubuntu/simmodel/simmodel_app/

# copy project
COPY . /home/ubuntu/simmodel/simmodel_app/

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
