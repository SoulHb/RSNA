FROM python:3.10

RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install ffmpeg libsm6 libxext6  -y

ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /app
COPY . .

RUN pip3 install -r ./requirements.txt

