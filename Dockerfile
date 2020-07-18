FROM python:3.8-buster

# Uncomment the following COPY line and the corresponding lines in the `RUN` command if you wish to
# include your requirements in the image itself. It is suggested that you only do this if your
# requirements rarely (if ever) change.
COPY requirements.txt .
#COPY model.py .

# Configure apt and install packages
RUN apt-get update && apt-get -y install git iproute2 procps lsb-release \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
RUN pip --no-cache-dir install -r requirements.txt
RUN pip install automl-alex