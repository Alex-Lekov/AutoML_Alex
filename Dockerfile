FROM python:3.9-buster

# Uncomment the following COPY line and the corresponding lines in the `RUN` command if you wish to
# include your requirements in the image itself. It is suggested that you only do this if your
# requirements rarely (if ever) change.
COPY requirements.txt .

# Configure apt and install packages
RUN apt-get update \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
#    && apt-get -y install git iproute2 procps lsb-release \
RUN pip install --upgrade pip
RUN pip --no-cache-dir install -r requirements.txt
RUN pip install automl-alex