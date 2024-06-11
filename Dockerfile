# Use a base image with necessary dependencies
FROM python:3.10

# Set non-interactive environment variable
ENV DEBIAN_FRONTEND=noninteractive

# Install python and java
RUN apt-get update && apt-get install -y \
    wget \
    sudo \
    python3-pip

# Create a new user named "kafka"
RUN useradd -m user && \
    echo "user ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Switch to the newly created user
USER user

# Set the working directory to the home directory of the kafka user
WORKDIR /home/user

RUN pip install --upgrade pip

RUN pip install tensorflow[and-cuda]==2.16.1

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin &&\
    sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600 &&\
    wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda-repo-wsl-ubuntu-12-4-local_12.4.1-1_amd64.deb &&\
    sudo dpkg -i cuda-repo-wsl-ubuntu-12-4-local_12.4.1-1_amd64.deb &&\
    sudo cp /var/cuda-repo-wsl-ubuntu-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/ &&\
    sudo apt-get update &&\
    sudo apt-get -y install cuda-toolkit-12-4
RUN rm cuda-repo-wsl-ubuntu-12-4-local_12.4.1-1_amd64.deb

ENV CUDNN_PATH="/home/user/.local/lib/python3.10/site-packages/nvidia/cudnn"
ENV LD_LIBRARY_PATH="/home/user/.local/lib/python3.10/site-packages/nvidia/cudnn/lib:/usr/local/cuda/lib64"

COPY requirements.txt /home/user

RUN pip install -r requirements.txt

RUN git clone http://github.com/niotrek/twm

WORKDIR /home/user/twm

CMD ["tail", "-f", "/dev/null"]