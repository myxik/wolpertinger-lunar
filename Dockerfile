FROM myxik/myxik_container:latest
RUN apt update && apt install -y curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    tmux \
    htop \
    gcc \
    xvfb \
    python-opengl \
    ffmpeg \
    swig \
    x11-xserver-utils

RUN pip install gym
RUN pip install box2d box2d-kengz

RUN apt update && apt install -y cmake libopenmpi-dev python3-dev zlib1g-dev

ENV PYTHONPATH "$(PYTHONPATH):/workspace"