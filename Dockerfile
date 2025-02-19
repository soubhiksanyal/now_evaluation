FROM python:3.8-slim
ARG DEBIAN_FRONTEND=noninteractive

# Set up locale
RUN apt-get update && apt-get install -y locales \
    && locale-gen en_US.UTF-8 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    build-essential \
    ca-certificates \
    g++ \
    git \
    libglfw3-dev \
    libgles2-mesa-dev \
    libglib2.0-0 \
    nano \
    sudo \
    libboost-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set user and group
RUN groupadd -g 555 devgroup && \
    useradd -l -u 556 -g devgroup -m -s /bin/bash devuser && \
    echo devuser ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/devuser && \
    chmod 0440 /etc/sudoers.d/devuser

# Switch to non-root user
USER devuser
WORKDIR /home/devuser

# Copy code into container
COPY . /home/devuser/now_evaluation

# Clone and set up projects
RUN git clone https://github.com/MPI-IS/mesh /home/devuser/mesh && \
    cd /home/devuser/mesh && \
    git checkout 49e70425cf373ec5269917012bda2944215c5ccd && \
    sed -i 's/--install-option/--config-settings/g' Makefile && \
    make all

RUN git clone https://github.com/Rubikplayer/flame-fitting /home/devuser/flame-fitting && \
    cd /home/devuser/flame-fitting && \
    git checkout ca806ce13a8964231136bd226bf3255fc2e476de && \
    cd /home/devuser && \
    cp -r flame-fitting/smpl_webuser now_evaluation/smpl_webuser && \
    cp -r flame-fitting/sbody now_evaluation/sbody

RUN git clone https://gitlab.com/libeigen/eigen.git /home/devuser/eigen && \
    cd /home/devuser/eigen && \
    git checkout 3.4.0 && \
    cp -r /home/devuser/eigen /home/devuser/now_evaluation/sbody/alignment/mesh_distance/eigen

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r /home/devuser/now_evaluation/requirements.txt && \
    cd /home/devuser/now_evaluation/sbody/alignment/mesh_distance && \
    sed -i 's/\.\/eigen/\/home\/devuser\/now_evaluation\/sbody\/alignment\/mesh_distance\/eigen/g' setup.py && \
    make

RUN pip install jupyterlab "numpy==1.23" tqdm

# Persist command history across sessions
RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/home/devuser/.history/.bash_history" \
    && mkdir -p /home/devuser/.history \
    && touch /home/devuser/.history/.bash_history \
    && echo "$SNIPPET" >> "/home/devuser/.bashrc"

RUN git clone --depth 1 https://github.com/junegunn/fzf.git /home/devuser/.fzf && \
    /home/devuser/.fzf/install --all

# Set entrypoint
ENTRYPOINT ["python", "/home/devuser/now_evaluation/compute_error.py", "--dataset_folder", "/dataset", "--predicted_mesh_folder", "/preds"]
