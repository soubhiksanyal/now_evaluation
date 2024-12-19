FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo
ENV LC_CTYPE="en_us.UTF-8"
ENV LC_ALL="en_us.UTF-8"
ENV LANG="en_us.UTF-8"

# Install system dependencies
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y wget \
    bzip2 \
    build-essential \
    manpages-dev \
    ca-certificates \
    g++ \
    git \
    libglfw3-dev \
    libgles2-mesa-dev \
    libglib2.0-0 \
    nano \
    locales \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Important: Remove the users group early to avoid conflicts
RUN delgroup users || true

# Set user and group - make sure to use a different group name than 'users'
RUN groupadd -g 555 devgroup && \
    useradd -l -u 556 -g devgroup -m -s /bin/bash devuser

# Install Miniforge in /opt
ENV CONDA_DIR=/opt/miniforge3
RUN wget -q --no-check-certificate https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && bash Miniforge3-Linux-x86_64.sh -b -p $CONDA_DIR \
    && rm Miniforge3-Linux-x86_64.sh

# Set permissions for the group to access Miniforge
RUN chgrp -R devgroup $CONDA_DIR && chmod -R g+rwx $CONDA_DIR

# Add conda to PATH and Set CUDA environment variables
ENV PATH="${CONDA_DIR}/bin:/usr/local/cuda/bin:${PATH}"
ENV TORCH_EXTENSIONS_DIR="/home/devuser/torch_extensions"
ENV CUDA_HOME="${CONDA_DIR}/envs/noweval"
ENV LD_LIBRARY_PATH="${CONDA_DIR}/envs/noweval/lib:${CONDA_DIR}/envs/noweval/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Initialize conda for bash and other shells
RUN $CONDA_DIR/bin/conda init bash

# Create python 3.8 conda env named noweval
RUN conda create -n noweval python=3.8

# Clean up
RUN mamba clean --all -f -y

# [Optional] Add sudo support. Omit if you don't need to install software after connecting.
RUN apt-get update \
    && apt-get install -y sudo libboost-dev \
    && echo devuser ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/devuser \
    && chmod 0440 /etc/sudoers.d/devuser

# Switch to non-root user
USER devuser
WORKDIR /home/devuser

# From now on, all commands will run within the conda environment
WORKDIR /workspaces/xriai-mediapipe-landmarks-to-flame
COPY --chown=devuser:devgroup . /workspaces/xriai-mediapipe-landmarks-to-flame

SHELL ["conda", "run", "--no-capture-output", "-n", "noweval", "/bin/bash", "-c"]

RUN git clone https://github.com/nlml/now_evaluation /home/devuser/now_evaluation

RUN pip install --upgrade pip \
    && pip install -r /home/devuser/now_evaluation/requirements.txt

RUN git clone https://github.com/MPI-IS/mesh /home/devuser/mesh \
    && cd /home/devuser/mesh \
    && git checkout 49e70425cf373ec5269917012bda2944215c5ccd \
    && sed -i 's/--install-option/--config-settings/g' Makefile \
    && make all

RUN git clone https://github.com/Rubikplayer/flame-fitting /home/devuser/flame-fitting \
    && cd /home/devuser/flame-fitting \
    && git checkout ca806ce13a8964231136bd226bf3255fc2e476de \
    && cd /home/devuser \
    && cp -r flame-fitting/smpl_webuser now_evaluation/smpl_webuser \
    && cp -r flame-fitting/sbody now_evaluation/sbody

RUN git clone https://gitlab.com/libeigen/eigen.git /home/devuser/eigen \
    && cd /home/devuser/eigen \
    && git checkout 3.4.0 \
    && cp -r /home/devuser/eigen /home/devuser/now_evaluation/sbody/alignment/mesh_distance/eigen

# replace ./eigen with /home/devuser/now_evaluation/sbody/alignment/mesh_distance/eigen in /home/devuser/now_evaluation/sbody/alignment/mesh_distance/setup.py
RUN cd /home/devuser/now_evaluation/sbody/alignment/mesh_distance \
    && sed -i 's/\.\/eigen/\/home\/devuser\/now_evaluation\/sbody\/alignment\/mesh_distance\/eigen/g' setup.py \
    && make

RUN pip install jupyterlab "numpy<1.24" tqdm

# Persist command history across sessions
ARG USERNAME=devuser
RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/home/devuser/.history/.bash_history" \
    && mkdir -p /home/devuser/.history \
    && touch /home/devuser/.history/.bash_history \
    && echo "$SNIPPET" >> "/home/$USERNAME/.bashrc"

RUN git clone --depth 1 https://github.com/junegunn/fzf.git /home/devuser/.fzf \
    && /home/devuser/.fzf/install --all

# Ensure the conda environment is activated on container startup
ENTRYPOINT ["bash", "-c", "source /opt/miniforge3/etc/profile.d/conda.sh && conda activate noweval && exec \"$@\"", "--"]

# Specify the default command
CMD ["python", "/home/devuser/now_evaluation/compute_error.py", "--dataset_folder", "/dataset", "--predicted_mesh_folder", "/preds"]
