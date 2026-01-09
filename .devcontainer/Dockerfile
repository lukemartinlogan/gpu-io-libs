FROM nvidia/cuda:12.6.0-devel-ubuntu24.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install base dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc-13 \
    g++-13 \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    pkg-config \
    gdb \
    sudo \
    libelf-dev \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Rename existing ubuntu user to iowarp (base image has ubuntu with UID 1000)
RUN usermod -l iowarp -d /home/iowarp -m ubuntu \
    && groupmod -n iowarp ubuntu \
    && echo "iowarp ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Switch to iowarp user
USER iowarp
WORKDIR /home/iowarp

# Install Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /home/iowarp/miniconda3 \
    && rm miniconda.sh \
    && /home/iowarp/miniconda3/bin/conda init bash \
    && /home/iowarp/miniconda3/bin/conda config --add channels conda-forge \
    && /home/iowarp/miniconda3/bin/conda config --set channel_priority strict

# Accept conda ToS and install dependencies
RUN /home/iowarp/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && /home/iowarp/miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
    && /home/iowarp/miniconda3/bin/conda install -y \
    boost \
    hdf5 \
    yaml-cpp \
    zeromq \
    cppzmq \
    cereal \
    catch2 \
    && /home/iowarp/miniconda3/bin/conda clean -ya

# Add conda to PATH for build steps
USER root
ENV PATH="/home/iowarp/miniconda3/bin:${PATH}"

# Clone and build iowarp core (context-transport-primitives)
RUN cd /tmp \
    && git clone https://github.com/iowarp/core.git -b 74-fix-context-transport-primitives-for-the-gpu \
    && cd core \
    && mkdir -p build && cd build \
    && cmake ../ --preset cuda-debug \
        -DCMAKE_CUDA_ARCHITECTURES=89 \
        -DHSHM_ENABLE_TESTS=OFF \
        -DHSHM_ENABLE_BENCHMARKS=OFF \
        -DCHIMAERA_ENABLE_TESTS=OFF \
        -DCHIMAERA_ENABLE_BENCHMARKS=OFF \
        -DCMAKE_PREFIX_PATH="/home/iowarp/miniconda3" \
    && make -j$(nproc) \
    && make install \
    && ldconfig \
    && cd /tmp && rm -rf core

# Switch back to iowarp user
USER iowarp

# Add conda and CUDA to bashrc
RUN echo '' >> /home/iowarp/.bashrc \
    && echo '# Conda initialization' >> /home/iowarp/.bashrc \
    && echo 'eval "$(/home/iowarp/miniconda3/bin/conda shell.bash hook)"' >> /home/iowarp/.bashrc \
    && echo '' >> /home/iowarp/.bashrc \
    && echo '# CUDA environment' >> /home/iowarp/.bashrc \
    && echo 'export CUDA_HOME=/usr/local/cuda' >> /home/iowarp/.bashrc \
    && echo 'export PATH=/usr/local/cuda/bin:$PATH' >> /home/iowarp/.bashrc

WORKDIR /workspace

CMD ["/bin/bash"]
