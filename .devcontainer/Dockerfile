FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install base dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    g++-13 \
    wget \
    curl \
    git \
    ninja-build \
    ccache \
    gdb \
    clang-format \
    vim \
    sudo \
    zsh \
    locales \
    && rm -rf /var/lib/apt/lists/*

# Set GCC 13 as default (C++20 support)
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100

# Install CMake 3.31.2
ARG CMAKE_VERSION=3.31.2
RUN wget -q https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh && \
    mkdir -p /opt/cmake && \
    sh cmake-${CMAKE_VERSION}-linux-x86_64.sh --prefix=/opt/cmake --skip-license && \
    ln -sf /opt/cmake/bin/cmake /usr/local/bin/cmake && \
    ln -sf /opt/cmake/bin/ctest /usr/local/bin/ctest && \
    ln -sf /opt/cmake/bin/cpack /usr/local/bin/cpack && \
    rm cmake-${CMAKE_VERSION}-linux-x86_64.sh

# Set up locale
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

# Create non-root user matching the remoteUser in devcontainer.json
ARG USERNAME=iowarp
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -s /bin/zsh && \
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

# Install Oh My Zsh for the user
USER $USERNAME
RUN sh -c "$(wget -O- https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

# Switch back to root for any final setup
USER root

# Display versions on container start
RUN echo '#!/bin/bash\n\
echo "IOWarp Development Environment"\n\
echo "=============================="\n\
gcc --version | head -n1\n\
g++ --version | head -n1\n\
cmake --version | head -n1\n\
if command -v nvcc &> /dev/null; then\n\
    nvcc --version | grep release\n\
else\n\
    echo "NVCC not available (CPU-only mode)"\n\
fi\n\
echo ""\n\
' > /usr/local/bin/show-versions && chmod +x /usr/local/bin/show-versions

# Set the default user
USER $USERNAME
WORKDIR /workspaces/gpu-io-libs

# Set default shell to zsh
ENV SHELL=/bin/zsh
