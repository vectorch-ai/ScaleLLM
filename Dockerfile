# set base image using nvidia cuda 12.1 for ubuntu 22.04
ARG BASE_IMAGE=nvcr.io/nvidia/cuda:12.1.0-devel-ubuntu22.04

# ---- Build ----
FROM $BASE_IMAGE as build

LABEL maintainer="mi@vectorch.com"

# install build tools
RUN apt-get update -q -y && \
    apt-get install -q -y \
    build-essential \
    ninja-build \
    cmake \
    ccache \
    python3-dev \
    zip \
    pkg-config \
    libssl-dev \
    libboost-all-dev \
    curl \
    git \
    wget

# install jemalloc (optional)
RUN cd /tmp && \
    wget -nc --no-check-certificate https://github.com/jemalloc/jemalloc/releases/download/5.3.0/jemalloc-5.3.0.tar.bz2 && \
    tar -xvf jemalloc-5.3.0.tar.bz2 && \
    (cd jemalloc-5.3.0 && \
        ./configure --enable-prof --disable-initial-exec-tls && \
        make -j$(nproc) && make install && \
        ldconfig)

# install rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH=$HOME/.cargo/bin:$PATH

WORKDIR /ScaleLLM

# copy code from host to container
COPY ./ ./

# build
RUN cmake -G Ninja -S . -B build
RUN cmake --build build --target scalellm --config Release -j$(nproc)

# install
RUN cmake --install build --prefix /app
RUN cp ./scripts/download_hf_models.py /app/download_hf_models.py
RUN cp ./scripts/entrypoint.sh /app/entrypoint.sh
RUN cp ./requirements.txt /app/requirements.txt

# ---- Production ----
FROM ubuntu:22.04 as runtime
WORKDIR /app

# copy artifacts from build
COPY --from=build /app ./

# install python3 and pip3
RUN apt-get update -q -y && \
    apt-get install -q -y \
    python3 \
    python3-pip && \
    apt-get -y autoremove && \
    apt-get clean

# install python dependencies
RUN pip3 install -r ./requirements.txt

# expose port for grpc
EXPOSE 8888

# expose port for http
EXPOSE 9999

# start the server
ENTRYPOINT [ "/app/entrypoint.sh" ]


