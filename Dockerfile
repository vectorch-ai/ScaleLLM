# ---- Build ----
FROM nvcr.io/nvidia/cuda:12.1.0-devel-ubuntu22.04 as build

ARG VERSION=main

LABEL maintainer="mi@vectorch.com"

# install build tools
RUN apt-get update -q -y && \
    apt-get install -q -y \
    build-essential \
    ninja-build \
    cmake \
    python3-dev \
    zip \
    pkg-config \
    libssl-dev \
    curl \
    git 

# install rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH=$HOME/.cargo/bin:$PATH

WORKDIR /ScaleLLM

# copy code from host to container
COPY ./ ./

# build
RUN cmake -G Ninja -S . -B build
RUN cmake --build build --target all --config Release -j$(nproc)

# install
RUN cmake --install build --prefix /app
RUN cp ./entrypoint.sh /app/entrypoint.sh
RUN cp ./requirements.txt /app/requirements.txt

# ---- Production ----
FROM nvcr.io/nvidia/cuda:12.1.0-base-ubuntu22.04 as runtime
WORKDIR /app

# copy artifacts from build
COPY --from=build /app ./

# install python and dependencies
RUN apt-get update -q -y && \
    apt-get install -q -y \
    python3 \
    python3-pip

# install python dependencies
RUN pip3 install -r ./requirements.txt

# expose port for grpc
EXPOSE 8888

# expose port for http
EXPOSE 9999

# start the server
ENTRYPOINT [ "./entrypoint.sh" ]


