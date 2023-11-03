# ---- Build ----
FROM nvcr.io/nvidia/cuda:12.1.0-devel-ubuntu22.04 as build
WORKDIR /build

ARG VERSION=main

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
ENV source "$HOME/.cargo/env"

# enlist code
RUN git clone --recursive  --branch=${VERSION} https://github.com/vectorch-ai/ScaleLLM.git

WORKDIR /build/ScaleLLM

# build
RUN cmake -G Ninja -S . -B build
RUN cmake --build build --target all --config Release -j$(nproc)

# install
RUN cmake --install build --prefix /app
COPY ./entrypoint.sh /app/entrypoint.sh
COPY ./requirements.txt /app/requirements.txt

# ---- Production ----
FROM nvcr.io/nvidia/cuda:12.1.0-base-ubuntu22.04 as runtime
WORKDIR /app
# install python and dependencies
RUN apt-get update -q -y && \
    apt-get install -q -y \
    python3 \
    python3-pip

# copy artifacts from build
COPY --from=build /app ./

# install python dependencies
RUN pip3 install -r ./requirements.txt

# expose ports
EXPOSE 8888 9999

# start the server
ENTRYPOINT [ "./entrypoint.sh" ]


