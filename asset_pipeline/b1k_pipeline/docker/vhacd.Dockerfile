from ghcr.io/dask/dask:latest

RUN apt-get update && \
    apt-get install -y make build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/kmammou/v-hacd.git /root/vhacd && \
    cd /root/vhacd/app && \
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build && \
    mv build/TestVHACD /bin/TestVHACD