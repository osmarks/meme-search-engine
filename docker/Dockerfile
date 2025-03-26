# Base Image
FROM debian:trixie-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    python3 \
    python3-pip \
    nginx \
    curl \
    pkg-config \
    libssl-dev \
    nasm \
    meson \
    ninja-build \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libpostproc-dev \
    libsqlite3-dev \
    libzstd-dev \
    libjpeg-dev \
    libpng-dev \
    libwebp-dev \
    libavdevice-dev \
    libclang-dev \
    libopenblas-dev \
    nodejs \
    npm \
    cmake \
    rustup \
    dav1d \
    libdav1d-dev \
    && apt-get clean

# Set environment variables for Rust
ENV PATH="/root/.cargo/bin:${PATH}"

# Install nightly Rust and set it as default
RUN rustup install nightly \
    && rustup default nightly

# Update Rust to the latest version and confirm nightly is active
RUN rustup update && rustc --version

# Build and install FAISS from source
RUN git clone https://github.com/Enet4/faiss /tmp/faiss -b c_api_head \
    && cd /tmp/faiss \
    && sed -i "14i #include <cstdint>" faiss/Index.h \
    && cmake -B build -DFAISS_ENABLE_PYTHON=OFF -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_C_API=ON -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=OFF . \
    && cmake --build build --target install \
    && cp build/c_api/libfaiss_c.so /usr/local/lib/ \
    && cp build/faiss/libfaiss.so /usr/local/lib \
    && rm -rf /tmp/faiss

# Set environment variables for library paths
ENV PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:/usr/lib/x86_64-linux-gnu/pkgconfig"
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"

# Set working directory
WORKDIR /app

# Clone the repository
RUN git clone https://github.com/osmarks/meme-search-engine.git . && git checkout 5121de63890ca5e3024a4a630385a274acbb510a

# Build Rust backend
RUN cargo +nightly build --release --bin meme-search-engine

# Install Python dependencies
RUN pip install -r requirements.txt --break-system-packages

COPY config/frontend_config.json /app

# Build frontend
RUN cd clipfront2 && npm install && node src/build.js

# Configure nginx
RUN rm /etc/nginx/sites-enabled/default
COPY config/nginx.conf /etc/nginx/sites-enabled/default

# Expose necessary ports
EXPOSE 80 1707 1708

# Start all services
CMD ["bash", "-c", "\
    python3 clip_server.py /app/config/clip_server_config.json & \
    /app/target/release/meme-search-engine /app/config/backend_config.json & \
    nginx -g 'daemon off;'"]
