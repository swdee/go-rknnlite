# Stage 1: Build OpenCV from source
FROM golang:1.24.2-bookworm AS builder

# Install build dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    git \
    wget \
    curl \
    pkg-config \
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libtbbmalloc2 \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    gfortran \
    openexr \
    libatlas-base-dev \
    python3-dev \
    python3-numpy \
    libtbb-dev \
    libdc1394-dev \
    libharfbuzz-dev \
    libfreetype6-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the OpenCV version to build (adjust as needed)
ENV OPENCV_VERSION=4.11.0

# Clone OpenCV and opencv_contrib repositories and checkout the desired version
RUN git clone https://github.com/opencv/opencv.git /opencv && \
    cd /opencv && git checkout ${OPENCV_VERSION} && \
    git clone https://github.com/opencv/opencv_contrib.git /opencv_contrib && \
    cd /opencv_contrib && git checkout ${OPENCV_VERSION}

# Build OpenCV with opencv_contrib modules
RUN mkdir -p /opencv/build && cd /opencv/build && \
    cmake -D CMAKE_BUILD_TYPE=Release \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D OPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules \
          -D BUILD_DOCS=OFF \
          -D BUILD_EXAMPLES=OFF \
          -D BUILD_TESTS=OFF \
          -D BUILD_PERF_TESTS=ON \
          -D BUILD_opencv_java=OFF \
          -D BUILD_opencv_python=NO \
          -D BUILD_opencv_python2=NO \
          -D BUILD_opencv_python3=NO \
          -D WITH_JASPER=OFF \
          -D WITH_TBB=ON \
          -D OPENCV_GENERATE_PKGCONFIG=ON \
          -D WITH_FREETYPE=ON \
          -D WITH_GSTREAMER=ON \
          .. && \
    make -j"$(nproc)" && \
    make install || (echo "CMake failed, printing logs:" && \
        cat CMakeFiles/CMakeError.log && \
        cat CMakeFiles/CMakeOutput.log && exit 1)

# Stage 2: Create the final image with Go, precompiled OpenCV, and GoCV
FROM golang:1.24.2-bookworm

# Install runtime dependencies needed by OpenCV (e.g., GStreamer and FFmpeg libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgtk-3-0 \
    libtbb12 \
    ffmpeg \
    libwebpdemux2 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed OpenCV libraries and headers from the builder stage
COPY --from=builder /usr/local /usr/local

# Ensure pkg-config can locate OpenCV's .pc files
ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH

# Set LD_LIBRARY_PATH so that the dynamic linker can find OpenCV libraries
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Set working directory
WORKDIR /go/src/versiontest

# If a go.mod doesn't already exist, initialize a temporary module.
# (This helps when running "go get" so that a module context is present.)
RUN go mod init tmp || true

# Set the GoCV version to build (adjust as needed)
ENV GOCV_VERSION=v0.41.0

# Download the GoCV package and build a sample application to verify integration
RUN go get -v gocv.io/x/gocv@${GOCV_VERSION} && \
    echo 'package main; import ("fmt"; "runtime"; "gocv.io/x/gocv"); func main() { fmt.Println("Go version:", runtime.Version()); fmt.Println("GoCV version:", gocv.Version()); fmt.Println("OpenCV version:", gocv.OpenCVVersion()) }' > main.go && \
    go build -o version main.go

# By default, run the sample application
CMD ["./version"]

