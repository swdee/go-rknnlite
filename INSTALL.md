
# How to Install

## Dependencies

go-rknnlite has a number of dependencies that are required.

The [rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2) must be installed on
your system with C header files and libraries available in the system path, 
eg: `/usr/include/rknn_api.h` and `/usr/lib/librknnrt.so`.  If your using an
official OS image provided by your SBC vendor these files probably already exist.

[GoCV](https://gocv.io/) is used for image processing.  Depending on GoCV's configuration this
has subdependencies such as OpenCV, FFMpeg, GStreamer etc which make for a complicated
build process. You can either follow [GoCV's installation instructions](https://github.com/hybridgroup/gocv/tree/release#how-to-install)
or follow our instructions below.


## Installation Options

We provide the following installation options.

* [Docker Image](#docker-install) - Use a prebuilt docker container
* Manual Installation - Step by step instructions you can manually execute on CLI
  * [Debian 12 (Bookworm)](#debian-12-bookworm)

You can install on other Linux OS's, the Manual Installation instructions can assist 
you in that process but some variation maybe needed.


## Docker Install

A prebuilt [docker image](https://hub.docker.com/repository/docker/swdee/go-rknnlite/general) 
is available as `swdee/go-rknnlite:latest` which contains a Debian bookworm OS base with
Golang, OpenCV, and GoCV configured.

You can use this image to run your own application or the go-rknnlite examples.  For example
to run the [MobileNet Demo](example/mobilenet) use the following commands.

```
cd example/mobilenet

docker run --rm \
  --device /dev/dri:/dev/dri \
  -v "$(pwd):/go/src/app" \
  -v "$(pwd)/../data:/go/src/data" \
  -v "/usr/include/rknn_api.h:/usr/include/rknn_api.h" \
  -v "/usr/lib/librknnrt.so:/usr/lib/librknnrt.so" \
  -w /go/src/app \
  swdee/go-rknnlite:latest \
  go run mobilenet.go
```

An explanation of each parameter in the docker command is as follows;

| Parameter                                            | Description                                                     |
|------------------------------------------------------|-----------------------------------------------------------------| 
| --device /dev/dri:/dev/dri                           | Pass the NPU device through into the container                  |
| -v "$(pwd):/go/src/app"                              | Mount the current Go application source code into the container |
| -v "$(pwd)/../data:/go/src/data"                     | Mount the example/data files into the container                 |
| -v "/usr/include/rknn_api.h:/usr/include/rknn_api.h" | Include the rknn-toolkit2 header files                          |
| -v "/usr/lib/librknnrt.so:/usr/lib/librknnrt.so"     | Include the rknn-toolkit2 shared library                        |
| -w /go/src/app                                       | Set the working directory in docker container                   |
| swdee/go-rknnlite:latest                             | Use the prebuilt go-rknnlite docker image                       |
| go run mobilenet.go                                  | Run the mobilenet.go demo                                       |


To view the OpenCV configuration in this prebuilt docker image run.
```
docker run --rm swdee/go-rknnlite:latest opencv_version --verbose
```


### Build your own Docker Image

You can also build your own docker image using the [Dockerfile](Dockerfile) in this directory, this allows
you to make customisations or change any versions of software to suit your application.

To change the Go version refer to the official Go build [tags here](https://hub.docker.com/_/golang )
and change the base image.

```
FROM golang:1.24.2-bookworm AS builder
```

You may also set the OpenCV and GoCV versions with the following variables.
```
ENV OPENCV_VERSION=4.11.0
ENV GOCV_VERSION=v0.41.0
```

Refer to the upstream project GoCV for version numbers as the OpenCV and GoCV 
versions must be compatible.


Run the following command to build your custom docker image.  It takes
approximately 25 minutes to build on RK3588 based SBC.
```
cd env/
docker build --progress=plain -t my-go-rknnlite .
```

Once successfully built run the image to confirm compiled program versions.
```
docker run --rm my-go-rknnlite
```

Output from the docker container.
```
Go version: go1.24.2
GoCV version: 0.41.0
OpenCV version: 4.11.0
```


## Manual Installation

## Debian 12 (Bookworm)

These manual instructions for Debian 12 (Bookworm) provide the steps to compile
a working environment.


### Install Go

Install latest Go package available on APT repository.   You can use other versions
at your own choice.
```
sudo apt install golang-1.23
```

Symlink the Go 1.13 binaries to system paths.
```
sudo ln -s /usr/lib/go-1.23/bin/go /usr/local/bin/go
sudo ln -s /usr/lib/go-1.23/bin/gofmt /usr/local/bin/gofmt
```

Confirm Go is installed and working.
```
go version
```


### Install Required Packages

Install required packages needed to build OpenCV which includes ffmpeg and
gstreamer support.
```
sudo apt-get install -y --no-install-recommends \
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
    libgstreamer-plugins-base1.0-dev    
```

Note: On Radxa's Debian image we also needed to install `librga-dev` for gstreamer
support to be built.  This is an additional library that provides 2D hardware
acceleration for Rockchip devices and is a part of Radxa's APT repository.
```
sudo apt install librga-dev
```


Clone OpenCV repository and checkout the desired version by
replacing the `OPENCV_VERSION` text with `4.11.0` for example.
```
cd /tmp

git clone https://github.com/opencv/opencv.git opencv 
cd opencv 
git checkout OPENCV_VERSION
```

Clone OpenCV Contrib and checkout the same version as OpenCV above.
```
cd /tmp
git clone https://github.com/opencv/opencv_contrib.git opencv_contrib
cd opencv_contrib 
git checkout OPENCV_VERSION 
```

Create build directory for OpenCV.
```
cd /tmp/opencv
mkdir build
cd build
```

Configure OpenCV.
```
cmake -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib/modules \
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
    ..
```

Check the configuration output to make sure all OpenCV features you want are included.


Compile and install.  It takes about 25 minutes to compile on a RK3588 based SBC.
```
make -j"$(nproc)" && sudo make install    
```

Check installation.
```
opencv_version --verbose
```

Clean up and remove files.
```
cd /tmp
rm -rf opencv/ opencv_contrib/
```



## Maintainer Notes Only

The following notes are for the go-rknnlite maintainer only, you do not need to do these.


### Building Docker Image


Building the docker image.
```
cd env/
docker build --progress=plain -t swdee/go-rknnlite:latest .
```

Check version after build is complete.
```
docker run --rm go-rknnlite:latest
```


### Uploading to Docker Hub 

Login to docker hub.
```
docker login
```

Push the tagged image as latest.
```
docker push swdee/go-rknnlite:latest
```

Create a version number for this latest image.
```
docker tag swdee/go-rknnlite:latest swdee/go-rknnlite:1.0.0
```

Push the versioned image.
```
docker push swdee/go-rknnlite:1.0.0
```

