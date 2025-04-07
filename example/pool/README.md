
# Multiple Runtime Pool

## Overview

The RK3588 has three NPU cores, by default when selecting `rknnlite.NPUCoreAuto` it will
run your RKNN Model on a single idle core.  Through the other options you can
specify which core to run on or to use combined cores 0 & 1 with `rknnlite.NPUCore01` or
all three cores with `rknnlite.NPUCore012`.

You can monitor the NPU usage by running:
```
$ watch -n 1 cat /sys/kernel/debug/rknpu/load

NPU load:  Core0:  0%, Core1:  0%, Core2:  0%,
```

However these settings don't exhaust the NPU's processing capacity, typically
you will only see saturation of a single core around 30% when 
running a single Model.  For this reason running multiple instances of the same
Model allows us to use all NPU cores.


## Usage


First make sure you have downloaded the data files first.
You only need to do this once for all examples.

```
cd example/
git clone https://github.com/swdee/go-rknnlite-data.git data
```


Command line Usage.
```
$ go run pool.go -h

Usage of /tmp/go-build3261134608/b001/exe/pool:
  -c string
        CPU Affinity, run on [fast|slow] CPU cores (default "fast")
  -d string
        A directory of images to run inference on (default "../data/imagenet/")
  -m string
        RKNN compiled model file (default "../data/mobilenet_v1-rk3588.rknn")
  -q    Run in quiet mode, don't display individual inference results
  -r int
        Repeat processing image directory the specified number of times, use this if you don't have enough images (default 1)
  -s int
        Size of RKNN runtime pool, choose 1, 2, 3, or multiples of 3 (default 1)
```

To run the example pool using 3 Runtimes in the pool and downloaded data.
```
cd example/pool/
go run pool.go -q -s 3 -r 4
```

Example summary.
```
Running...
Processed 4000 images in 9.36881374s, average inference per image is 2.34ms
```

When selecting the number of Runtimes to initialize the pool with select 1, 2, 3, or
a multiple of 3 to spread them across all three NPU cores.


### Docker

To run the Pool example using the prebuilt docker image, make sure the data files have been downloaded first,
then run.
```
# from project root directory

docker run --rm \
  --device /dev/dri:/dev/dri \
  -v "$(pwd):/go/src/app" \
  -v "$(pwd)/example/data:/go/src/data" \
  -v "/usr/include/rknn_api.h:/usr/include/rknn_api.h" \
  -v "/usr/lib/librknnrt.so:/usr/lib/librknnrt.so" \
  -w /go/src/app \
  swdee/go-rknnlite:latest \
  go run ./example/pool/pool.go -q -s 3 -r 4
```




## Benchmarks

For an EfficentNet-Lite0 Model we achieve the following average inference times
for the number of Runtimes in the Pool processing 8000 images.


| Number of Runtimes | Execution Time | Core Saturation      | Average Inference Time Per Image |
| ---- |----------------|----------------------|----------------------------------|
| 1 | 57.21s         | ~35% core saturation | 7.15ms                            |
| 2 | 29.59s         | ~35% core saturation | 3.70ms                           |
| 3 | 20.46s         | ~35% core saturation | 2.56ms                           |
| 6 | 12.12s         | ~60% core saturation | 1.52ms                           |
| 9 | 10.01s         | ~74% core saturation | 1.25ms                           |
| 12 | 9.55s          | ~80% core saturation | 1.19ms                           |
| 15 | 9.36s          | ~80% core saturation | 1.17ms                           |


Core saturation peaks around 80% across all three cores so going beyond 9 Runtimes
has diminishing returns.   

Note that the more Runtimes created the more memory is needed for each instance
of the Model loaded.

Previously we achieved ~60% core saturation but through the use of CPU Affinity
and running this program on the fast Cortex-A76 cores only we can further
saturate the NPU cores to ~80%.
