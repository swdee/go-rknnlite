
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
git clone --depth=1 https://github.com/swdee/go-rknnlite-data.git data
```


Command line Usage.
```
$ go run pool.go -h

Usage of /tmp/go-build1518837318/b001/exe/pool:
  -c string
        CPU Affinity, run on [fast|slow] CPU cores (default "fast")
  -d string
        A directory of images to run inference on (default "../data/imagenet/")
  -m string
        RKNN compiled model file (default "../data/models/rk3588/mobilenet_v1-rk3588.rknn")
  -p string
        Rockchip CPU Model number [rk3562|rk3566|rk3568|rk3576|rk3582|rk3582|rk3588] (default "rk3588")
  -q    Run in quiet mode, don't display individual inference results
  -r int
        Repeat processing image directory the specified number of times, use this if you don't have enough images (default 1)
  -s int
        Size of RKNN runtime pool, choose 1, 2, 3, or multiples of 3 (default 1)
```

To run the example pool using 3 Runtimes in the pool and downloaded data.  Make
sure you replace the `<platform>` with your Rockchip model, eg: `rk3588`.
```
cd example/pool/
go run pool.go -q -s 3 -r 4 -p <platform>
```

Example summary.
```
Running...
Processed 4000 images in 9.61513744s, average inference per image is 2.40ms
```

### Pool Multiples

When selecting the number of Runtimes to initialize the pool with select 1, 2, 3, or
a multiple of 3 to spread them across all three NPU cores on the rk358x series.

The rk3576 only has two NPU cores, so select multiples for 2.


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
  go run ./example/pool/pool.go -q -s 3 -r 4 -p <platform>
```




## Benchmarks

For the [MobileNet-v1 model](../mobilenet/) we achieve the following average inference times
for the number of Runtimes in the Pool processing 8000 images.

Note that the more Runtimes created in the pool the more memory is needed for 
each instance of the Model loaded.


### rk3588

Results for the rk3588 with 6 TOPS three core NPU operating at 1Ghz.

| Number of Runtimes | Execution Time | Core Saturation      | Average Inference Time Per Image |
| ---- |----------------|----------------------|----------------------------------|
| 1 | 61.82s         | ~35% core saturation | 7.73ms                           |
| 2 | 27.94s         | ~35% core saturation | 3.49ms                           |
| 3 | 18.50s         | ~35% core saturation | 2.31ms                           |
| 6 | 12.18s         | ~60% core saturation | 1.52ms                           |
| 9 | 9.51s          | ~74% core saturation | 1.19ms                           |
| 12 | 9.02s          | ~80% core saturation | 1.13ms                           |
| 15 | 8.75s          | ~80% core saturation | 1.09ms                           |


Core saturation peaks around 80% across all three cores so going beyond 9 Runtimes
has diminishing returns.   



### rk3576

Results for the rk3576 with 6 TOPS two core NPU operating at 950Mhz.


| Number of Runtimes | Execution Time | Core Saturation      | Average Inference Time Per Image |
|--------------------|----------------|----------------------|----------------------------------|
| 1                  | 94.06s         | ~30% core saturation | 11.76ms                          |
| 2                  | 42.60s         | ~24% core saturation | 5.33ms                           |
| 4                  | 23.53s         | ~37% core saturation | 2.94ms                           |
| 6                  | 17.80s         | ~48% core saturation | 2.23ms                           |
| 8                  | 14.98s         | ~58% core saturation | 1.87ms                           |
| 10                 | 14.11s         | ~62% core saturation | 1.76ms                           |
| 12                 | 13.70s         | ~64% core saturation | 1.71ms                           |



### rk3566 

Results for the rk3566 with 1 TOPS single core NPU operating at 900Mhz.


| Number of Runtimes | Execution Time | Core Saturation      | Average Inference Time Per Image |
|--------------------|----------------|----------------------|----------------------------------|
| 1                  | 177.66s        | ~29% core saturation | 22.21ms                          |
| 2                  | 87.25s         | ~43% core saturation | 10.91ms                          |
| 3                  | 68.06s         | ~58% core saturation | 8.51ms                           |
| 4                  | 60.60s         | ~70% core saturation | 7.58ms                           |
| 5                  | 59.40s         | ~70% core saturation | 7.43ms                           |
| 6                  | 53.81s         | ~76% core saturation | 6.73ms                           |


