
# Multiple Runtime Pool

## Overview

The RK3588 has three NPU cores, by default when selecting `rknnlite.NPUCoreAuto` it will
run your RKNN Model on a single idle core.  Through the other options you can
specify which core to run on or used combined cores 0 & 1 with `rknnlite.NPUCore01` or
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
```
cd example/data/
./download.sh
```


Command line Usage.
```
$ go run pool.go -h

Usage of /tmp/go-build3261134608/b001/exe/pool:
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
go run pool.go -q -s 3
```

Example summary.
```
Running...
Processed 4000 images in 9.534736507s, average inference per image is 2.38ms
```

When selecting the number of Runtimes to initialize the pool with select 1, 2, 3, or
a multiple of 3 to spread them across all three NPU cores.



## Benchmarks

For an EfficentNet-Lite0 Model we achieve the following average inference times
for the number of Runtimes in the Pool.


| Number of Runtimes | Execution Time | Core Saturation | Average Inference Time Per Image |
| ---- | ---- | ---- | --- |
| 1 | 59.97s | ~30% core saturation | 7.91ms |
| 2 | 34.56s | ~30% core saturation | 4.55ms |
| 3 | 22.94s | ~30% core saturation | 3.02ms |
| 6 | 13.89s | ~48% core saturation | 1.83ms |
| 9 | 12.54s | ~54% core saturation | 1.65ms |
| 12 | 11.97s | ~57% core saturation | 1.57ms |
| 15 | 12.03s | ~58% core saturation | 1.58ms |


Core saturation peaks around 60% across all three cores so going beyond 9 Runtimes
has diminishing returns.   

Note that the more Runtimes created the more memory is needed for each instance
of the Model loaded.