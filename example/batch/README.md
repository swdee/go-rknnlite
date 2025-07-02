
# Batch Models

## Overview

Typically computer vision inference models have a single input tensor in 
the shape of `NHWC` such as `[1,224,224,3]`.  The rknn-toolkit2 allows you to 
build the model with Batch tensor inputs by setting the `rknn_batch_size` parameter 
in the following python conversion script.

```
rknn.build(do_quantization=do_quant, dataset=DATASET_PATH, rknn_batch_size=8)
```

This results in a .rknn model with modified tensor input dimensions of `[8,224,244,3]`.

When taking input from a video source frame-by-frame, the use of batching to process
frames has little use case, as your only dealing with a single frame to be
processed as soon as possible.   However batching can be useful if you have many
images to process at a single point in time, some examples of this could be;
 * Running YOLO object detection on a frame, then passing all detected objects 
   through a ReIdentification model in batches.
 * Some applications will buffer video frames and upon an external signal, it
   will then trigger the processing of those buffered frames as a batch.


## Batch Sizing

The NPU's in the different platforms RK356x, RK3576, and RK3588 have different
amounts of SRAM and NPU core numbers, so finding the optimal batch size for your
Model is critical.

A benchmarking tool has been created to test different batch sizes of your own
RKNN Models.  Use your python conversion script to compile the ONNX model to RKNN
with various `rknn_batch_size` values you would like to test.  Name those RKNN
Models using this format `<name>-batch{N1,N2,...,Nk}.rknn`. For example I wish
to test batch sizes of 1, 4, 8, and 16 of an OSNet model and have created the
following files and placed them in the directory `/tmp/models` on the host OS.
```
osnet-batch1.rknn
osnet-batch4.rknn
osnet-batch8.rknn
osnet-batch16.rknn
```

We can then pass all these Models to the benchmark using the `-m` argument in 
the format of `-m "/tmp/models/osnet-batch{1,4,8,16}"`.  

To run the benchmark of your models on the rk3588 or replace with your 
Platform model.
```
# from project root directory

go test -bench=BenchmarkBatchSize -benchtime=10s \
  -args -p rk3588 -m "/tmp/models/osnet-batch{1,4,8,16}.rknn"
```

Similarly using Docker we can mount the `/tmp/models` directory and run.
```
# from project root directory

docker run --rm \
  --device /dev/dri:/dev/dri \
  -v "$(pwd):/go/src/app" \
  -v "$(pwd)/example/data:/go/src/data" \
  -v "/usr/include/rknn_api.h:/usr/include/rknn_api.h" \
  -v "/usr/lib/librknnrt.so:/usr/lib/librknnrt.so" \
  -v "/tmp/models/:/tmp/models/" \
  -w /go/src/app \
  swdee/go-rknnlite:latest \
  go test -bench=BenchmarkBatchSize -benchtime=10s \ 
    -args -p rk3588 -m "/tmp/models/osnet-batch{1,4,8,16}"
```

Running the above benchmark command outputs the following results.

#### rk3588

```
BenchmarkBatchSize/Batch01-8                1897           8806025 ns/op                 8.806 ms/batch          8.806 ms/img
BenchmarkBatchSize/Batch04-8                 885          21555109 ns/op                21.55 ms/batch           5.389 ms/img
BenchmarkBatchSize/Batch08-8                 534          22335645 ns/op                22.34 ms/batch           2.792 ms/img
BenchmarkBatchSize/Batch16-8                 303          40253162 ns/op                40.25 ms/batch           2.516 ms/img
```

#### rk3576

```
BenchmarkBatchSize/Batch01-8                1312           8987117 ns/op                 8.985 ms/batch          8.985 ms/img
BenchmarkBatchSize/Batch04-8                 640          18836090 ns/op                18.83 ms/batch           4.709 ms/img
BenchmarkBatchSize/Batch08-8                 385          31702649 ns/op                31.70 ms/batch           3.963 ms/img
BenchmarkBatchSize/Batch16-8                 194          63801596 ns/op                63.80 ms/batch           3.988 ms/img
```

#### rk3566

```
BenchmarkBatchSize/Batch01-4                 661          18658568 ns/op                18.66 ms/batch          18.66 ms/img
BenchmarkBatchSize/Batch04-4                 158          74716574 ns/op                74.71 ms/batch          18.68 ms/img
BenchmarkBatchSize/Batch08-4                  70         155374027 ns/op               155.4 ms/batch           19.42 ms/img
BenchmarkBatchSize/Batch16-4                  37         294969497 ns/op               295.0 ms/batch           18.44 ms/img
```


### Interpreting Benchmark Results


The `ms/batch` metric represents the number of milliseconds it took for the 
whole batch inference to run and `ms/img` represents the average number of 
milliseconds it took to run inference per image.

As can be seen in the rk3588 results the ideal batch size is 8 as it gives 
a low `2.792` ms/img inference time versus total batch inference time of 
`22.34ms`.  The same applies to the rk3576.

The rk3566 has a single core NPU, the results show there is no benefit 
in running batching at all.

These results were for an OSNet Model, it's possible that different Models perform 
differently so you should run these benchmarks for your own application to 
optimize accordingly.


## Usage

An example batch program is provided that combines inferencing on a Pool of runtimes,
make sure you have downloaded the data files first for the examples.
You only need to do this once for all examples.


```
cd example/
git clone --depth=1 https://github.com/swdee/go-rknnlite-data.git data
```


Run the batch example on rk3588 or replace with your Platform model.
```
cd example/batch
go run batch.go -s 3 -p rk3588
```

This will result in the output of:
```
Driver Version: 0.9.6, API Version: 2.3.0 (c949ad889d@2024-11-07T11:35:33)
Model Input Number: 1, Ouput Number: 1
Input tensors:
  index=0, name=input, n_dims=4, dims=[8, 224, 224, 3], n_elems=1204224, size=1204224, fmt=NHWC, type=INT8, qnt_type=AFFINE, zp=-14, scale=0.018658
Output tensors:
  index=0, name=output, n_dims=2, dims=[8, 1000, 0, 0], n_elems=8000, size=8000, fmt=UNDEFINED, type=INT8, qnt_type=AFFINE, zp=-55, scale=0.141923
Running...
File ../data/imagenet/n01514859_hen.JPEG, inference time 40ms
File ../data/imagenet/n01518878_ostrich.JPEG, inference time 40ms
File ../data/imagenet/n01530575_brambling.JPEG, inference time 40ms
File ../data/imagenet/n01531178_goldfinch.JPEG, inference time 40ms
...snip...
File ../data/imagenet/n13054560_bolete.JPEG, inference time 8ms
File ../data/imagenet/n13133613_ear.JPEG, inference time 8ms
File ../data/imagenet/n15075141_toilet_tissue.JPEG, inference time 8ms
Processed 1000 images in 2.098619346s, average inference per image is 2.10ms
```

See the help for command line parameters.
```
$ go run batch.go -h

Usage of /tmp/go-build1506342544/b001/exe/batch:
  -d string
        A directory of images to run inference on (default "../data/imagenet/")
  -m string
        RKNN compiled model file (default "../data/models/rk3588/mobilenetv2-batch8-rk3588.rknn")
  -p string
        Rockchip CPU Model number [rk3562|rk3566|rk3568|rk3576|rk3582|rk3582|rk3588] (default "rk3588")
  -q    Run in quiet mode, don't display individual inference results
  -r int
        Repeat processing image directory the specified number of times, use this if you don't have enough images (default 1)
  -s int
        Size of RKNN runtime pool, choose 1, 2, 3, or multiples of 3 (default 1)
```



### Docker

To run the batch example using the prebuilt docker image, make sure the data files have been downloaded first,
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
  go run ./example/batch/batch.go -p rk3588 -s 3
```


## API

A convenience function `rknnlite.NewBatch()` is provided to concatenate individual
images into a single input tensor for the Model and then extract their results 
from the combined outputs.

```
// create a new batch processor
batch := rt.NewBatch(batchSize, height, width, channels)
defer batch.Close()


for idx, file := range files {

    // add files to the batch at the given index
    batch.AddAt(idx, file)
    
    // OR you can add images incrementally without specifying an index
    batch.Add(file)
}

// pass the concatenated Mat to the runtime for inference
outputs, err := rt.Inference([]gocv.Mat{batch.Mat()})

// then get a single image result by index
output, err := batch.GetOutputInt(4, outputs.Output[0], int(outputs.OutputAttributes().DimForDFL))
```

See the full example code for more details.