# SAHI (Slicing Aided Hyper Inference) Example

## Overview

When dealing with high resolution source images, detail within them gets lost
as they are scaled down for input to the tensor size of the inference model.  A
YOLO model typically has a tensor input size of 640x640 so with in a large 4k 
(3840x2160) image few objects will be detected.

Running the [YOLOv5 example](../example/yolov5) on the 4k 
[input image](https://github.com/swdee/go-rknnlite-data/raw/master/protest.jpg) results in
only a few detections at the front of the scene.

![Protest YOLOv5 Output](https://github.com/swdee/go-rknnlite-data/raw/master/docimg/protest-yolov5-out-scaled.jpg)

To work around this problem SAHI was proposed where the source image is sliced 
up into smaller images (tiles) and for each tile to have inference performed.  Once
all tiles have been processed, the detection results are combined to produce
an output for the full sized 4k image as seen below.

![Protest SAHI Output](https://github.com/swdee/go-rknnlite-data/raw/master/docimg/protest-sahi-out-scaled.jpg)

This image was sliced up into 24 separate tiles, this example makes use of a [Pool](../example/pool)
to process these tiles concurrently.


## Usage

Make sure you have downloaded the data files first for the examples.
You only need to do this once for all examples.

```
cd example/
git clone --depth=1 https://github.com/swdee/go-rknnlite-data.git data
```


Command line Usage.
```
$ go run sahi.go -h

Usage of /tmp/go-build2138786984/b001/exe/sahi:
  -i string
        Image file to run object detection on (default "../data/protest.jpg")
  -l string
        Text file containing model labels (default "../data/coco_80_labels_list.txt")
  -m string
        RKNN compiled YOLO model file (default "../data/models/rk3588/yolov5s-rk3588.rknn")
  -o string
        The output JPG file with object detection markers (default "../data/protest-sahi-out.jpg")
  -p string
        Rockchip CPU Model number [rk3562|rk3566|rk3568|rk3576|rk3582|rk3582|rk3588] (default "rk3588")
  -s int
        Size of RKNN runtime pool, choose 1, 2, 3, or multiples of 3 (default 1)
```


Run the SAHI example using a Pool of 6 runtimes on rk3588 or replace with your 
Platform model
```
cd example/sahi/
go run sahi.gp -s 6 -p rk3588
```

This will result in the output of:
```
Source image dimensions 3840x2160

Processing Slice (614 0 1383 769) with box size (769 769)
person @ (55 9 159 249) 0.868512
person @ (521 32 602 239) 0.821592
person @ (435 40 550 244) 0.802215
person @ (612 80 692 235) 0.718739
person @ (688 54 767 186) 0.635571
person @ (299 21 392 243) 0.598754
tie @ (105 64 126 116) 0.322937
person @ (0 23 20 148) 0.265283

Processing Slice (0 0 769 769) with box size (769 769)
...snip...

Combined object detection results
person @ (3078 1392 3226 1751) 0.925490
person @ (2508 1221 2610 1587) 0.465206
...snip...
person @ (2288 104 2339 157) 0.273433
person @ (2592 927 2664 1043) 0.272372
person @ (2630 103 2664 154) 0.261715
person @ (2638 825 2705 989) 0.256055
SAHI Execution speed=314.001875ms, slices=24, objects=171
Saved object detection result to ../data/protest-sahi-out.jpg
done
```

Note that the above output has been truncated for brevity.


### Docker

To run the SAHI example using the prebuilt docker image, make sure the data files have been downloaded first,
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
  go run ./example/sahi/sahi.go -s 6 -p rk3588
```


## API Parameters

When initializing a `preprocess.NewSAHI()` instance the function arguments
require you to pass the input tensor dimensions (640x640) and an overlap ratio value
for the width (0.2) and height (0.2).

The overlap ratio is a value from 0.0 to 1.0 and represents the percentage of the 
input tensor that should be used for overlapping the tiles.

For example an input tensor width of 640 pixels and overlap width of 0.2 means 
the overlap used is 20% of 640, which is 128 pixels.

```
sahi := preprocess.NewSAHI(640, 640, 0.2, 0.2)
```

The source image is then sliced accordingly and inference can be run on each slice. 
```
slices := sahi.Slice(img)
```

This example makes use of YOLOv5, however you may use the other YOLO model versions
(v8, v10, v11, and x) as well.

Each slice's inference result needs to be added to the SAHI instance.
```
for _, slice := range slices {
  // run inference
  
  // add result
  sahi.AddResult(slice, detectResults)
}
```

Finally the combined result for the full sized image can be retrieved.
```
detectResults := sahi.GetDetectResults(postprocess.YOLOv5COCOParams().NMSThreshold, 0.7)
```

The combined result applies an NMS (non maximum suppression) algorithm to
eliminate the duplicate objects detected at a slices edge (overlap region).  It also
uses a small box overlap threshold (0.7) where objects that were sliced in half
get eliminated when they overlap each other by more than the threshold amount.


