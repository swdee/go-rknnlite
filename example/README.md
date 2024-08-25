
# Examples

## YOLO Benchmarks

Examples for a variety of YOLO versions have been provided, the following table
shows benchmarks of how each version performs with respect to resource requirements. 
Note that detection results vary depending on the YOLO version.

The `data/palace.jpg` image file was used for benchmarking.

| Model      | Average Total Time | Inference | Post Processing | Rendering |
|------------|--------------------|-----------|-----------------|-----------|
| YOLOv5     | 29.6ms             | 33.6ms    | 1.0ms           | 4.4ms     |
| YOLOv8     | 44.8ms             | 47.5ms    | 5.0ms           | 4.2ms     |
| YOLOv10    | 49.4ms             | 57.1ms    | 2.9ms           | 4.1ms     |
| YOLOX      | 42.4ms             | 48.7ms    | 0.5ms           | 5.5ms     |
| YOLOv5-seg | 207.5ms            | 57.0ms    | 107.6ms         | 75.9ms    |
| YOLOv8-seg | 232.5ms            | 67.9ms    | 113.3ms         | 73.4ms    |

The Inference, Post Processing, and Rendering columns show how processing time
is split across the Total Time.   These figures are derived from the first
run on the benchmark so when totaled together they are slightly higher than the
Average Total Time.

The Inference column represents processing on the NPU, Post Processing and Rendering
values are performed on the CPU.

|     ![YOLOv5 Output](https://github.com/swdee/go-rknnlite-data/raw/master/yolobench/yolov5-out.jpg)     |     ![YOLOv8 Output](https://github.com/swdee/go-rknnlite-data/raw/master/yolobench/yolov5-out.jpg)     | 
|:-------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------:|
|                                      YOLOv5 - 21 Objects Detected                                       |                                      YOLOv8 - 22 Objects Detected                                       |
|    ![YOLOv10 Output](https://github.com/swdee/go-rknnlite-data/raw/master/yolobench/yolov10-out.jpg)    |      ![YOLOX Output](https://github.com/swdee/go-rknnlite-data/raw/master/yolobench/yolox-out.jpg)      |
|                                      YOLOv10 - 19 Objects Detected                                      |                                       YOLOX - 29 Objects Detected                                       |
| ![YOLOv5-seg Output](https://github.com/swdee/go-rknnlite-data/raw/master/yolobench/yolov5-seg-out.jpg) | ![YOLOv8-seg Output](https://github.com/swdee/go-rknnlite-data/raw/master/yolobench/yolov8-seg-out.jpg) |
|                                    YOLOv5-seg - 21 Objects Detected                                     |                                    YOLOv8-seg - 20 Objects Detected                                    |