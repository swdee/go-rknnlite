
# Examples

## YOLO Benchmarks

Examples for a variety of YOLO versions have been provided, the following table
shows benchmarks of how each version performs with respect to resource requirements
running on the RK3588. 
Note that detection results vary depending on the YOLO version.

The `data/palace.jpg` image file was used for benchmarking.


![Average Time Graph](https://github.com/swdee/go-rknnlite-data/raw/master/yolobench/avg-time-graph.png)


### rk3588

Results for the rk3588 with 6 TOPS three core NPU operating at 1Ghz.


| Model        | Average Total Time | Inference | Post Processing | Rendering |
|--------------|--------------------|-----------|-----------------|-----------|
| YOLOv5s      | 30.2ms             | 34.9ms    | 0.4ms           | 2.2ms     |
| YOLOv8s      | 39.6ms             | 41.5ms    | 2.4ms           | 2.3ms     |
| YOLOv10s     | 48.8ms             | 47.5ms    | 1.3ms           | 2.0ms     |
| YOLOXs       | 38.4ms             | 39.4ms    | 0.2ms           | 2.6ms     |
| YOLOv11s     | 46.8ms             | 49.0ms    | 0.9ms           | 1.7ms     |
| YOLOv8n-pose | 37.8ms             | 37.3ms    | 0.5ms           | 1.4ms     |
| YOLOv5s-seg  | 108.1ms            | 46.0ms    | 64.6ms          | 4.4ms     |
| YOLOv8s-seg  | 122.0ms            | 53.0ms    | 72.4ms          | 4.3ms     |


### rk3576

Results for the rk3576 with 6 TOPS two core NPU operating at 950Mhz.


| Model        | Average Total Time | Inference | Post Processing | Rendering |
|--------------|--------------------|-----------|-----------------|-----------|
| YOLOv5s      | 29.2ms             | 48.3ms    | 1.0ms           | 3.8ms     |
| YOLOv8s      | 43.4ms             | 60.6ms    | 4.1ms           | 4.0ms     |
| YOLOv10s     | 49.2ms             | 68.2ms    | 2.3ms           | 3.5ms     |
| YOLOXs       | 37.9ms             | 56.2ms    | 0.6ms           | 4.7ms     |
| YOLOv11s     | 38.4ms             | 70.1ms    | 2.0ms           | 3.4ms     |
| YOLOv8n-pose | 33.9ms             | 49.5ms    | 0.9ms           | 2.3ms     |
| YOLOv5s-seg  | 151.9ms            | 65.9ms    | 95.7ms          | 7.0ms     |
| YOLOv8s-seg  | 154.2ms            | 78.7ms    | 126.7ms         | 8.2ms     |


### rk3566

Results for the rk3566 with 1 TOPS single core NPU operating at 900Mhz.


| Model        | Average Total Time | Inference | Post Processing | Rendering |
|--------------|--------------------|-----------|-----------------|-----------|
| YOLOv5s      | 70.2ms             | 70.4ms    | 2.4ms           | 7.2ms     |
| YOLOv8s      | 95.5ms             | 84.6ms    | 10.3ms          | 7.6ms     |
| YOLOv10s     | 122.3ms            | 116.3ms   | 7.6ms           | 8.1ms     |
| YOLOXs       | 83.0ms             | 78.9ms    | 1.7ms           | 9.8ms     |
| YOLOv11s     | 117.3ms            | 117.6ms   | 4.9ms           | 7.1ms     |
| YOLOv8n-pose | 59.8ms             | 62.2ms    | 1.6ms           | 4.5ms     |
| YOLOv5s-seg  | 433.64ms           | 95.7ms    | 349.7ms         | 20.0ms    |
| YOLOv8s-seg  | 420.8ms            | 110.7ms   | 312.3ms         | 19.7ms    |



The Inference, Post Processing, and Rendering columns show how processing time
is split across the Total Time.   These figures are derived from the first
run on the benchmark so when totaled together they are slightly higher than the
Average Total Time.

The Inference column represents processing on the NPU, Post Processing and Rendering
values are performed on the CPU.

|     ![YOLOv5 Output](https://github.com/swdee/go-rknnlite-data/raw/master/yolobench/yolov5-out.jpg)     |     ![YOLOv8 Output](https://github.com/swdee/go-rknnlite-data/raw/master/yolobench/yolov5-out.jpg)     | 
|:-------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------:|
|                                      YOLOv5s - 21 Objects Detected                                      |                                      YOLOv8s - 22 Objects Detected                                      |
|    ![YOLOv10 Output](https://github.com/swdee/go-rknnlite-data/raw/master/yolobench/yolov10-out.jpg)    |      ![YOLOX Output](https://github.com/swdee/go-rknnlite-data/raw/master/yolobench/yolox-out.jpg)      |
|                                     YOLOv10s - 19 Objects Detected                                      |                                      YOLOXs - 29 Objects Detected                                       |
|    ![YOLOv11 Output](https://github.com/swdee/go-rknnlite-data/raw/master/yolobench/yolov11-out.jpg)    |      ![YOLOv8-pose Output](https://github.com/swdee/go-rknnlite-data/raw/master/yolobench/yolov8-pose-out.jpg) |  
|                                     YOLOv11s - 19 Objects Detected                                      |                                    YOLOv8n-pose - 9 Objects Detected                                    |                                   
| ![YOLOv5-seg Output](https://github.com/swdee/go-rknnlite-data/raw/master/yolobench/yolov5-seg-out.jpg) | ![YOLOv8-seg Output](https://github.com/swdee/go-rknnlite-data/raw/master/yolobench/yolov8-seg-out.jpg) |
|                                    YOLOv5s-seg - 21 Objects Detected                                    |                                    YOLOv8s-seg - 20 Objects Detected                                    |

