
# Examples

## YOLO Benchmarks

Examples for a variety of YOLO versions have been provided, the following table
shows benchmarks of how each version performs with respect to resource requirements
running on the RK3588. 
Note that detection results vary depending on the YOLO version.

The `data/palace.jpg` image file was used for benchmarking.


![Average Time Graph](https://github.com/swdee/go-rknnlite-data/raw/master/yolobench/avg-time-graph.png)


### rk3588

Results for the rk3588 with 6 TOPS three core NPU operating at 1Ghz for p50 timing values.

| Model        | Average Total Time | Inference | Post Processing | Rendering |
|--------------|--------------------|-----------|-----------------|-----------|
| YOLOv5s      | 33.3ms             | 25.5ms    | 1.5ms           | 5.5ms     |
| YOLOv8s      | 35.3ms             | 29.7ms    | 2.4ms           | 2.0ms     |
| YOLOv10s     | 36.4ms             | 33.2ms    | 1.2ms           | 1.7ms     |
| YOLOXs       | 32.3ms             | 29.4ms    | 0.3ms           | 2.3ms     |
| YOLOv11s     | 37.1ms             | 34.1ms    | 1.0ms           | 1.7ms     |
| YOLO26       | 40.8ms           | 37.2ms    | 1.3ms           | 1.9ms     |
| YOLO-NAS     | 54.3ms             | 50.8ms    | 0.8ms           | 0.6ms     |
| YOLOv8n-pose | 31.4ms             | 25.9ms    | 1.4ms           | 3.5ms     |
| YOLOv8-OBB   | 16.5ms             | 15.5ms    | 0.6ms           | 0.0ms     |
| YOLOv5s-seg  | 57.0ms             | 37.6ms    | 12.7ms          | 4.3ms     |
| YOLOv8s-seg  | 66.9ms             | 46.4ms    | 14.9ms          | 4.2ms     |


### rk3576

Results for the rk3576 with 6 TOPS two core NPU operating at 950Mhz for p50 timing values.


| Model        | Average Total Time | Inference | Post Processing | Rendering |
|--------------|--------------------|-----------|-----------------|-----------|
| YOLOv5s      | 31.7ms             | 25.1ms    | 1.2ms           | 5.4ms     |
| YOLOv8s      | 51.6ms             | 35.4ms    | 8.4ms           | 5.7ms     |
| YOLOv10s     | 39.1ms             | 33.5ms    | 2.1ms           | 2.9ms     |
| YOLOXs       | 36.0ms             | 30.7ms    | 0.6ms           | 4.0ms     |
| YOLOv11s     | 49.4ms             | 38.9ms    | 4.0ms           | 5.0ms     |
| YOLO26       | 45.4ms             | 38.3ms    | 3.1ms           | 3.4ms     |
| YOLO-NAS     | 38.9ms             | 36.1ms    | 1.3ms           | 1.0ms     |
| YOLOv8n-pose | 33.2ms             | 27.9ms    | 1.4ms           | 3.4ms     |
| YOLOv8-OBB   | 16.7ms             | 15.3ms    | 0.7ms           | 0.1ms     |
| YOLOv5s-seg  | 61.7ms             | 32.5ms    | 20.7ms          | 6.9ms     |
| YOLOv8s-seg  | 80.5ms             | 47.3ms    | 23.9ms          | 6.6ms     |

### rk3566

Results for the rk3566 with 1 TOPS single core NPU operating at 900Mhz for p50 timing values.


| Model        | Average Total Time | Inference | Post Processing | Rendering |
|--------------|--------------------|-----------|-----------------|-----------|
| YOLOv5s      | 69.5ms             | 58.7ms    | 2.3ms           | 6.8ms     |
| YOLOv8s      | 92.4ms             | 73.7ms    | 9.9ms           | 7.4ms     |
| YOLOv10s     | 126.8ms            | 104.9ms   | 7.6ms           | 8.2ms     |
| YOLOXs       | 82.8ms             | 70.8ms    | 1.5ms           | 9.1ms     |
| YOLOv11s     | 116.9ms            | 103.1ms   | 4.7ms           | 6.9ms     |
| YOLO26       | 145.4ms            | 127.5ms   | 8.4ms           | 7.7ms     |
| YOLO-NAS     | 125.0ms            | 113.1ms   | 4.1ms           | 2.0ms     |
| YOLOv8n-pose | 58.4ms             | 51.0ms    | 1.5ms           | 4.2ms     |
| YOLOv8-OBB   | 37.3ms             | 33.8ms    | 2.0ms           | 0.2ms     |
| YOLOv5s-seg  | 144.9ms            | 78.3ms    | 49.8ms          | 14.2ms    |
| YOLOv8s-seg  | 168.4ms            | 96.4ms    | 56.8ms          | 14.0ms    |

The Inference, Post Processing, and Rendering columns show how processing time
is split across the Total Time.   These figures are derived from the first
run on the benchmark so when totaled together they are slightly higher than the
Average Total Time.

The Inference column represents processing on the NPU, Post Processing and Rendering
values are performed on the CPU.

|     ![YOLOv5 Output](https://github.com/swdee/go-rknnlite-data/raw/master/yolobench/yolov5-out.jpg)     |   ![YOLOv8 Output](https://github.com/swdee/go-rknnlite-data/raw/master/yolobench/yolov5-out.jpg)    | 
|:-------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------:|
|                                      YOLOv5s - 21 Objects Detected                                      |                                    YOLOv8s - 22 Objects Detected                                     |
|    ![YOLOv10 Output](https://github.com/swdee/go-rknnlite-data/raw/master/yolobench/yolov10-out.jpg)    |    ![YOLOX Output](https://github.com/swdee/go-rknnlite-data/raw/master/yolobench/yolox-out.jpg)     |
|                                     YOLOv10s - 19 Objects Detected                                      |                                     YOLOXs - 29 Objects Detected                                     |
|    ![YOLOv11 Output](https://github.com/swdee/go-rknnlite-data/raw/master/yolobench/yolov11-out.jpg)    | ![YOLOv8-pose Output](https://github.com/swdee/go-rknnlite-data/raw/master/yolobench/yolo26-out.jpg) |  
|                                     YOLOv11s - 19 Objects Detected                                      |                                     YOLO26 - 22 Objects Detected                                     |
|         ![YOLOv8-pose Output](https://github.com/swdee/go-rknnlite-data/raw/master/yolobench/yolov8-pose-out.jpg) | |  
|           YOLOv8n-pose - 9 Objects Detected                                    | |
| ![YOLOv5-seg Output](https://github.com/swdee/go-rknnlite-data/raw/master/yolobench/yolov5-seg-out.jpg) | ![YOLOv8-seg Output](https://github.com/swdee/go-rknnlite-data/raw/master/yolobench/yolov8-seg-out.jpg) |
|                                    YOLOv5s-seg - 21 Objects Detected                                    |                                    YOLOv8s-seg - 20 Objects Detected                                    |

