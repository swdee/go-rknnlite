# YOLONAS Example


## Usage

Make sure you have downloaded the data files first for the examples.
You only need to do this once for all examples.

```
cd example/
git clone --depth=1 https://github.com/swdee/go-rknnlite-data.git data
```

Run the YOLO-NAS example on rk3588 or replace with your Platform model.
```
cd example/yolo-nas
go run yolo-nas.go -p rk3588
```

This will result in the output of:
```
Driver Version: 0.9.6, API Version: 2.3.0 (c949ad889d@2024-11-07T11:35:33)
Model Input Number: 1, Ouput Number: 2
Input tensors:
  index=0, name=images, n_dims=4, dims=[1, 640, 640, 3], n_elems=1228800, size=1228800, fmt=NHWC, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003922
Output tensors:
  index=0, name=output, n_dims=3, dims=[1, 8400, 4, 0], n_elems=33600, size=33600, fmt=UNDEFINED, type=INT8, qnt_type=AFFINE, zp=-90, scale=3.387419
  index=1, name=1100, n_dims=3, dims=[1, 8400, 80, 0], n_elems=672000, size=672000, fmt=UNDEFINED, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003898
bus @ (98 132 565 447) 0.970715
person @ (213 240 284 508) 0.962918
person @ (108 240 230 535) 0.951223
person @ (474 233 565 521) 0.943426
person @ (77 321 132 518) 0.682229
Model first run speed: inference=74.299132ms, post processing=2.090633ms, rendering=715.155µs, total time=77.10492ms
Saved object detection result to ../data/bus-yolo-nas-out.jpg
Benchmark time=5.940757606s, count=100, average total time=59.407576ms
done
```


The saved JPG image with object detection markers.

![bus-yolo-nas-out.jpg](https://github.com/swdee/go-rknnlite-data/raw/master/docimg/bus-yolo-nas-out.jpg)



To use your own RKNN compiled model and images.
```
go run yolo-nas.go -m <RKNN model file> -i <image file> -l <labels txt file> -o <output jpg file> -p <platform>
```

The labels file should be a text file containing the labels the Model was trained on.
It should have one label per line.


See the help for command line parameters.
```
$ go run yolo-nas.go --help

Usage of /tmp/go-build4215758863/b001/exe/yolo-nas:
  -i string
        Image file to run object detection on (default "../data/bus.jpg")
  -l string
        Text file containing model labels (default "../data/coco_80_labels_list.txt")
  -m string
        RKNN compiled YOLO model file (default "../data/models/rk3588/yolonas-s-rk3588.rknn")
  -o string
        The output JPG file with object detection markers (default "../data/bus-yolo-nas-out.jpg")
  -p string
        Rockchip CPU Model number [rk3562|rk3566|rk3568|rk3576|rk3582|rk3582|rk3588] (default "rk3588")
```



### Docker

To run the YOLO-NAS example using the prebuilt docker image, make sure the data files have been downloaded first,
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
  go run ./example/yolo-nas/yolo-nas.go -p rk3588
```



## Benchmarks

The following table shows a comparison of the benchmark results across the three distinct platforms.


| Platform | Execution Time | Average Inference Time Per Image |
|----------|----------------|----------------------------------|
| rk3588   | 5.94s          | 59.40ms                          |
| rk3576   | 4.68s          | 46.86ms                          |
| rk3566   | 13.16s         | 131.61ms                         |


### RK3588 Bug

Note that the rk3588 benchmark is using a RKNN model compiled with rknn-toolkit2 
version 2.1.0 which has slower inference time than the newer version 2.3.2 which
was used on the other platforms.

This is due to a [bug reported here](https://github.com/airockchip/rknn-toolkit2/issues/378).  If 
you wish to use your own trained YOLO-NAS model on rk3588, make sure you compile 
to RKNN format using the older version of rknn-toolkit2.



## Converting YOLO-NAS Model

YOLO-NAS is not a model that 
is supported by the upstream vendor in their model zoo, so to export it 
to ONNX and convert to RKNN format follow these steps.

Setup a python virtual environment using Python 3.10.

Install python dependencies for the [YOLO-NAS project](https://github.com/Deci-AI/super-gradients).
```
pip install super-gradients torch onnx onnx-simplifier
```

Patch the download links in the super-gradients source code.
```
sed -i -e "s/sghub.deci.ai/sg-hub-nv.s3.amazonaws.com/g" .venv/lib/python3.10/site-packages/super_gradients/training/pretrained_models.py
sed -i -e "s/sghub.deci.ai/sg-hub-nv.s3.amazonaws.com/g" .venv/lib/python3.10/site-packages/super_gradients/training/utils/checkpoint_utils.py
```

Create a python script named `export_onnx.py` with contents:
```
import torch
from super_gradients.training import models
from super_gradients.common.object_names import Models

# Load the YOLO-NAS model (you can also load your own checkpoint by using the `checkpoint_path` arg)
model = models.get(Models.YOLO_NAS_S, pretrained_weights="coco")

# Switch to eval mode and prepare for conversion
model.eval()

# Specify the expected input shape: [batch_size, channels, height, width]
model.prep_model_for_conversion(input_size=(1, 3, 640, 640))

# Export with torch.onnx.export
dummy_input = torch.randn(1, 3, 640, 640)
torch.onnx.export(
    model,
    dummy_input,
    "yolo_nas_s_manual.onnx",
    opset_version=11,
    input_names=["images"],
    output_names=["output"],
) 
```

Then run the script in your Python 3.10 virtual environment and it will save
the model as ONNX format.
```
python export_onnx.py
```

The above script uses the pretrained COCO dataset so you can use the YOLOv8 
[convert.py script](https://github.com/airockchip/rknn_model_zoo/blob/main/examples/yolov8/python/convert.py) 
from the model zoo to export as RKNN format.


