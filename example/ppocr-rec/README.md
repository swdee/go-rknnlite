# PaddleOCR (PPOCR)

[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) provides multilingual
OCR based on the PaddlePaddle lightweight OCR system, supporting recognition of
80+ languages.

## Usage

Make sure you have downloaded the data files first for the examples.
You only need to do this once for all examples.

```
cd example/
git clone https://github.com/swdee/go-rknnlite-data.git data
```

Run the PPOCR Recognition example.
```
cd example/ppocr-rec
go run ppocr-rec.go
```

This will result in the output of:
```
Driver Version: 0.8.2, API Version: 1.6.0 (9a7b5d24c@2023-12-13T17:31:11)
Model Input Number: 1, Ouput Number: 1
Input tensors:
  index=0, name=x, n_dims=4, dims=[1, 48, 320, 3], n_elems=46080, size=92160, fmt=NHWC, type=FP16, qnt_type=AFFINE, zp=0, scale=1.000000
Output tensors:
  index=0, name=softmax_11.tmp_0, n_dims=3, dims=[1, 40, 6625, 0], n_elems=265000, size=530000, fmt=UNDEFINED, type=FP16, qnt_type=AFFINE, zp=0, scale=1.000000
Model first run speed: inference=24.707428ms, post processing=478.906µs, total time=25.186334ms
Recognize result: JOINT, score=0.71
Benchmark time=321.330438ms, count=10, average total time=32.133043ms
done
```

Sample images input and text detected.


| Input Image                       | Text Recognised | Confidence Score |
|-----------------------------------|-----------------|------------------|
| ![joint.png](joint.png)           | JOINT           | 0.71             |
| ![region.jpg](region.jpg)         |    浙G·Z6825        | 0.65         |
| ![cn-text.png](cn-text.png)       |    中华老字号        | 0.71          |
| ![mozzarella.jpg](mozzarella.jpg) |    MOZZARELLA - 188        | 0.67  |





## Background

This PPOCR example is a Go conversion of the [C API example](https://github.com/airockchip/rknn_model_zoo/blob/main/examples/PPOCR/PPOCR-Rec/cpp/main.cc).


