
# go-rknnlite

go-rknnlite provides Go language bindings for the [RKNN Toolkit2](https://github.com/airockchip/rknn-toolkit2/tree/master)
C API interface.  It aims to provide lite bindings in the spirit of the closed source
Python lite bindings used for running AI Inference models on the Rockchip NPU 
via the RKNN software stack.

These bindings have only been tested on the [RK3588](https://www.rock-chips.com/a/en/products/RK35_Series/2022/0926/1660.html)
(specifically the Radxa Rock Pi 5B) but should work on other RK3588 based SBC's.
It should also work with other models in the RK35xx series supported by the RKNN Toolkit2.


## Usage

```
go get github.com/swdee/go-rknnlite
```

## Dependencies

The [rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2) must be installed on 
your system with C header files available in the system path, eg: `/usr/include/rknn_api.h`.

Refer to the official documentation on how to install this on your system as it
will vary based on OS and SBC vendor.

### Rock Pi 5b

My usage was on the Radxa Rock Pi 5b running the official Debian 11 OS image. 

I used the prebuilt RKNN libraries built [here](https://github.com/radxa-pkg/rknn2/releases).

```
wget https://github.com/radxa-pkg/rknn2/releases/download/1.6.0-2/rknpu2-rk3588_1.6.0-2_arm64.deb
apt install ./rknpu2-rk3588_1.6.0-2_arm64.deb 
```


## Examples

See the [example](example) directory.

* [MobileNet Demo](example/mobilenet)
* [Pooled Runtime Usage](example/pool)


## Pooled Runtimes

Running multiple Runtimes in a Pool allows you to take advantage of all three
NPU cores.  For our usage of an EfficentNet-Lite0 model, a single runtime has
an inference speed of 7.9ms per image, however running a Pool of 9 runtimes brings
the average inference speed down to 1.65ms per image.

See the [Pool example](example/pool).


## Reference Material

* [rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2) - RKNN software stack
tools and C API.
* [C API Reference Documentation](https://github.com/airockchip/rknn-toolkit2/blob/master/doc/04_Rockchip_RKNPU_API_Reference_RKNNRT_V2.0.0beta0_EN.pdf)