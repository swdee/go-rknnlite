
# go-rknnlite

go-rknnlite provides Go language bindings for the [RKNN Toolkit2](https://github.com/airockchip/rknn-toolkit2/tree/master)
C API interface.  It aims to provide lite bindings in the spirit of the closed source
Python lite bindings used for running AI Inference models on the Rockchip NPU 
via the RKNN software stack.

These bindings have only been tested on the [RK3588](https://www.rock-chips.com/a/en/products/RK35_Series/2022/0926/1660.html)
(specifically the Radxa Rock Pi 5B) but should work on other RK3588 based SBC's.
It should also work with other models in the RK35** series supported by the RKNN Toolkit2.


# Usage

```
go get github.com/swdee/go-rknnlite
```