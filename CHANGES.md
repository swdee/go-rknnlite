
# go-rknnlite Changes

## General Changes

See the [commit log](https://github.com/swdee/go-rknnlite/commits/master/) for general changes.


## Performance Changes

### 22 May, 2026

The YOLOv5 and YOLOv8 Segmentation post processing code has been refactored and gained 
significant improvement with post processing performance being 75% faster.  Total overhead
including inference, post processing, and rendering had a 40.9% gain on RK3588 platform reducing
per frame processing time from 115ms to 68ms.

This was achieved with the following optimizations:

1) Optimized matmul algorithm: remove sigmoid + channel-contiguous accumulation.

2) Wrote optimized NEON/SIMD matmul function using improvements from (1). 

3) Changed segmentation mask resizing to work on per object ROI mask size instead of full frame image resizing.




## Breaking Changes

Some notes on breaking changes.


### June 21, 2025

[PR #41](https://github.com/swdee/go-rknnlite/pull/41/files)

Created the SAHI preprocessor.  This required moving the `postprocess/detect.go` file
into its own package at `postprocess/result/detect.go` to avoid a cyclic import. 
The `postprocess.NewIDGenerator()` also needed to be shifted to the `result` package.

These changes only effect the internal's of the API code, so there will be no
breaking changes if your usage follows those provided in the [examples](example/). 

However if you are doing something custom outside of the code examples then these changes
could result in code breakage and would require the following updates.

| Previous API naming          | New API naming          |
|------------------------------|-------------------------|
| postprocess.NewIDGenerator() | result.NewIDGenerator() |
| postprocess.DetectionResult  | result.DetectionResult  |
| postprocess.BoxRectMode      | result.BoxRectMode  |
| postprocess.BoxRect          | result.BoxRect  |
| postprocess.DetectResult     | result.DetectResult  |
| postprocess.KeyPoint         | result.KeyPoint  |



### Apr 2, 2025 

[PR #31](https://github.com/swdee/go-rknnlite/pull/31/files)

Changed `rknnlite.NewPool()` function to require passing of NPU CoreMask list.  This
was done to support other Rockchip RK35xx models that feature either single or dual
NPU cores.  

The original code assumed usage of RK3588 with three NPU cores by creating a
Pool as follows
```
pool, err := rknnlite.NewPool(*poolSize, *modelFile)
```

Update any existing code by passing in RK3588's NPU core list.
```
pool, err := rknnlite.NewPool(*poolSize, *modelFile, rknnlite.RK3588)
```

For other Rockchip models you can make use of the convenience variables or pass
in your own CoreMask list.

https://github.com/swdee/go-rknnlite/blob/5b10c181077fe21f89a89628e247cc594aa8782d/runtime.go#L33-L43
