
# go-rknnlite Changes

## General Changes

See the [commit log](https://github.com/swdee/go-rknnlite/commits/master/) for general changes.


## Breaking Changes

Some notes on breaking changes.


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
