#!/bin/bash

set -euo pipefail

# platforms we support
PLATFORMS=("rk3562" "rk3566" "rk3568" "rk3576" "rk3582" "rk3588")

# for each platform, point at its “primary” build target
declare -A PRIMARY=(
  [rk3562]=rk3562  [rk3566]=rk3566  [rk3588]=rk3588
  [rk3568]=rk3566  [rk3582]=rk3588  [rk3576]=rk3576
)

# entries: subdir, script, input, dtype, extra-args, outprefix
MODELS=(
  "mobilenet         mobilenet.py      ../model/mobilenetv2-12.onnx       i8    --model     mobilenetv2"
  "yolov5            convert.py        ../model/yolov5s.onnx              i8    ''          yolov5s"
  "yolov8            convert.py        ../model/yolov8s.onnx              i8    ''          yolov8s"
  "yolov8_obb        convert.py        ../model/yolov8n-obb.onnx          i8    ''          yolov8n-obb"
  "yolov10           convert.py        ../model/yolov10s.onnx             i8    ''          yolov10s"
  "yolo11            convert.py        ../model/yolo11s.onnx              i8    ''          yolov11s"
  "yolov8_pose       convert.py        ../model/yolov8n-pose.onnx         i8    ''          yolov8n-pose"
  "yolov5_seg        convert.py        ../model/yolov5s-seg.onnx          i8    ''          yolov5s-seg"
  "RetinaFace        convert.py        ../model/RetinaFace_mobile320.onnx i8    ''          retinaface-320"
  "yolov8_seg        convert.py        ../model/yolov8s-seg.onnx          i8    ''          yolov8s-seg"
  "yolox             convert.py        ../model/yolox_s.onnx              i8    ''          yoloxs"
  "LPRNet            convert.py        ../model/lprnet.onnx               i8    ''          lprnet"
  "PPOCR/PPOCR-Det   convert.py        ../model/ppocrv4_det.onnx          i8    ''          ppocrv4_det"
  "PPOCR/PPOCR-Rec   convert.py        ../model/ppocrv4_rec.onnx          fp    ''          ppocrv4_rec"
  "mobilenet_v1      rknn_convert      /opt/models/mobilenet_v1/model_config.yml    ''    ''    mobilenet_v1"
  "yolov8            convert-lpd.py    /opt/lpd-yolov8/lpd-yolov8n.onnx             i8    ''    lpd-yolov8n"
  "yolov8            convert.py        /opt/go-rknnlite-build/yolonas-s.onnx        i8    ''    yolonas-s"
  "mobilenet         mobilenet-rknn-batch.py      ../model/mobilenetv2-12.onnx      i8    --model     mobilenetv2-batch8"
  "osnet-market1501  build|onnx_to_rknn.py        osnet_x1_0_market_256x128.onnx    i8    ''          osnet-market1501-batch8"
)

# compile all entries (or just filter) for one platform
compile_for_platform() {
  local platform="$1"
  local filter="${2-}"  # optional: only compile entries whose outprefix == filter

  # if this platform is a "child", first build its primary (once)
  local primary="${PRIMARY[$platform]:-$platform}"

  if [[ "$platform" != "$primary" ]]; then
    echo ">>> $platform reuses $primary models - first build $primary models"

    # ensure primary is built (pass along any model filter)
    compile_for_platform "$primary" "$filter"

    # then make symlinks for each model (or just the filtered one)
    echo ">>> $platform reuses $primary models - creating symlinks"
    mkdir -p "/opt/rkmodels/$platform"

    for entry in "${MODELS[@]}"; do
      outprefix=$(awk '{print $6}' <<<"$entry")
      # if we’re filtering, skip the others
      if [[ -n "$filter" && "$outprefix" != "$filter" ]]; then
        continue
      fi
      # create symlink
      dst="/opt/rkmodels/$platform/${outprefix}-${platform}.rknn"
      rel="../${primary}/${outprefix}-${primary}.rknn"
      ln -sfn "$rel" "$dst"
    done
    # done for this child platform—don’t re-build
    return
  fi

  echo "=== platform: $platform ==="
  for entry in "${MODELS[@]}"; do
    read -r subdir script model dtype extra outprefix <<<"$entry"

    # skip if filter is set and doesn't match this entry
    if [[ -n "$filter" && "$outprefix" != "$filter" ]]; then
      continue
    fi

    echo "-> building $outprefix for $platform"
    local out="/opt/rkmodels/${platform}/${outprefix}-${platform}.rknn"

    if [[ "$script" == "rknn_convert" ]]; then
      # mobilenet_v1 special: use the CLI and then rename
      python -m rknn.api.rknn_convert \
        -t "$platform" \
        -i "$model" \
        -o "/opt/rkmodels/$platform/"
      mv "/opt/rkmodels/$platform/${outprefix}.rknn" \
         "$out"
      continue
    fi

    # build the go-rknnlite-build models
    if [[ "$script" == build\|* ]]; then
      # strip everything up to (and including) the first pipe to get script name
      scriptName="${script#*|}"
      # go into the go-rknnlite-build tree
      pushd "/opt/go-rknnlite-build/${subdir}" >/dev/null
        python "$scriptName" "$model" "$platform" "$dtype" "$out"
      popd >/dev/null
      continue
    fi

    # the old examples
    pushd "/opt/rknn_model_zoo/examples/${subdir}/python/" >/dev/null

    if [[ "$subdir" == "mobilenet" ]]; then
      python "$script" $extra "$model" \
        --target "$platform" \
        --dtype "$dtype" \
        --output_path "$out"
    else
      python "$script" "$model" "$platform" "$dtype" "$out"
    fi

    popd >/dev/null
  done
}


# usage command line usage
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 {all|<platform>|<model_name>}"
  echo "  platforms: ${PLATFORMS[*]}"
  echo "  model_name: $(printf "%s\n" "${MODELS[@]}" | awk '{print $6}' | sort | uniq)"
  exit 1
fi

arg="$1"

# check to build for all
if [[ "$arg" == "all" ]]; then
  for plat in "${PLATFORMS[@]}"; do
    compile_for_platform "$plat"
  done
  exit 0
fi

# check to build for specific platform
for plat in "${PLATFORMS[@]}"; do
  if [[ "$arg" == "$plat" ]]; then
    compile_for_platform "$arg"
    exit 0
  fi
done

# is it a known model name
for entry in "${MODELS[@]}"; do
  out=$(awk '{print $6}' <<<"$entry")
  if [[ "$arg" == "$out" ]]; then
    for plat in "${PLATFORMS[@]}"; do
      compile_for_platform "$plat" "$arg"
    done
    exit 0
  fi
done

echo "ERROR: '$arg' is neither a platform nor a model name."
echo "Valid platforms: ${PLATFORMS[*]}"
echo "Valid models: $(printf "%s\n" "${MODELS[@]}" | awk '{print $6}' | sort | uniq)"
exit 1