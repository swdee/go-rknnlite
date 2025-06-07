#!/bin/bash

set -e

PLATFORMS=("rk3562" "rk3566" "rk3568" "rk3576" "rk3588")

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
)

function compile_for_platform() {
  local platform="$1"
  echo "Compiling models for platform: $platform"

  for entry in "${MODELS[@]}"; do
    IFS=' ' read -r subdir script model dtype extra outprefix <<< "$entry"
    dir="/opt/rknn_model_zoo/examples/${subdir}/python/"
    out="/opt/rkmodels/${platform}/${outprefix}-${platform}.rknn"
    echo "-> $model -> $out"
    cd "$dir"

    if [[ "$subdir" == "mobilenet" ]]; then
      python "$script" $extra "$model" --target "$platform" --dtype "$dtype" --output_path "$out"
    else
      python "$script" "$model" "$platform" "$dtype" "$out"
    fi
  done
}

# check valid command line argument platform option
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 {rk3562|rk3566|rk3568|rk3576|rk3588|all}"
  exit 1
fi

if [[ "$1" == "all" ]]; then
  for platform in "${PLATFORMS[@]}"; do
    compile_for_platform "$platform"
  done

elif [[ " ${PLATFORMS[*]} " == *" $1 "* ]]; then
  compile_for_platform "$1"

else
  echo "Invalid platform: $1"
  echo "Valid options are: ${PLATFORMS[*]} or 'all'"
  exit 1
fi
