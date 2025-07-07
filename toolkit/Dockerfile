FROM python:3.12-bookworm

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install wget, git, and libGL (for OpenCV)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      wget \
      git \
      libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Create /opt/models and download all specified ONNX model files
RUN mkdir -p /opt/models && \
    wget -O /opt/models/mobilenetv2-12.onnx \
      https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/mobilenet/mobilenetv2-12.onnx && \
    wget -O /opt/models/yolov5s.onnx \
      https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/yolov5/yolov5s.onnx && \
    wget -O /opt/models/yolov8s.onnx \
      https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/yolov8/yolov8s.onnx && \
    wget -O /opt/models/yolov8n-obb.onnx \
      https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/yolov8_obb/yolov8n-obb.onnx && \
    wget -O /opt/models/yolov10s.onnx \
      https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/yolov10/yolov10s.onnx && \
    wget -O /opt/models/yolo11s.onnx \
      https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/yolo11/yolo11s.onnx && \
    wget -O /opt/models/yolov8n-pose.onnx \
      https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/yolov8_pose/yolov8n-pose.onnx && \
    wget -O /opt/models/yolov5s-seg.onnx \
      https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/yolov5_seg/yolov5s-seg.onnx && \
    wget -O /opt/models/RetinaFace_mobile320.onnx \
      https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/RetinaFace/RetinaFace_mobile320.onnx && \
    wget -O /opt/models/yolov8s-seg.onnx \
      https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/yolov8_seg/yolov8s-seg.onnx && \
    wget -O /opt/models/yolox_s.onnx \
      https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/yolox/yolox_s.onnx && \
    wget -O /opt/models/lprnet.onnx \
      https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/LPRNet/lprnet.onnx && \
    wget -O /opt/models/ppocrv4_det.onnx \
      https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/PPOCR/ppocrv4_det.onnx && \
    wget -O /opt/models/ppocrv4_rec.onnx \
      https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/PPOCR/ppocrv4_rec.onnx

# download the tflite model data
RUN mkdir -p /opt/models/mobilenet_v1 && \
    wget -O /opt/models/mobilenet_v1/mobilenet_v1_1.0_224_quant.tflite \
      https://github.com/airockchip/rknn-toolkit2/raw/refs/heads/master/rknn-toolkit2/examples/tflite/mobilenet_v1_qat/mobilenet_v1_1.0_224_quant.tflite && \
    wget -O /opt/models/mobilenet_v1/model_config.yml \
      https://raw.githubusercontent.com/airockchip/rknn-toolkit2/refs/heads/master/rknn-toolkit2/examples/tflite/mobilenet_v1_qat/model_config.yml

# Clone the RKNN Model Zoo into /opt/rknn_model_zoo
RUN git clone --depth 1 https://github.com/airockchip/rknn_model_zoo.git /opt/rknn_model_zoo

# link all onnx models to the zoo example models directory
RUN ln -s /opt/models/mobilenetv2-12.onnx  /opt/rknn_model_zoo/examples/mobilenet/model/ && \
    ln -s /opt/models/yolov5s.onnx  /opt/rknn_model_zoo/examples/yolov5/model/ && \
    ln -s /opt/models/yolov8s.onnx  /opt/rknn_model_zoo/examples/yolov8/model/ && \
    ln -s /opt/models/yolov8n-obb.onnx  /opt/rknn_model_zoo/examples/yolov8_obb/model/ && \
    ln -s /opt/models/yolov10s.onnx  /opt/rknn_model_zoo/examples/yolov10/model/ && \
    ln -s /opt/models/yolo11s.onnx  /opt/rknn_model_zoo/examples/yolo11/model/ && \
    ln -s /opt/models/yolov8n-pose.onnx  /opt/rknn_model_zoo/examples/yolov8_pose/model/ && \
    ln -s /opt/models/yolov5s-seg.onnx  /opt/rknn_model_zoo/examples/yolov5_seg/model/ && \
    ln -s /opt/models/RetinaFace_mobile320.onnx  /opt/rknn_model_zoo/examples/RetinaFace/model/ && \
    ln -s /opt/models/yolov8s-seg.onnx  /opt/rknn_model_zoo/examples/yolov8_seg/model/ && \
    ln -s /opt/models/yolox_s.onnx  /opt/rknn_model_zoo/examples/yolox/model/ && \
    ln -s /opt/models/lprnet.onnx  /opt/rknn_model_zoo/examples/LPRNet/model/ && \
    ln -s /opt/models/ppocrv4_det.onnx  /opt/rknn_model_zoo/examples/PPOCR/PPOCR-Det/model/ && \
    ln -s /opt/models/ppocrv4_rec.onnx  /opt/rknn_model_zoo/examples/PPOCR/PPOCR-Rec/model/

# download LPD model and create convert script for it in the model zoo
RUN git clone --depth 1 https://github.com/swdee/lpd-yolov8.git /opt/lpd-yolov8 && \
    cp /opt/rknn_model_zoo/examples/yolov8/python/convert.py /opt/rknn_model_zoo/examples/yolov8/python/convert-lpd.py && \
    sed -i "s|^DATASET_PATH *= *['\"].*['\"]|DATASET_PATH = '/opt/lpd-yolov8/subset.txt'|" /opt/rknn_model_zoo/examples/yolov8/python/convert-lpd.py

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install the Python dependencies
RUN pip install --timeout 300 --retries 10 -r \
    "https://raw.githubusercontent.com/airockchip/rknn-toolkit2/refs/tags/v2.3.2/rknn-toolkit2/packages/x86_64/requirements_cp312-2.3.2.txt"

# Install the RKNN-Toolkit2 wheel
RUN pip install \
    "https://github.com/airockchip/rknn-toolkit2/raw/refs/tags/v2.3.2/rknn-toolkit2/packages/x86_64/rknn_toolkit2-2.3.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"

# install extra python deps for converting tflite models
RUN pip install --no-cache-dir \
    pyyaml \
    "tensorflow<=2.16.0rc0"

# download other onnx models
RUN git clone --depth 1 https://github.com/swdee/go-rknnlite-build.git /opt/go-rknnlite-build && \
    git -C /opt/go-rknnlite-build fetch --depth 1 origin ce8b5ce1dc53b1c38324e7506374731ad21070c8 && \
    git -C /opt/go-rknnlite-build checkout FETCH_HEAD

# copy our modified mobilenet.py script into the rknn_model_zoo directory
RUN cp /opt/go-rknnlite-build/mobilenet-batch/mobilenet-rknn.py /opt/rknn_model_zoo/examples/mobilenet/python/mobilenet-rknn-batch.py


# By default do nothing
CMD ["bash"]

