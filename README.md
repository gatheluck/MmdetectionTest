# MMDetection Test

[![CI](https://github.com/gatheluck/sandbox/workflows/CI/badge.svg)](https://github.com/gatheluck/MmdetectionTest/actions?query=workflow%3ACI)
[![License](https://img.shields.io/github/license/gatheluck/MmdetectionTest?color=green)](LICENSE)

## [Prerequisites](https://mmdetection.readthedocs.io/en/latest/get_started.html#)
- Python 3.6+
- Pytorch 1.3+
- CUDA 9.2+
- GCC 5+
- MMCV (`mmcv-full` is needed)

### Note
- If you want to use Python 3.6, `python3.6-dev` is needed  (instead of `python3-dev`) to escape from error of `pycocotools` (`pycocotools/_mask.c:6:10: fatal error: Python.h: No such file or directory`).

## Preparation
### Dataset
Python training script placed under `src/scripts/train.py` requires StrawDI dataset and converted annotation to run. Please following the instructions bellow.

- Download StrawDI dataset from [the webpage](https://strawdi.github.io/) and unzip downloaded file (It will create directory `StrawDI_Db1`).
- Rename the directory `StrawDI_Db1` to `strawdi` and place it under `data`. 
- Run `src/scripts/convert_annotation.py` to convert StrawDI's original segmentation annotation to detection annotation.

    ```bash
    % poetry run python src/scripts/convert_annotation.py --data_dir data/strawdi --save_dir tmp

    # Above script will generate three annotation files under `tmp`.
    % ls tmp

    test.json  train.json  val.json
    ```
- Make `annotations` directory under `data/strawdi` and move converted annotations to `data/strawdi/annotations`. 
    ```bash
    % mkdir data/strawdi/annotations
    % mv tmp/*.json data/strawdi/annotations
    % find data/strawdi  -name "*.json"

    data/strawdi/annotations/test.json
    data/strawdi/annotations/val.json
    data/strawdi/annotations/train.json
    ```

## Usage

### Convert trainined model from Pytorch to ONNX

```bash
% python src/scripts/pytorch2onnx.py \
    checkpoint_path=${PATH_TO_CHECKPOINT} \
    image_path=${PATH_TO_SAMPLE_IMAGE}
```

### Predict by ONNX model

```bash
% python src/scripts/predict.py \
    checkpoint_path=${PATH_TO_ONNX_CHECKPOINT} \
    test_image_dir=${PATH_TO_DIR_IMAGES_ARE_PLACED}
```