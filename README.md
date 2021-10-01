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