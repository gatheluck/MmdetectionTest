import logging
import pathlib
import sys
from functools import partial
from typing import List, Tuple

from mmdet.models.detectors.base import BaseDetector

import mmcv

# If you use try instead of if, it will raise mypy error,
# For detail please check following;
# https://www.gitmemory.com/issue/python/mypy/9856/752267294
if sys.version_info >= (3, 8):
    from typing import Final
else:
    from typing_extensions import Final

import torch
from mmdet.core.export import preprocess_example_input

logger = logging.getLogger(__name__)


class PytorchToOnnxConverter:
    OPSET_VERSION: Final = 11

    @classmethod
    def convert(
        cls,
        model: BaseDetector,
        input_shape: Tuple[int, int, int, int],
        pipeline_config: List[mmcv.utils.config.ConfigDict],
        is_dynamic_export: bool,
        image_path: pathlib.Path,
        output_path: pathlib.Path,
    ) -> None:
        """Convert mmdetection detector to onnx model.

        Args:
            model (BaseDetector): A mmdetection detector class.
            input_shape (Tuple[int, int, int, int]): A shape of input tensor.
            pipeline_config (List[mmcv.utils.config.ConfigDict]):
                A list of ConfigDict for pipeline.
            is_dynamic_export (bool): If Ture, converted model able to take
                any size of input.
            image_path (pathlib.Path): A path of sample input image.
                No need to match input shape.
            output_path (pathlib.Path): A path where converted model is saved.

        Raises:
            ValueError: If test pipeline config does not have `transforms` or
                `transforms` includes no or multiple normalize transform.

        """
        if model.with_mask:
            logger.error("Currently model with mask is not supported.")
            raise ValueError()

        normalize_config: Final = cls._extract_normalize_config(pipeline_config)
        input_config: Final = {
            "input_shape": input_shape,
            "input_path": str(image_path),
            "normalize_cfg": normalize_config,
        }

        # prepare input
        # image size is resized based on input shape
        one_img, one_meta = preprocess_example_input(input_config)
        img_list, img_meta_list = [one_img], [[one_meta]]

        # replace original forward function
        origin_forward = model.forward
        model.forward = partial(
            model.forward, img_metas=img_meta_list, return_loss=False, rescale=False
        )

        input_name: Final = "input"
        output_names: Final = ["dets", "labels"]
        dynamic_axes = None
        if is_dynamic_export:
            dynamic_axes = {
                input_name: {0: "batch", 2: "height", 3: "width"},
                "dets": {
                    0: "batch",
                    1: "num_dets",
                },
                "labels": {
                    0: "batch",
                    1: "num_dets",
                },
            }

        torch.onnx.export(
            model,
            img_list,
            str(output_path),
            input_names=[input_name],
            output_names=output_names,
            export_params=True,
            keep_initializers_as_inputs=True,
            do_constant_folding=True,
            verbose=True,
            opset_version=cls.OPSET_VERSION,
            dynamic_axes=dynamic_axes,
        )

        model.forward = origin_forward

    @classmethod
    def _extract_normalize_config(
        cls, pipeline: List[mmcv.utils.config.ConfigDict]
    ) -> mmcv.utils.config.ConfigDict:
        """Extract ConfigDict of normalize transform from pipeline config.

        Args:
            pipeline (List[mmcv.utils.config.ConfigDict]): A list of ConfigDict
                for pipeline.

        Returns:
            mmcv.utils.config.ConfigDict: A ConfigDict of normalize transform.

        Raises:
            ValueError: If test pipeline config does not have `transforms` or
                `transforms` includes no or multiple normalize transform.

        """
        transforms = None
        for process in pipeline:
            if "transforms" in process:
                transforms = process["transforms"]
                break

        if transforms is None:
            logger.error("Failed to find `transforms`.")
            raise ValueError()

        normalize_list: Final = [
            transform for transform in transforms if transform["type"] == "Normalize"
        ]

        if len(normalize_list) != 1:
            logger.error(f"'Normalize' should have one not {len(normalize_list)}.")
            raise ValueError()

        return normalize_list[0]
