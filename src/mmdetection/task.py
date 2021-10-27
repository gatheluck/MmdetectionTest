import logging
import pathlib
import sys
from typing import Dict, Iterable, List

import numpy as np
import torch

# If you use try instead of if, it will raise mypy error,
# For detail please check following;
# https://www.gitmemory.com/issue/python/mypy/9856/752267294
if sys.version_info >= (3, 8):
    from typing import Final
else:
    from typing_extensions import Final

from mmdet.core.export.model_wrappers import ONNXRuntimeDetector
from mmdet.models.detectors.base import BaseDetector

from mmcv.parallel import MMDataParallel

logger = logging.getLogger(__name__)


class OnnxDetector:
    """ONNX model for object detection by MM Detection.

    Attributes:
        onnx_model (BaseDetector): An trained onnx detector.
        labels (Iterable[str]): An iterable of labels.

    """

    def __init__(
        self,
        checkpoint_path: pathlib.Path,
        labels: Iterable[str] = ("strawberry",),
        device_id: int = 0,
    ) -> None:
        """

        Args:
            checkpoint_path (pathlib.Path): A path to onnx model checkpint.
            labels (Iterable[str]): An iterable of labels.
            device_id (int): An id of device to use.

        Returns:
            List[np.ndarray]: A prediction result. Fist four float values
                in np.ndarray represent bbox and other values represents
                logit of classification.

        """
        model: Final = ONNXRuntimeDetector(
            str(checkpoint_path), class_names=labels, device_id=device_id
        )
        self._cuda_availability_check(model)
        self.onnx_model: Final = MMDataParallel(model, device_ids=[device_id])
        self.labels: Final = labels

    def predict(self, x: torch.Tensor, meta: Dict) -> List[np.ndarray]:
        """Execute prediction by ONNX model.

        Args:
            x (torch.Tensor): A input tensor. Its shape should be
                (B, C, H, W) order.

            meta (Dict): A dict of meta info about input.

        Returns:
            List[np.ndarray]: A prediction result. Fist four float values
                in np.ndarray represent bbox and other values represents
                logit of classification.

        Raises:
            ValueError: If input tensor shape is invalid or if batch size is
                not one.

        """
        if len(x.size()) != 4:
            logger.error("Input shape is invalid. it should be (B, C, H, W).")
            raise ValueError()
        if x.size(0) != 1:
            logger.error("Currently only single batch is supported.")
            raise ValueError()

        with torch.no_grad():
            result = self.onnx_model(
                return_loss=False,
                rescalse=True,
                img=[x.cuda().contiguous()],
                img_metas=[[meta]],
            )

        # Although return type hint looks correct, mypy cannot handle return
        # type from self.onnx_model correctly. So we need to add ignore to
        # following.
        return result[0]  # type: ignore

    @classmethod
    def _cuda_availability_check(cls, model: BaseDetector) -> None:
        """Check if cuda is available on the input model.

        Args:
            model (BaseDetector): A mmdetection detector class.

        """
        if not model.is_cuda_available:
            logging.error("This model deos not support cuda.")
            raise ValueError()
