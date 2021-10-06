import logging
import pathlib
from typing import List, Tuple

import numpy as np
from mmdet.models.detectors.base import BaseDetector

logger = logging.getLogger(__name__)


def save_as_image(
    detector: BaseDetector,
    image_path: pathlib.Path,
    output_path: pathlib.Path,
    result: List[np.ndarray],
    score_thr: float,
    bbox_color: Tuple[int, int, int] = (72, 101, 241),
    text_color: Tuple[int, int, int] = (72, 101, 241),
) -> None:
    """Save prediction result as an image.

    Args:
        detector (BaseDetector): A pretrained mmdet detector.
        image_path (pathlib.Path): A path of test image.
        output_path (pathlib.Path): A path of output.
        result (List[np.ndarray]): A prediction result.
        score_thr (float): A threshold of prediction.
        bbox_color (Tuple[int, int, int]): A RGB color of bounding box.
        text_color (Tuple[int, int, int]): A RGB color of test.

    Raises:
        ValueError: If `image_path` does not exist.

    """
    if not image_path.exists():
        logger.error(f"path `{str(image_path)}` does not exist.")
        raise ValueError

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(detector, "module"):
        detector = detector.module

    detector.show_result(
        image_path,
        result,
        score_thr=score_thr,
        show=False,
        wait_time=0,
        win_name="",
        bbox_color=bbox_color,
        text_color=text_color,
        out_file=str(output_path),
    )
