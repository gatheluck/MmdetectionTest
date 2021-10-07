"""

Run prediction by pretrained MM Detection detector. Detected Bounding
boxes are overrayed to input images.

Usage:
    $ python src/scripts/predict.py \
        mmcv_config_path=${PATH_TO_MMCV_CONFIG} \
        checkpoint_path=${PATH_TO_CHECKPOINT} \
        test_image_dir=${PATH_TO_DIR_IMAGES_ARE_PLACED}

"""
import logging
import pathlib
import sys

# If you use try instead of if, it will raise mypy error,
# For detail please check following;
# https://www.gitmemory.com/issue/python/mypy/9856/752267294
if sys.version_info >= (3, 8):
    from typing import Final
else:
    from typing_extensions import Final

import hydra
from mmdet.apis import inference_detector, init_detector
from omegaconf.omegaconf import DictConfig, OmegaConf

from src.mmdetection.save import save_as_image

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@hydra.main(config_path="../config/hydra", config_name="predict")
def main(cfg: DictConfig) -> None:
    """Predict detection result by pretrained detector.

    Args:
        cfg (DictConfig): A config loaded from yaml and CLI.

    """
    OmegaConf.set_readonly(cfg, True)
    logger.info("\n" + OmegaConf.to_yaml(cfg, resolve=True))

    cwd: Final = pathlib.Path(hydra.utils.get_original_cwd())
    mmcv_config_path: Final = cwd / cfg.mmcv_config_path
    checkpoint_path: Final = cwd / cfg.checkpoint_path

    try:
        test_image_dir: Final = cwd / cfg.test_image_dir
    except Exception:
        logger.error("`test_image_dir` is not specified.")
        raise ValueError

    detector: Final = init_detector(
        str(mmcv_config_path), str(checkpoint_path), device=cfg.device
    )

    supported_suffix: Final = {".png", ".jpg", ".jpeg"}
    for image_path in test_image_dir.glob("**/*"):
        if image_path.suffix not in supported_suffix:
            continue

        logger.info(f"predicting `{str(image_path)}`")
        result = inference_detector(detector, str(image_path))
        output_path = pathlib.Path("result_" + image_path.name)
        save_as_image(
            detector,
            image_path,
            output_path,
            result,
            score_thr=cfg.score_threshold,
        )


if __name__ == "__main__":
    main()
