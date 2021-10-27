"""

Run prediction by pretrained ONNX MM Detection detector.
Detected Bounding boxes are overrayed to input images.

Usage:
    $ python src/scripts/predict.py \
        checkpoint_path=${PATH_TO_ONNX_CHECKPOINT} \
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
from mmdet.core.export import preprocess_example_input
from omegaconf.omegaconf import DictConfig, OmegaConf

from mmcv import Config
from src.mmdetection.onnx_converter import PytorchToOnnxConverter
from src.mmdetection.save import save_as_image
from src.mmdetection.task import OnnxDetector

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@hydra.main(config_path="../config/hydra", config_name="predict_onnx")
def main(cfg: DictConfig) -> None:
    """Predict detection result by pretrained onnx detector.

    Args:
        cfg (DictConfig): A config loaded from yaml and CLI.

    """
    OmegaConf.set_readonly(cfg, True)
    logger.info("\n" + OmegaConf.to_yaml(cfg, resolve=True))

    cwd: Final = pathlib.Path(hydra.utils.get_original_cwd())
    mmcv_train_config_path: Final = cwd / cfg.mmcv_train_config_path
    checkpoint_path: Final = cwd / cfg.checkpoint_path
    test_image_dir: Final = cwd / cfg.test_image_dir

    input_shape: Final = (1, 3, cfg.input_height, cfg.input_width)
    test_pipeline_config: Final = Config.fromfile(
        str(mmcv_train_config_path)
    ).test_pipeline
    normalize_config: Final = PytorchToOnnxConverter._extract_normalize_config(
        test_pipeline_config
    )

    detector: Final = OnnxDetector(checkpoint_path)

    supported_suffix: Final = {".png", ".jpg", ".jpeg"}
    for image_path in test_image_dir.glob("**/*"):
        if image_path.suffix not in supported_suffix:
            continue

        logger.info(f"predicting `{str(image_path)}`")
        input_config = {
            "input_shape": input_shape,
            "input_path": str(image_path),
            "normalize_cfg": normalize_config,
        }
        img, meta = preprocess_example_input(input_config)
        result = detector.predict(img, meta)

        output_path = pathlib.Path("result_" + image_path.name)
        save_as_image(
            detector.onnx_model,
            meta["show_img"],
            output_path,
            result,
            score_thr=cfg.score_threshold,
        )


if __name__ == "__main__":
    main()
