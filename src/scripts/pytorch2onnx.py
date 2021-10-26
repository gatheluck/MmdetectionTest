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
from mmdet.core.export import build_model_from_cfg
from omegaconf.omegaconf import DictConfig, OmegaConf

from mmcv import Config
from src.mmdetection.onnx_converter import PytorchToOnnxConverter

logger = logging.getLogger(__name__)


@hydra.main(config_path="../config/hydra", config_name="pytorch2onnx")
def main(cfg: DictConfig) -> None:
    """Convert pretrained MM Detection detector from PyTorch to ONNX.

    Args:
        cfg (DictConfig): A config loaded from yaml and CLI.

    """
    OmegaConf.set_readonly(cfg, True)

    cwd: Final = pathlib.Path(hydra.utils.get_original_cwd())
    mmcv_train_config_path: Final = cwd / cfg.mmcv_train_config_path
    checkpoint_path: Final = cwd / cfg.checkpoint_path
    image_path: Final = cwd / cfg.image_path

    input_shape: Final = (1, 3, cfg.input_height, cfg.input_width)
    test_pipeline_config: Final = Config.fromfile(
        str(mmcv_train_config_path)
    ).test_pipeline

    model = build_model_from_cfg(str(mmcv_train_config_path), str(checkpoint_path))

    PytorchToOnnxConverter.convert(
        model=model,
        input_shape=input_shape,
        pipeline_config=test_pipeline_config,
        is_dynamic_export=cfg.is_dynamic_export,
        image_path=image_path,
        output_path=pathlib.Path("./checkpoint.onnx"),
    )


if __name__ == "__main__":
    main()
