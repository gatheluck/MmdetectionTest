"""

Run training of MM Detection detector.

Usage:
    $ python src/scripts/train.py

Note:
    This script referes `src/config/hydra/train.yaml`.

"""
import pathlib
import sys

# If you use try instead of if, it will raise mypy error,
# For detail please check following;
# https://www.gitmemory.com/issue/python/mypy/9856/752267294
if sys.version_info >= (3, 8):
    from typing import Final
else:
    from typing_extensions import Final

import copy
import time

import hydra
from mmdet import __version__
from mmdet.apis import train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import get_root_logger
from omegaconf.omegaconf import DictConfig, OmegaConf

from mmcv.utils import get_git_hash
from src.mmcv.config import ConfigConverter


@hydra.main(config_path="../config/hydra", config_name="train")
def main(cfg: DictConfig) -> None:
    """Train MM Detection detector.

    Args:
        cfg (DictConfig): A config loaded from yaml and CLI.

    """
    OmegaConf.set_readonly(cfg, True)

    # Convert hydra config to mmdet config
    cwd: Final = pathlib.Path(hydra.utils.get_original_cwd())
    mmcv_config, meta = ConfigConverter.from_hydra(cfg, cwd)
    mmcv_config.dump(mmcv_config.exp_name)

    # Log some basic info
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    dash_line = "-" * 60 + "\n"
    log_file = f"{timestamp}.log"
    logger = get_root_logger(log_file=log_file, log_level=mmcv_config.log_level)
    logger.info("Environment info:\n" + dash_line + meta["env_info"] + "\n" + dash_line)
    logger.info(f"Distributed training: {mmcv_config.distributed}")
    logger.info(f"Config:\n{mmcv_config.pretty_text}")

    # Setup dataset
    datasets = [build_dataset(mmcv_config.data.train)]
    if len(mmcv_config.workflow) == 2:
        val_dataset = copy.deepcopy(mmcv_config.data.val)
        val_dataset.pipeline = mmcv_config.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if mmcv_config.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        mmcv_config.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7], CLASSES=datasets[0].CLASSES
        )

    # Setup model
    model = build_detector(
        mmcv_config.model,
        train_cfg=mmcv_config.get("train_cfg"),
        test_cfg=mmcv_config.get("test_cfg"),
    )
    model.init_weights()
    model.CLASSES = datasets[0].CLASSES

    train_detector(
        model,
        datasets,
        mmcv_config,
        distributed=mmcv_config.distributed,
        validate=True,
        timestamp=timestamp,
        meta=meta,
    )


if __name__ == "__main__":
    main()
