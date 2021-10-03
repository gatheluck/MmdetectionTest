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
from omegaconf.omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def save_result(
    detector,
    image_path,
    result,
    score_thr,
    output_path,
) -> None:
    if hasattr(detector, "module"):
        detector = detector.module

    detector.show_result(
        image_path,
        result,
        score_thr=score_thr,
        show=False,
        wait_time=0,
        win_name="title",
        bbox_color=(72, 101, 241),
        text_color=(72, 101, 241),
        out_file=output_path,
    )


@hydra.main(config_path="../config/hydra", config_name="predict")
def main(cfg) -> None:
    OmegaConf.set_readonly(cfg, True)
    logger.info("\n" + OmegaConf.to_yaml(cfg, resolve=True))

    cwd: Final = pathlib.Path(hydra.utils.get_original_cwd())
    mmdet_config_path: Final = cwd / cfg.mmdet_config_path
    checkpoint_path: Final = cwd / cfg.checkpoint_path

    try:
        input_image_dir: Final = cwd / cfg.input_image_dir
    except Exception:
        logger.error("input_image_dir is not specified.")
        raise ValueError

    detector = init_detector(
        str(mmdet_config_path), str(checkpoint_path), device=cfg.device
    )

    supported_suffix: Final = {".png", ".jpg", ".jpeg"}
    for input_image_path in input_image_dir.glob("**/*"):
        if input_image_path.suffix not in supported_suffix:
            continue

        logger.info(f"predicting `{str(input_image_path)}`")
        result = inference_detector(detector, str(input_image_path))
        output_path = "result_" + input_image_path.name
        save_result(
            detector, input_image_path, result, score_thr=0.05, output_path=output_path
        )


if __name__ == "__main__":
    main()
