"""

Download pretrained weight from MM Detection.

Usage:
$ python src/script/download_pretrained_weights.py -O ${PATH_TO_SAVE_DIR}

"""
import logging
import pathlib
import sys
import urllib.request

# If you use try instead of if, it will raise mypy error,
# For detail please check following;
# https://www.gitmemory.com/issue/python/mypy/9856/752267294
if sys.version_info >= (3, 8):
    from typing import Final
else:
    from typing_extensions import Final

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

WEIGHTS_URL: Final = {
    "yolov3_d53_320_273e_coco": "https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_320_273e_coco/yolov3_d53_320_273e_coco-421362b6.pth",
    "yolov3_d53_mstrain-416_273e_coco": "https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_mstrain-416_273e_coco/yolov3_d53_mstrain-416_273e_coco-2b60fcd9.pth",
    "yolov3_d53_mstrain-608_273e_coco": "https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_mstrain-608_273e_coco/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth",
    "yolov3_d53_fp16_mstrain-608_273e_coco": "https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_fp16_mstrain-608_273e_coco/yolov3_d53_fp16_mstrain-608_273e_coco_20210517_213542-4bc34944.pth",
    "yolov3_mobilenetv2_mstrain-416_300e_coco": "https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_mobilenetv2_mstrain-416_300e_coco/yolov3_mobilenetv2_mstrain-416_300e_coco_20210718_010823-f68a07b3.pth",
    "yolov3_mobilenetv2_320_300e_coco": "https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_mobilenetv2_320_300e_coco/yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth",
}


def download_and_save(url: str, save_path: pathlib.Path) -> None:
    """Download file from url and save.

    Args:
        url (str): A target url to download file.
        save_path (pathlib.Path): A path whitch downloaded file is saved.

    """
    logger.info(f"Downloading file from {url}.")
    urllib.request.urlretrieve(url, str(save_path))
    logger.info(f"Saved as {str(save_path)}.")


if __name__ == "__main__":
    default_savedir: Final = "models/mmdet_pretrained"

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-O", "--save_dir", type=str, default=default_savedir)
    args = parser.parse_args()

    save_dir: Final = pathlib.Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for weight_url in WEIGHTS_URL.values():
        save_path = save_dir / weight_url.split("/")[-1]
        if save_path.exists():
            logger.info(f"{str(save_path)} is already exists.")
            continue

        download_and_save(weight_url, save_path)
