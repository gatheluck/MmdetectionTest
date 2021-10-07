"""

Convert segmentation annotation of StrawDI dataset to COCO format
object detection dataset. If you want to run the code, please download
StrawDI dataset from https://strawdi.github.io/ and place unziped
dataset somewhere (We call the directory as ${DATA_DIR}).

Usage:
    $ python src/scripts/convert_annotation.py \
        --data_dir ${PATH_TO_DATA_DIR} --save_dir ${PATH_TO_SAVE_DIR}

Note:
    This code hyposizes that directory composition of StrawDi dataset
    like following:

    ${DATA_DIR}/strawdi/train/img/*.png
                       /label/*.png

                       /val/img/*.png
                       /label/*.png

                       /test/img/*.png
                       /label/*.png

"""
import json
import logging
import pathlib
import sys
from datetime import datetime
from typing import Collection, Dict, List, Optional, Tuple

import numpy as np
import tqdm
from imgaug.augmentables.bbs import BoundingBox
from PIL import Image
from pydantic import BaseModel

# If you use try instead of if, it will raise mypy error,
# For detail please check following;
# https://www.gitmemory.com/issue/python/mypy/9856/752267294
if sys.version_info >= (3, 8):
    from typing import Final
else:
    from typing_extensions import Final

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class CocoInfo(BaseModel):
    year: int
    version: str
    description: str
    contributor: str
    url: str
    date_created: str


class CocoImage(BaseModel):
    id: int
    width: int
    height: int
    file_name: str
    license: int
    flickr_url: str
    coco_url: str
    date_captured: str


class CocoDetectionAnnotation(BaseModel):
    id: int
    image_id: int
    category_id: int
    segmentation: Optional[List[List[float]]]
    area: float
    bbox: List[float]
    iscrowd: int


class CocoCategory(BaseModel):
    id: int
    name: str
    supercategory: str


class CocoLicense(BaseModel):
    id: int
    name: str
    url: str


class CocoAnnotation(BaseModel):
    info: Optional[CocoInfo]
    images: List[CocoImage]
    annotations: List[CocoDetectionAnnotation]
    categories: List[CocoCategory]
    licenses: List[CocoLicense]


class Converter:
    """

    Convert StrawDi segmeatation annotation to COCO format object
    detection annotation. About COCO format annotation, please check
    following link: https://cocodataset.org/#format-data


    Attributes:
        annotations_id (int): An id for each annotation. This attribute
            is incremented inside of self._process_mask method.
        datetime_str (str): An datatime info. This attribute is used
            inside of self._process_image method.

    """

    def __init__(self) -> None:
        self.annotations_id = 1
        self.datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def __call__(self, data_dir: pathlib.Path, save_dir: pathlib.Path) -> None:
        """Entry point of Converter.

        Args:
            data_dir (pathlib.Path): A path to directory where
                the dataset is placed.
            save_dir (pathlib.Path): A path to directory where
                the results will be saved.

        """

        target_file_paths: Final[Dict[str, Tuple[pathlib.Path, pathlib.Path]]] = {
            "train": (
                data_dir / "strawdi/train/img",
                data_dir / "strawdi/train/label",
            ),
            "val": (
                data_dir / "strawdi/val/img",
                data_dir / "strawdi/val/label",
            ),
            "test": (
                data_dir / "strawdi/test/img",
                data_dir / "strawdi/test/label",
            ),
        }

        save_dir.mkdir(parents=True, exist_ok=True)

        for data_type, (image_dir_path, label_dir_path) in target_file_paths.items():
            logger.info(f"Processing {data_type} data.")
            image_paths = sorted(
                image_dir_path.glob("*.png"), key=lambda p: int(p.stem)
            )
            label_paths = sorted(
                label_dir_path.glob("*.png"), key=lambda p: int(p.stem)
            )

            coco_format_annotation = self._convert(image_paths, label_paths)

            # Save converted annotation as json.
            output_path = save_dir / (data_type + ".json")
            with open(str(output_path), "w") as f:
                json.dump(coco_format_annotation.dict(), f)
            logger.info(f"saved to {str(output_path)}.")

    def _convert(
        self,
        image_paths: Collection[pathlib.Path],
        label_paths: Collection[pathlib.Path],
    ) -> CocoAnnotation:
        """Convert to COCO style object detection annotation.

        Args:
            image_paths (Collection[pathlib.Path]): A Collection
                of image path.
            label_paths (Collection[pathlib.Path]): A Collection
                of label path.

        Returns:
            CocoAnnotation: A dict of COCO format annotation.

        Raises:
            ValueError: If the length of `image_paths` and `label_paths`
                are different.
            ValueError: If the name of image and label file is not same.

        """
        coco_lisence: Final = CocoLicense(
            id=1,
            name="",
            url="",
        )
        coco_category: Final = CocoCategory(
            id=1,
            name="strawberry",
            supercategory="food",
        )
        coco_images: List[CocoImage] = list()
        coco_annotations: List[CocoDetectionAnnotation] = list()

        if len(image_paths) != len(label_paths):
            logger.error("`image_paths` and `label_paths` must be same len.")
            raise ValueError

        logger.info(f"found {len(image_paths)} data.")

        bar = tqdm.tqdm(total=len(image_paths))
        for image_path, label_path in zip(image_paths, label_paths):
            if image_path.name != label_path.name:
                logger.error("A file name of image and label must be same.")
                raise ValueError

            coco_images.extend(self._process_image(image_path))
            coco_annotations.extend(self._process_label(label_path))
            bar.update(1)

        return CocoAnnotation(
            info=None,
            images=coco_images,
            annotations=coco_annotations,
            categories=[coco_category],
            licenses=[coco_lisence],
        )

    def _process_image(
        self,
        image_path: pathlib.Path,
    ) -> List[CocoImage]:
        """Create COCO format annotation from a single image. A return
        value of this method should be added to "images" section.

        Args:
            image_path (pathlib.Path): A path of image.

        Returns:
            List[CocoImage]: A COCO format annotation.

        """
        image: Final = np.array(Image.open(str(image_path)))
        return [
            CocoImage(
                id=int(image_path.stem),
                width=int(image.shape[0]),
                height=int(image.shape[1]),
                file_name=image_path.name,
                license=1,
                flickr_url="",
                coco_url="",
                date_captured=self.datetime_str,
            )
        ]

    def _process_label(
        self,
        label_path: pathlib.Path,
    ) -> List[CocoDetectionAnnotation]:
        """Create COCO format annotation from a single label. A return
        value of this method should be added to "annotations" section.

        Args:
            label_path (pathlib.Path): A path of label.

        Returns:
            List[CocoDetectionAnnotation]: A COCO format annotation.

        """
        mask: Final = np.array(Image.open(str(label_path)))
        object_ids: Final = np.unique(mask[mask > 0])

        annotations = list()
        for object_id in object_ids:
            height_max = np.where(mask == object_id)[0].max()
            height_min = np.where(mask == object_id)[0].min()
            width_max = np.where(mask == object_id)[1].max()
            width_min = np.where(mask == object_id)[1].min()

            bounding_box = BoundingBox(
                x1=width_min, y1=height_min, x2=width_max, y2=height_max
            )

            annotations.append(
                CocoDetectionAnnotation(
                    id=self.annotations_id,
                    image_id=int(label_path.stem),
                    category_id=1,
                    area=float(bounding_box.area),
                    bbox=[
                        float(bounding_box.x1),
                        float(bounding_box.y1),
                        float(bounding_box.width),
                        float(bounding_box.height),
                    ],
                    iscrowd=0,
                )
            )
            self.annotations_id += 1

        return annotations


if __name__ == "__main__":
    default_data_dir: Final = "data"
    default_save_dir: Final = "tmp"

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, default=default_data_dir)
    parser.add_argument("-O", "--save_dir", type=str, default=default_save_dir)
    args = parser.parse_args()

    data_dir: Final = pathlib.Path(args.data_dir)
    save_dir: Final = pathlib.Path(args.save_dir)
    Converter()(data_dir, save_dir)
