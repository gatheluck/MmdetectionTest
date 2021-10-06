dataset_type = "CocoDataset"
classes = ("strawberry",)

data_root = "data/strawdi/"

train_ann_file = data_root + "annotations/train.json"
val_ann_file = data_root + "annotations/val.json"
test_ann_file = data_root + "annotations/test.json"

train_img_prefix = data_root + "train/"
val_img_prefix = data_root + "val/"
test_img_prefix = data_root + "test/"

mean = [0, 0, 0]
std = [255.0, 255.0, 255.0]

img_norm_cfg = dict(mean=mean, std=std, to_rgb=True)
train_pipeline = [
    dict(type="LoadImageFromFile", to_float32=True),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="PhotoMetricDistortion"),
    dict(
        type="Expand",
        mean=img_norm_cfg["mean"],
        to_rgb=img_norm_cfg["to_rgb"],
        ratio_range=(1, 2),
    ),
    dict(
        type="MinIoURandomCrop",
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3,
    ),
    dict(type="Resize", img_scale=[(320, 320), (608, 608)], keep_ratio=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(608, 608),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=train_ann_file,
        img_prefix=train_img_prefix,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=val_ann_file,
        img_prefix=val_img_prefix,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=test_ann_file,
        img_prefix=test_img_prefix,
        pipeline=test_pipeline,
    ),
)