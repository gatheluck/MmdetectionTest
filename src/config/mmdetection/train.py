_base_ = [
    "predefined/_base_/default_runtime.py",
    "model/yolov3_d53.py",
    "dataset/strawdi.py",
    "optimizer/sgd.py",
]

# runtime settings
runner = dict(type="EpochBasedRunner", max_epochs=20)
evaluation = dict(interval=1, metric=["bbox"])
