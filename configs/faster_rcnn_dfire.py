_base_ = [
    "mmdet::_base_/models/faster-rcnn_r50_fpn.py",
    "mmdet::_base_/schedules/schedule_1x.py",
    "mmdet::_base_/default_runtime.py",
]

COCO_ROOT  = r"D:\MLPrac\data\coco"
IMG_PREFIX = ""

data_root = COCO_ROOT

metainfo = dict(
    classes=("fire", "smoke"),
    palette=[(220, 20, 60), (128, 128, 128)],
)

backend_args = None

train_pipeline = [
    dict(type="LoadImageFromFile",  backend_args=backend_args),
    dict(type="LoadAnnotations",    with_bbox=True),
    dict(type="RandomFlip",         prob=0.5),
    dict(type="Resize",             scale=(640, 640), keep_ratio=True),
    dict(type="PackDetInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="Resize",            scale=(640, 640), keep_ratio=True),
    dict(type="LoadAnnotations",   with_bbox=True),
    dict(type="PackDetInputs"),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="CocoDataset",
        metainfo=metainfo,
        data_root=data_root,
        ann_file="dfire_train.json",
        data_prefix=dict(img=""),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="CocoDataset",
        metainfo=metainfo,
        data_root=data_root,
        ann_file="dfire_val.json",
        data_prefix=dict(img=""),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="CocoDataset",
        metainfo=metainfo,
        data_root=data_root,
        ann_file="dfire_test.json",
        data_prefix=dict(img=""),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)

val_evaluator = dict(
    type="CocoMetric",
    ann_file=f"{data_root}/dfire_val.json",
    metric="bbox",
    format_only=False,
    backend_args=backend_args,
)

test_evaluator = dict(
    type="CocoMetric",
    ann_file=f"{data_root}/dfire_test.json",
    metric="bbox",
    format_only=False,
    backend_args=backend_args,
)

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=2)  # fire + smoke
    )
)

# Расписание
max_epochs = 24
train_cfg  = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=4)
val_cfg    = dict(type="ValLoop")
test_cfg   = dict(type="TestLoop")

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="SGD", lr=0.005, momentum=0.9, weight_decay=0.0005),
)

param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type="MultiStepLR", begin=0, end=max_epochs, by_epoch=True, milestones=[16, 22], gamma=0.1),
]

# Логирование
work_dir   = r"D:\MLPrac\outputs\runs\frcnn_dfire"
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    checkpoint=dict(type="CheckpointHook", interval=4, save_best="coco/bbox_mAP"),
)

load_from = None