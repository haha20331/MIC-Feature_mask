# Obtained from: https://github.com/lhoyer/HRDA
# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# dataset settings
dataset_type = 'MedUnlabeledDataset'
data_root = 'data/Med_unlabeled/'
img_norm_cfg = dict(
    mean=[235.491, 174.961, 223.921], std=[19.454, 41.481, 15.969], to_rgb=True)
#crop_size = (1024, 1024)
crop_size = (512, 512)
gta_train_pipeline = [
    dict(type='LoadImageFromFile'),
    #dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512)),
    #dict(type='Resize', img_scale=(2560, 1440)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

cityscapes_train_pipeline = [
    dict(type='LoadImageFromFile'),
    #dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512)),
    #dict(type='Resize', img_scale=(2048, 1024)),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    #dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        #img_scale=(2048, 1024),
        # MultiScaleFlipAug is disabled by not providing img_ratios and
        # setting flip=False
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='UDADataset',
        source=dict(
            type='MedLabeledDataset',
            data_root='data/Med_labeled/',
            img_dir='all_6~11_images',
            ann_dir='all_6~11_labels',
            pipeline=gta_train_pipeline),
        target=dict(
            type='MedUnlabeledDataset',
            data_root='data/Med_unlabeled/',
            img_dir='leftImg8bit/12~49_train',
            ann_dir='gtFine/train',
            pipeline=cityscapes_train_pipeline)),
    val=dict(
        type='MedUnlabeledDataset',
        data_root='data/Med_unlabeled/',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline),
    test=dict(
        type='MedUnlabeledDataset',
        data_root='data/Med_unlabeled/',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline))
