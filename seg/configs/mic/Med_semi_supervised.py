# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------


#查看每個檔案的config內容
_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture
    '../_base_/models/med_daformer_sepaspp_mitb5.py',
    # GTA->Cityscapes High-Resolution Data Loading
    '../_base_/datasets/1_4_med.py',
    # DAFormer Self-Training
    '../_base_/uda/dacs_a999_fdthings.py',
    # AdamW Optimizer
    '../_base_/schedules/adamw.py',
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    '../_base_/schedules/poly10warm.py'
]
# Random Seed
seed = 2  # seed with median performance
# HRDA Configuration
model = dict(
    type='HRDAEncoderDecoder',
    decode_head=dict(
        type='HRDAHead',
        # Use the DAFormer decoder for each scale.
        single_scale_head='DAFormerHead',
        # Learn a scale attention for each class channel of the prediction.
        attention_classwise=True,
        # Set the detail loss weight $\lambda_d=0.1$.
        hr_loss_weight=0.1),
    # Use the full resolution for the detail crop and half the resolution for
    # the context crop.
    scales=[1, 0.5],
    # Use a relative crop size of 0.5 (=512/1024) for the detail crop.
########################## hr_crop_size ###########################
    hr_crop_size=(256, 256),
    # Use LR features for the Feature Distance as in the original DAFormer.
    feature_scale=0.5,
    # Make the crop coordinates divisible by 8 (output stride = 4,
    # downscale factor = 2) to ensure alignment during fusion.
    crop_coord_divisible=8,
    # Use overlapping slide inference for detail crops for pseudo-labels.
    hr_slide_inference=True,
    # Use overlapping slide inference for fused crops during test time.
    test_cfg=dict(
        mode='slide',
        batched_slide=True,
        stride=[128, 128],
        crop_size=[512, 512]
        ))
#64, 128
data = dict(
    train=dict(
        # Rare Class Sampling
        # min_crop_ratio=2.0 for HRDA instead of min_crop_ratio=0.5 for
        # DAFormer as HRDA is trained with twice the input resolution, which
        # means that the inputs have 4 times more pixels.
        rare_class_sampling=dict(
            min_pixels=3000, class_temp=0.01, min_crop_ratio=2.0),
        # Pseudo-Label Cropping v2 (from HRDA):
        # Generate mask of the pseudo-label margins in the data loader before
        # the image itself is cropped to ensure that the pseudo-label margins
        # are only masked out if the training crop is at the periphery of the
        # image.
        target=dict(crop_pseudo_margins=[30, 240, 30, 30]),
    ),
    # Use one separate thread/worker for data loading.
    workers_per_gpu=1,
    # Batch size
    samples_per_gpu=8,
)
# MIC Parameters
uda = dict(
    # Apply masking to color-augmented target images
    mask_mode='separatetrgaug',
    # Use the same teacher alpha for MIC as for DAFormer
    # self-training (0.999)
    mask_alpha='same',
    # Use the same pseudo label confidence threshold for
    # MIC as for DAFormer self-training (0.968)
    mask_pseudo_threshold='same',
    # Equal weighting of MIC loss
    mask_lambda=1,
    # Use random patch masking with a patch size of 64x64
    # and a mask ratio of 0.7
    # mask_generator=dict(
    #     type='block', mask_ratio=0.7, mask_block_size=64, _delete_=True),
######################### mask_block_size = 圖片經過transform後的大小/16 #########################
    mask_generator=dict(
        type='block', mask_ratio=0.7, mask_block_size=32, _delete_=True),

######################### feature mask ratio ##########################
    mask_feature_ratio=dict(flag=True, f1_ratio=0, f2_ratio=0.7, 
                            f3_ratio=0, f4_ratio=0),
    loss_weight = 1,
    student_consistency_loss_flag = True,#4個一起開
    student_mix_img_loss_flag = True,
    student_mask_feature_loss_flag = True,
    student_mask_img_loss_flag = True,
    mask_img_and_feature_loss_flag = False,#廢棄功能，常駐關
    A1_aug = {
            'color_jitter_flag': True,
            'blur_flag': True,
            # 'color_jitter_flag': False,
            # 'blur_flag': False,
        },
    A1_backward_flag = True,#常駐開
    enable_fdist_flag = False,
    )
# Optimizer Hyperparameters
optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))

n_gpus = 1
gpu_model = 'NVIDIATITANRTX'
runner = dict(type='IterBasedRunner', max_iters=40000)
# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=30000, max_keep_ckpts=1)
evaluation = dict(interval=250, metric=['mIoU', 'mDice'])
# Meta Information for Result Analysi
name = 'med_1_4_fmask=070_SC'
#name = 'med_1_4_supBaseline'
# name = 'test'
exp = 'basic'
name_dataset = 'Pascal_semisupervised'
name_architecture = 'hrda1-512-0.1_daformer_sepaspp_sl_mitb5'
name_encoder = 'mitb5'
name_decoder = 'hrda1-512-0.1_daformer_sepaspp_sl'
name_uda = 'dacs_a999_fdthings_rcs0.01-2.0_cpl2_m64-0.7-spta'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'

# For the other configurations used in the paper, please refer to experiment.py
