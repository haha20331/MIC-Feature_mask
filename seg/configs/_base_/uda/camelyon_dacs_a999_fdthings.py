# Obtained from: https://github.com/lhoyer/DAFormer
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# UDA with Thing-Class ImageNet Feature Distance + Increased Alpha
_base_ = ['camelyon_dacs.py']
uda = dict(
    alpha=0.999,
    imnet_feature_dist_lambda=0.005,
    imnet_feature_dist_classes=[0,1,2],
    imnet_feature_dist_scale_min_ratio=0.75,
)
