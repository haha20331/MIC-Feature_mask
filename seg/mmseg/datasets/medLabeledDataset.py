# Obtained from: https://github.com/lhoyer/DAFormer
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------


from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class MedLabeledDataset(CustomDataset):
    CLASSES = ('black', 'red')
    PALETTE = [[0, 0, 0], [128, 0, 0]]
    #CLASSES = ('None','black', 'red')
    #PALETTE = [[255,255,255], [0, 0, 0], [128, 0, 0]]

    def __init__(self, **kwargs):
        assert kwargs.get('split') in [None, 'train']
        if 'split' in kwargs:
            kwargs.pop('split')
        super(MedLabeledDataset, self).__init__(
            img_suffix='.png',
            #seg_map_suffix='_labelTrainIds.png',
            seg_map_suffix='.png',
            split=None,
            **kwargs)
