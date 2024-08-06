# Obtained from: https://github.com/lhoyer/DAFormer
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------


from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class CamelyonDataset(CustomDataset):
    CLASSES = ('Normal', 'Benign','Tumor')
    PALETTE = [[0, 0, 0],[128,128,128], [255, 255, 255]]
    #CLASSES = ('None','black', 'red')
    #PALETTE = [[255,255,255], [0, 0, 0], [128, 0, 0]]

    def __init__(self, crop_pseudo_margins=None,**kwargs):
        assert kwargs.get('split') in [None, 'train']
        if 'split' in kwargs:
            kwargs.pop('split')
        super(CamelyonDataset, self).__init__(
            img_suffix='.png',
            #seg_map_suffix='_labelTrainIds.png',
            seg_map_suffix='.png',
            split=None,
            **kwargs)
        self.pseudo_margins = crop_pseudo_margins
        self.valid_mask_size = [1024, 1024]
