from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class MedUnlabeledDataset(CustomDataset):
    CLASSES = ('black', 'red')
    PALETTE = [[0, 0, 0], [255, 255, 255]]
    #CLASSES = ('None','black', 'red')
    #PALETTE = [[255,255,255], [0, 0, 0], [128, 0, 0]]

    #def __init__(self, **kwargs):
    def __init__(self,
                 crop_pseudo_margins=None,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 **kwargs):
        assert kwargs.get('split') in [None, 'train']
        if 'split' in kwargs:
            kwargs.pop('split')
        super(MedUnlabeledDataset, self).__init__(
            img_suffix='.png',
            #seg_map_suffix='_labelTrainIds.png',
            seg_map_suffix='.png',
            split=None,
            **kwargs)
