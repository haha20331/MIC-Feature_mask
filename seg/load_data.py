import os
import mmcv
import numpy as np
import os.path as osp
from PIL import Image

def save_visualized_semantic_seg(seg_map, save_dir, file_name):
    palette = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
               [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
               [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
               [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
               [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
               [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
               [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
               [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
               [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
               [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
               [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
               [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
               [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
               [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
               [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255]]
    palette_flat = [item for sublist in palette for item in sublist]
    palette_flat = palette_flat + [0] * (256 * 3 - len(palette_flat))  # Ensure palette has 256*3 elements
    palette_bytes = bytes(palette_flat)
    seg_map = Image.fromarray(seg_map, 'P')
    seg_map.putpalette(palette_bytes)
    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 拼接文件路径
    save_path = os.path.join(save_dir, file_name)
    # 保存图像
    seg_map.save(save_path)
    print(f"Saved visualization to: {save_path}")


class LoadAnnotations(object):
    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = False 
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        
        # print("correct")
        filename = "./2007_000529.png"
        img_bytes = self.file_client.get(filename)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        save_visualized_semantic_seg(gt_semantic_seg,"./","new_2007_000529.png")

        # print("wrong")
        # filename = "test_data/red_11_65536_65536.png"
        # img_bytes = self.file_client.get(filename)
        # gt_semantic_seg = mmcv.imfrombytes(
        #     img_bytes, flag='unchanged',
        #     backend=self.imdecode_backend).squeeze().astype(np.uint8)
        # save_visualized_semantic_seg(gt_semantic_seg,"./","red_11_65536_65536.png")


load=LoadAnnotations()
load()
