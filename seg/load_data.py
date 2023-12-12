import os
import mmcv
import numpy as np
import os.path as osp
from PIL import Image

def save_visualized_semantic_seg(seg_map, save_dir, file_name):
    palette = [0,0,0,128,0,0]
    seg_map = Image.fromarray(seg_map, 'P')
    seg_map.putpalette(palette)
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
        filename = "./test_data/new_11_65536_65536.png"
        img_bytes = self.file_client.get(filename)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        save_visualized_semantic_seg(gt_semantic_seg,"./","new_red_11_65536_65536.png")

        # print("wrong")
        # filename = "test_data/red_11_65536_65536.png"
        # img_bytes = self.file_client.get(filename)
        # gt_semantic_seg = mmcv.imfrombytes(
        #     img_bytes, flag='unchanged',
        #     backend=self.imdecode_backend).squeeze().astype(np.uint8)
        # save_visualized_semantic_seg(gt_semantic_seg,"./","red_11_65536_65536.png")


load=LoadAnnotations()
load()
