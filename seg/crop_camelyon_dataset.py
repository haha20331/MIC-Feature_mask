import os
import sys
import cv2
import glob
import tifffile
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, TiffImagePlugin
from PIL import ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

def split_and_resize_tiff_PIL(input_img_path, input_mask_path, output_path, patch_size=1024, target_size=512):
    green_threshold = 200
    r_sub_g_gap = 5
    b_sub_g_gap = 5
    filename = input_img_path.split('/')[-1][:-4]
    if filename[0:4] == "test":
        output_file = "test"
    else:
        output_file = "train"
    # 讀取TIFF圖片
    img = Image.open(input_img_path)
    # mask = Image.open(input_mask_path)
    mask = np.array(tifffile.imread(input_mask_path, key=0))

    # 確保輸出路徑存在
    output_img_path = os.path.join(output_path, f"patched_images/{output_file}/{filename}")
    if not os.path.exists(output_img_path):
        os.makedirs(output_img_path)
    
    output_mask_path = os.path.join(output_path, f"patched_masks/{output_file}/{filename}")
    if not os.path.exists(output_mask_path):
        os.makedirs(output_mask_path)
    

    # 取得原始圖片的寬和高
    print(img.size)
    print(mask.shape)
    width, height = img.size

    # 計算要切分成多少個小塊
    rows = height // patch_size
    cols = width // patch_size
    num = 0
    print(rows, cols)

    for i in range(0, rows):
        for j in range(0, cols):
            if j % 10 == 0 :
                print("\r{}, {}".format(i, j), end='')
            # 計算每個小塊的左上角和右下角座標
            left = j * patch_size
            upper = i * patch_size
            right = left + patch_size
            lower = upper + patch_size

            # 切分小塊
            patch_img = img.crop((left, upper, right, lower))
            patch_img_array = np.array(patch_img)
            average_color = patch_img_array.mean(axis=(0, 1))
            if (average_color[1] < green_threshold) and (abs(average_color[0]-average_color[1])>r_sub_g_gap):
                patch_mask = mask[upper:lower, left:right]

                # 將小塊resize為目標大小
                # patch_img_resized = patch_img.resize((target_size, target_size))
                # patch_mask_resized = cv2.resize(patch_mask, (target_size, target_size))

                # 儲存切分後並resize的小塊
                output_filename = f"{filename}_{i}_{j}.png"

                output_img_filepath = os.path.join(output_img_path, output_filename)
                patch_img.save(output_img_filepath)

                output_mask_filepath = os.path.join(output_mask_path, output_filename)
                # patch_mask_resized.save(output_mask_filepath)
                cv2.imwrite(output_mask_filepath, patch_mask)
                num+=1
            # else: 
            #     print("pass")
    print("  get",num,'/' ,i*j)
    

    img.close()
    del mask
    
    
    return height, width


def test(img_path):
    img = Image.open(img_path)
    img = np.array(img)
    print(img.shape)


if __name__ == "__main__":
    img_tif_path = glob.glob("/mnt/Nami/Bill0041/datasets/camelyon16/images/*.tif")
    output_directory = "/mnt/Nami/yea/camelyon16/"

    tqdm_iter = tqdm(img_tif_path[0:])
    for input_img_path in tqdm_iter:
        print(input_img_path)
        # if "test" in input_img_path: continue
        input_mask_path = input_img_path[:-4].replace("images", "masks") + "_mask.tif"
        height, width = split_and_resize_tiff_PIL(input_img_path, input_mask_path, output_directory)
        tqdm_iter.set_description("({}, {})".format(height, width))
        # test(input_img_path)

