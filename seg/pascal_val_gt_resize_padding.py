from PIL import Image, ImageOps
import os
import glob

# 定义输入和输出路径
input_folder = '/home/yea0826/MIC-master/MIC-Feature_mask/seg/data/pascal_unlabeled/1464_gt/val_orgin'
output_folder = '/home/yea0826/MIC-master/MIC-Feature_mask/seg/data/pascal_unlabeled/1464_gt/val'

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取输入文件夹中的所有PNG文件
input_images = glob.glob(os.path.join(input_folder, '*.png'))

for input_image_path in input_images:
    # 读取图像
    img = Image.open(input_image_path)

    # 将图像调整为512x512像素，保持原比例
    # 首先，找到合适的缩放尺寸，以保持图像比例
    img.thumbnail((512, 512), Image.ANTIALIAS)

    # 创建一个新的512x512像素的图像，填充颜色为255（白色）
    new_img = Image.new('L', (512, 512), 255)

    # 找到图像居中对齐的位置
    left = 0
    top = 0
    right = left + img.width
    bottom = top + img.height

    # 将缩放后的图像粘贴到新的图像中
    new_img.paste(img, (left, top, right, bottom))

    # 保存结果图像
    output_image_path = os.path.join(output_folder, os.path.basename(input_image_path))
    new_img.save(output_image_path)
    # print(f"Resized and padded image saved to {output_image_path}")
