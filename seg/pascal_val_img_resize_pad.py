from PIL import Image, ImageOps
import os

# 定义输入和输出路径
input_dir = '/home/yea0826/MIC-master/MIC-Feature_mask/seg/data/pascal_unlabeled/1464_img/val_orgin'
output_dir = '/home/yea0826/MIC-master/MIC-Feature_mask/seg/data/pascal_unlabeled/1464_img/val'

# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 遍历输入目录中的所有 JPG 文件
for filename in os.listdir(input_dir):
    if filename.endswith('.jpg'):
        input_image_path = os.path.join(input_dir, filename)
        output_image_path = os.path.join(output_dir, filename)

        # 读取图像
        img = Image.open(input_image_path)

        # 将图像调整为512x512像素，保持原比例
        # 首先，找到合适的缩放尺寸，以保持图像比例
        img.thumbnail((512, 512), Image.ANTIALIAS)

        # 创建一个新的512x512像素的图像，填充颜色为0（黑色）
        new_img = Image.new('RGB', (512, 512), (0, 0, 0))

        # 找到图像居中对齐的位置
        left = 0
        top = 0
        right = left + img.width
        bottom = top + img.height

        # 将缩放后的图像粘贴到新的图像中
        new_img.paste(img, (left, top, right, bottom))

        # 保存结果图像
        new_img.save(output_image_path)

        print(f"Resized and padded image saved to {output_image_path}")