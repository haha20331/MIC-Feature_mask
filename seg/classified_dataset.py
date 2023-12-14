import os
import shutil

data_dir_name = "1_8_unlabeled.txt"
labeled_unlabeled = "City_unlabeled"
# 指定目标目录
destination_folder = '/home/yea0826/MIC-master/MIC-Feature_mask/seg/data/' + labeled_unlabeled + '/'

# 读取 labeled.txt 文件
with open(data_dir_name, 'r') as file:
    # 逐行读取文件内容
    lines = file.readlines()

# 遍历每一行，处理源文件和目标文件路径
for line in lines:
    # 使用 split 方法按空格分割字符串
    result = line.strip().split(' ')

    # 源文件路径和目标文件路径

    destination_path_img = os.path.join(destination_folder+"12_5_img", result[0][12:])
    destination_path_json = os.path.join(destination_folder+"12_5_gt", result[1][7:-17]+"polygons.json")
    result[1] = result[1][:-17] + "polygons.json"
    print(result[0],destination_path_img)
    print(result[1],destination_path_json)
    # 确保目标目录存在 
    os.makedirs(os.path.dirname(destination_path_img), exist_ok=True)
    os.makedirs(os.path.dirname(destination_path_json), exist_ok=True)
    # 移动文件
    shutil.copy(result[0], destination_path_img)
    shutil.copy(result[1], destination_path_json)
    # 输出移动的文件信息（可选）
    #print(f"Moved: {result[0]} to {destination_path_img}")
    #print(f"Moved: {result[1]} to {destination_path_json}")