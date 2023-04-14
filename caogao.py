import os
import shutil

path = "/root/yujiaxin/yujiaxin/ROPE1MonoDETR-main/runs/monodetr/"
for folder_name in os.listdir(path):
    if folder_name.startswith("train.log.20230223"):
        folder_path = os.path.join(path, folder_name)
        os.remove(folder_path)
 