import os
import shutil
dataset_pth = "./rabbit-breed-classification/train/"

img_list = os.listdir(dataset_pth)
img_list.sort()
img_org = []
img_save = []
for i, fn in enumerate(img_list):
    if i % 3 == 1:
        img_org.append(fn)
        img_save.append(fn[:fn.find("_")] + ".jpg")
    else:
        continue
    
save_pth = "target_domain/rabbit_breed/"
if not os.path.exists(save_pth):
    os.mkdir(save_pth)

class_name = ["californian", "holland-lop", "lionhead", "new-zealand"]
for i, name in enumerate(class_name): 
    if not os.path.exists(os.path.join(save_pth, name)):
        os.mkdir(os.path.join(save_pth, name))


for i, fn in enumerate(img_save):
    
    shutil.copyfile(os.path.join(dataset_pth, img_org[i]), os.path.join(save_pth, class_name[i // 80], fn))