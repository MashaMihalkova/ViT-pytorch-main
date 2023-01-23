import os
mri_labels = {0: "MildDemented ", 1: "ModerateDemented ", 2: "NonDemented ", 3: "VeryMildDemented "}

for p_f, dirs, files in os.walk('data/Alzheimer_s Dataset/train'):
    print(dirs)
    # for dir_ in dirs:
    for file in files:
            print(p_f+'/'+file)
            cl_gt = p_f.split('/')[-1]





