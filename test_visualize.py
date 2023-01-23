import typing
import io
import os

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from urllib.request import urlretrieve

from PIL import Image
from torchvision import transforms

from models.modeling import VisionTransformer, CONFIGS
from torchvision import models


def load_state(model_path):
    state_dict = torch.load(model_path)['model']
    state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
    state_dict = {k[6:] if k.startswith('model.') else k: state_dict[k] for k in state_dict.keys()}

    return state_dict

# os.makedirs("attention_data", exist_ok=True)
# if not os.path.isfile("attention_data/ilsvrc2012_wordnet_lemmas.txt"):
#     urlretrieve("https://storage.googleapis.com/bit_models/ilsvrc2012_wordnet_lemmas.txt", "attention_data/ilsvrc2012_wordnet_lemmas.txt")
# if not os.path.isfile("attention_data/ViT-B_16-224.npz"):
#     urlretrieve("https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-B_16-224.npz", "attention_data/ViT-B_16-224.npz")

mri_labels = dict(enumerate(open('attention_data/ilsvrc2012_wordnet_lemmas.txt')))

# mri_labels = {"MildDemented": 0, "ModerateDemented": 1, "NonDemented": 2, "VeryMildDemented": 3}
# mri_labels = {0: "MildDemented", 1: "ModerateDemented", 2: "NonDemented", 3: "VeryMildDemented"}

# Prepare Model
config = CONFIGS["ViT-B_16"]
model = VisionTransformer(config, num_classes=1000, zero_head=False, img_size=224, vis=True)
# model = models.resnet101(pretrained=True)
# model.fc = torch.nn.Linear(model.fc.in_features, 4)
w2 = 'checkpoint/ViT-B_16.npz'
# w2 = 'output/custom_alzheimer_checkpoint_59_5.pkl'
# checkpoint = torch.load(w2)
# model.load_state_dict(checkpoint)

model.load_from(np.load(w2))
model.eval()

transform = transforms.Compose([
            transforms.RandomResizedCrop((224, 224), scale=(0.05, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.ToTensor(),
            # transforms.Normalize((0.5), (0.5))
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # transforms.Resize((224, 224)),
    # transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
# path_im = "data/Alzheimer_s Dataset/val/MildDemented/26.jpg"
# path_im = "data/Alzheimer_s Dataset/val/ModerateDemented/27 (2).jpg"
# path_im = "data/Alzheimer_s Dataset/val/NonDemented/26 (65).jpg"
# path_im = "data/Alzheimer_s Dataset/val/VeryMildDemented/26 (51).jpg"
# path_im = "data/Alzheimer_s Dataset/train/MildDemented/mildDem711.jpg"
# path_im = "data/Alzheimer_s Dataset/train/ModerateDemented/moderateDem48.jpg"
# path_im = "data/Alzheimer_s Dataset/train/NonDemented/nonDem412.jpg"
# path_im = "data/Alzheimer_s Dataset/train/VeryMildDemented/verymildDem943.jpg"
path_im = "img/cat.jpg"

im = Image.open(path_im)

im = im.convert("RGB")
x = transform(im)
x.size()

logits, att_mat = model(x.unsqueeze(0))

att_mat = torch.stack(att_mat).squeeze(1)

# Average the attention weights across all heads.
att_mat = torch.mean(att_mat, dim=1)

# To account for residual connections, we add an identity matrix to the
# attention matrix and re-normalize the weights.
residual_att = torch.eye(att_mat.size(1))
aug_att_mat = att_mat + residual_att
aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

# Recursively multiply the weight matrices
joint_attentions = torch.zeros(aug_att_mat.size())
joint_attentions[0] = aug_att_mat[0]

for n in range(1, aug_att_mat.size(0)):
    joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

# Attention from the output token to the input space.
v = joint_attentions[-1]
grid_size = int(np.sqrt(aug_att_mat.size(-1)))
mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
mask = cv2.resize(mask / mask.max(), im.size)[..., np.newaxis]
result = (mask * im).astype("uint8")


fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))

ax1.set_title('Original')
ax2.set_title('Attention Map')
i = Image.fromarray(mask, 'RGB')
plt.imsave('img/mask.jpg', i)
plt.imsave('img/res.jpg', result)
plt.imsave('img/im.jpg', im)
_ = ax1.imshow(im)
_ = ax2.imshow(result)

probs = torch.nn.Softmax(dim=-1)(logits)
print(probs)
top5 = torch.argsort(probs, dim=-1, descending=True)
print("Prediction Label and Attention Map!\n")
print(path_im)
for idx in top5[0, :5]:
    print(f'{probs[0, idx.item()]:.5f} : {mri_labels[idx.item()]}')


