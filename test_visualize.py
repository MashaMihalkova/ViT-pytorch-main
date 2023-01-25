import typing
import io
import os

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from urllib.request import urlretrieve
import argparse
from PIL import Image
from torchvision import transforms

from models.modeling import VisionTransformer, CONFIGS
from torchvision import models
from utils.data_utils import get_loader
# import torchvision.transforms as T


parser = argparse.ArgumentParser()
# Required parameters
parser.add_argument("--name",  # required=True,
                    help="Name of this run. Used for monitoring.", default='ADNI_dataset')  # custom_alzheimer
parser.add_argument("--dataset", choices=["cifar10", "cifar100", "custom_alzheimer", "ADNI_dataset"],
                    default="ADNI_dataset",
                    help="Which downstream task.")
parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                             "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                    default="ViT-B_16",
                    help="Which variant to use.")
parser.add_argument("--pretrained_dir", type=str, default='output/custom_alzheimer_checkpoint.pkl',
                    # "checkpoint/ViT-B_16.npz",
                    help="Where to search for pretrained ViT models.")
parser.add_argument("--output_dir", default="output", type=str,
                    help="The output directory where checkpoints will be written.")

parser.add_argument("--img_size", default=224, type=int,
                    help="Resolution size")
parser.add_argument("--train_batch_size", default=1, type=int,
                    help="Total batch size for training.")
parser.add_argument("--eval_batch_size", default=1, type=int,
                    help="Total batch size for eval.")
parser.add_argument("--eval_every", default=1, type=int,
                    help="Run prediction on validation set every so many steps."
                         "Will always run one evaluation at the end of training.")

parser.add_argument("--learning_rate", default=3e-2, type=float,
                    help="The initial learning rate for SGD.")
parser.add_argument("--weight_decay", default=0, type=float,
                    help="Weight deay if we apply some.")
parser.add_argument("--num_steps", default=30, type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                    help="How to decay the learning rate.")
parser.add_argument("--warmup_steps", default=500, type=int,
                    help="Step of training to perform learning rate warmup for.")
parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm.")

parser.add_argument("--local_rank", type=int, default=-1,
                    help="local_rank for distributed training on gpus")
parser.add_argument('--seed', type=int, default=45,
                    help="random seed for initialization")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument('--fp16', action='store_true',
                    help="Whether to use 16-bit float precision instead of 32-bit")
parser.add_argument('--fp16_opt_level', type=str, default='O2',
                    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                         "See details at https://nvidia.github.io/apex/amp.html")
parser.add_argument('--loss_scale', type=float, default=0,
                    help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                         "0 (default value): dynamic loss scaling.\n"
                         "Positive power of 2: static loss scaling value.\n")
args = parser.parse_args()


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
model = VisionTransformer(config, num_classes=2, kol_sl=5, bs=1, zero_head=False, img_size=224, vis=True)
# model = models.resnet101(pretrained=True)
# model.fc = torch.nn.Linear(model.fc.in_features, 4)
# w2 = 'checkpoint/ViT-B_16.npz'
w2 = 'output/ADNI_dataset_checkpoint.pkl'  # ADNI_dataset_checkpoint.pkl
checkpoint = torch.load(w2)
model.load_state_dict(checkpoint)

# model.load_from(np.load(w2))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
train_loader, val_loader, test_loader = get_loader(args)
for step, batch in enumerate(val_loader):
    batch = tuple(t.to(device) for t in batch)
    x, y = batch
    with torch.no_grad():
        # logits = model(x)[0]
        logits, att_mat = model(x, kol_sl=5, bs=1)   # if one img x.unsqueeze(0)
        for sl in range(att_mat.shape[0]):
            att_mat_sq = torch.mean(att_mat[sl].squeeze(1), dim=1)

            # To account for residual connections, we add an identity matrix to the
            # attention matrix and re-normalize the weights.
            residual_att = torch.eye(att_mat_sq.size(1))

            aug_att_mat = att_mat_sq + residual_att.to(device)
            aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

            # Recursively multiply the weight matrices
            joint_attentions = torch.zeros(aug_att_mat.size())
            joint_attentions[0] = aug_att_mat[0]
            # joint_attentions.to(device)
            for n in range(1, aug_att_mat.size(0)):
                joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1].to(device))

            # Attention from the output token to the input space.
            v = joint_attentions[-1]
            grid_size = int(np.sqrt(aug_att_mat.size(-1)))
            # mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
            mask = v[0, 0:].reshape(grid_size, grid_size).detach().numpy()

            tensor_ = x[0][sl+60]
            # transform = transforms.ToPILImage()
            # img = transform(tensor_)
            # img = img.convert("RGB")


            # mask = cv2.resize(mask / mask.max(), img.size)[..., np.newaxis]
            img = np.array(tensor_.cpu())
            mask = cv2.resize(mask / mask.max(), (img.shape[1],img.shape[0]))[..., np.newaxis]

            result = (mask * img[:,:,None])  #.astype("uint8")


            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))

            ax1.set_title('Original')
            ax2.set_title('Attention Map')
            i = Image.fromarray(mask, 'RGB')
            plt.imsave('img/mask.jpg', i)
            plt.imsave('img/res.jpg', result.reshape(result.shape[:-1]))
            plt.imsave('img/im.jpg', img)
            _ = ax1.imshow(img)
            _ = ax2.imshow(result)
            plt.show()
        print(1)


##########################################
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


