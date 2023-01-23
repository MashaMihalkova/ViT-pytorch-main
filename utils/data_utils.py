import logging

import numpy as np
import torch
import os

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from utils.create_ADNI import build_loaders, build_datasets
from setting import settings
from utils.create_DataFrame import load_data_table_both, load_data_table_15T, load_data_table_3T
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    # for cifar dataset
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # custom for alzheimer dataset
    data_dir = 'data/Alzheimer_s Dataset'
    data_transforms = {
        'train': transforms.Compose([
            # transforms.CenterCrop(200),
            transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
            # transforms.Resize(200),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    if args.dataset == "cifar10":
        trainset = datasets.CIFAR10(root="./data",
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root="./data",
                                   train=False,
                                   download=True,
                                   transform=transform_test) if args.local_rank in [-1, 0] else None

    elif args.dataset == 'cifar100':
        trainset = datasets.CIFAR100(root="./data",
                                     train=True,
                                     download=True,
                                     transform=transform_train)
        testset = datasets.CIFAR100(root="./data",
                                    train=False,
                                    download=True,
                                    transform=transform_test) if args.local_rank in [-1, 0] else None
    elif args.dataset == 'custom_alzheimer':
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                                  data_transforms[x])
                          for x in ['train', 'val']}
        dataloaders = {
            'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=args.train_batch_size,
                                                 shuffle=True, num_workers=1),
            'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=args.train_batch_size,
                                               shuffle=True, num_workers=1)
        }

        # dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        # class_names = image_datasets['train'].classes
        train_loader = dataloaders['train']
        test_loader = dataloaders['val']
    elif args.dataset == 'ADNI_dataset':
        # df = load_data_table_both()
        df = load_data_table_15T()

        r = int(settings['r'])
        test_subjects_per_class = 30
        val_subjects_per_class = 18

        subjects_AD = df[df['Group'] == 'AD']['Subject'].unique()
        subjects_CN = df[df['Group'] == 'CN']['Subject'].unique()
        subjects_CN = [p for p in subjects_CN if p not in subjects_AD]

        patients_AD_train, patients_AD_test = train_test_split(subjects_AD, test_size=test_subjects_per_class,
                                                               random_state=r)
        patients_AD_train, patients_AD_val = train_test_split(patients_AD_train, test_size=val_subjects_per_class,
                                                              random_state=r)
        patients_CN_train, patients_CN_test = train_test_split(subjects_CN, test_size=test_subjects_per_class,
                                                               random_state=r)
        patients_CN_train, patients_CN_val = train_test_split(patients_CN_train, test_size=val_subjects_per_class,
                                                              random_state=r)

        patients_train = np.concatenate([patients_AD_train, patients_CN_train])
        patients_test = np.concatenate([patients_AD_test, patients_CN_test])
        patients_val = np.concatenate([patients_AD_val, patients_CN_val])

        train_dataset, val_dataset, test_dataset = build_datasets(df, patients_train, patients_val, patients_test,
                                                                  normalize=True)
        # build_datasets()
        train_loader, val_loader, test_loader = build_loaders(train_dataset, val_dataset, test_dataset)
        return train_loader, val_loader, test_loader

    if args.local_rank == 0:
        torch.distributed.barrier()

    # train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    # test_sampler = SequentialSampler(testset)
    # train_loader = DataLoader(trainset,
    #                           # sampler=train_sampler,
    #                           batch_size=args.train_batch_size,
    #                           num_workers=1,
    #                           pin_memory=True)
    # test_loader = DataLoader(testset,
    #                          # sampler=test_sampler,
    #                          batch_size=args.eval_batch_size,
    #                          num_workers=1,
    #                          pin_memory=True) if testset is not None else None

    return train_loader, test_loader
