import numpy as np
from matplotlib import pyplot as plt

import utils
from tqdm import tqdm_notebook

import torch
from torch.utils.data import Dataset, DataLoader

import nibabel as nib
import skimage.transform as skTrans


def load_nifti(file_path, mask=None, z_factor=None, remove_nan=True):
    """Load a 3D array from a NIFTI file."""
#     print(file_path)
    img = nib.load(file_path)
    struct_arr = np.array(img.get_data())
    # mid = struct_arr.shape[0] // 2
    # arr = struct_arr[mid - 3:mid + 3, :, :]
    # struct_arr = skTrans.resize(arr, (6, 229, 193), order=1, preserve_range=True)
    struct_arr = skTrans.resize(struct_arr, (193, 229, 193), order=1, preserve_range=True)
    # if remove_nan:
    #     struct_arr = np.nan_to_num(struct_arr)
    if mask is not None:
#         struct_arr *= mask
        struct_arr = struct_arr*mask
    # if z_factor is not None:
    #     struct_arr = np.around(zoom(struct_arr, z_factor), 0)

    return struct_arr

class ADNIDataset(Dataset):
    def __init__(self, filenames, labels, mask=None, transform=None):
        self.filenames = filenames
        self.labels = torch.LongTensor(labels)
        self.mask = mask
        self.transform = transform

        # Required by torchsample.
        self.num_inputs = 1
        self.num_targets = 1

        # Default values. Should be set via fit_normalization.
        self.mean = 0
        self.std = 1

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        """Return the image as a numpy array and the label."""
        label = self.labels[idx]

        nii_data = load_nifti(self.filenames[idx], mask=self.mask)
        struct_arr = np.ndarray(nii_data.shape)
        for slice_Number in range(nii_data.shape[0]):
            struct_arr[slice_Number,:,:] = nii_data[slice_Number,:, :]
            # plt.imsave(f'data/mni_struct/sl_n{slice_Number}.png', nii_data[slice_Number,:, :])

        struct_arr = (struct_arr - self.mean) / (self.std + 1e-10)  # prevent 0 division by adding small factor
        # struct_arr = struct_arr[None]  # add (empty) channel dimension
        struct_arr = torch.FloatTensor(struct_arr)

        if self.transform is not None:
            struct_arr = self.transform(struct_arr)

        return struct_arr, label

    def image_shape(self):
        """The shape of the MRI images."""
        return load_nifti(self.filenames[0], mask=self.mask).shape

    def fit_normalization(self, num_sample=None, show_progress=False):
        """
        Calculate the voxel-wise mean and std across the dataset for normalization.

        Args:
            num_sample (int or None): If None (default), calculate the values across the complete dataset,
                                      otherwise sample a number of images.
            show_progress (bool): Show a progress bar during the calculation."
        """

        if num_sample is None:
            num_sample = len(self)

        image_shape = self.image_shape()
        all_struct_arr = np.zeros((num_sample, image_shape[0], image_shape[1], image_shape[2]))

        sampled_filenames = np.random.choice(self.filenames, num_sample, replace=False)
        if show_progress:
            sampled_filenames = tqdm_notebook(sampled_filenames)

        for i, filename in enumerate(sampled_filenames):
            struct_arr = load_nifti(filename, mask=self.mask)
            all_struct_arr[i] = struct_arr

        self.mean = all_struct_arr.mean(0)
        self.std = all_struct_arr.std(0)

    def get_raw_image(self, idx):
        """Return the raw image at index idx (i.e. not normalized, no color channel, no transform."""
        return utils.load_nifti(self.filenames[idx], mask=self.mask)


def build_datasets(df, patients_train, patients_val, patients_test, print_stats=False, normalize=False):
    """
    Build PyTorch datasets based on a data table and a patient-wise train-test split.

    Args:
        df (pandas dataframe): The data table from ADNI.
        patients_train (iterable of strings): The patients to include in the train set.
        patients_val (iterable of strings): The patients to include in the val set.
        print_stats (boolean): Whether to print some statistics about the datasets.
        normalize (boolean): Whether to caluclate mean and std across the dataset for later normalization.

    Returns:
        The train and val dataset.
    """
    # Compile train and val dfs based on patients.
    df_train = df[df.apply(lambda row: row['Subject'] in patients_train, axis=1)]
    df_val = df[df.apply(lambda row: row['Subject'] in patients_val, axis=1)]
    df_test = df[df.apply(lambda row: row['Subject'] in patients_test, axis=1)]

    mask = None

    # Extract filenames and labels from dfs.
    train_filenames = np.array(df_train['filepath'])
    val_filenames = np.array(df_val['filepath'])
    test_filenames = np.array(df_test['filepath'])
    train_labels = np.array(df_train['Group'] == 'AD', dtype=int)  # [:, None]
    val_labels = np.array(df_val['Group'] == 'AD', dtype=int)  # [:, None]
    test_labels = np.array(df_test['Group'] == 'AD', dtype=int)  # [:, None]

    train_dataset = ADNIDataset(train_filenames, train_labels, mask=mask)
    val_dataset = ADNIDataset(val_filenames, val_labels, mask=mask)
    test_dataset = ADNIDataset(test_filenames, test_labels, mask=mask)

    if normalize:
        print('Calculating mean and std for normalization:')
        train_dataset.fit_normalization(200, show_progress=True)
        val_dataset.mean, val_dataset.std = train_dataset.mean, train_dataset.std
        test_dataset.mean, test_dataset.std = train_dataset.mean, train_dataset.std
    else:
        print('Dataset is not normalized, this could dramatically decrease performance')

    return train_dataset, val_dataset, test_dataset


def build_loaders(train_dataset, val_dataset, test_dataset, bs):
    """Build PyTorch data loaders from the datasets."""

    # In contrast to Korolev et al. 2017, we do not enforce one sample per class in each batch.
    # TODO: Maybe change batch size to 3 or 4. Check how this affects memory and accuracy.
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=0,
                              pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=0,
                            pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=0,
                             pin_memory=torch.cuda.is_available())

    return train_loader, val_loader, test_loader
