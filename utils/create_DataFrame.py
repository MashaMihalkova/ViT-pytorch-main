import pandas as pd
import os
from setting import settings
import glob

# Ritter/Haynes lab file system at BCCN Berlin.
ADNI_DIR = settings["ADNI_DIR"]

# Filepaths for 1.5 Tesla scans.
table_15T = os.path.join(ADNI_DIR, settings["1.5T_table"])
image_dir_15T = os.path.join(ADNI_DIR, settings["1.5T_image_dir"])

table_15T_compl = os.path.join(ADNI_DIR, settings["1.5T_table_compl"])
image_dir_15T_compl = os.path.join(ADNI_DIR, settings["1.5T_image_dir_compl"])

table_3T = os.path.join(ADNI_DIR, settings["3T_table"])
image_dir_3T = os.path.join(ADNI_DIR, settings["3T_image_dir"])
corrupt_images_15T = None  # ['067_S_0077/Screening']
corrupt_images_3T = None


def load_data_table(table, image_dir, corrupt_images=None):
    """Read data table, find corresponding images, filter out corrupt,
    missing and MCI images, and return the samples as a pandas dataframe."""

    # Read table into dataframe.
    print('Loading dataframe for', table)
    df = pd.read_csv(table)
    print('Found', len(df), 'images in table')

    # Add column with filepaths to images.
    df['filepath'] = df.apply(lambda row: get_image_filepath(row, image_dir), axis=1)

    # Filter out corrupt images (i.e. images where the preprocessing failed).
    len_before = len(df)
    if corrupt_images is not None:
        df = df[df.apply(lambda row: '{}/{}'.format(row['Subject'], row['Visit']) not in corrupt_images, axis=1)]
    print('Filtered out', len_before - len(df), 'of', len_before, 'images because of failed preprocessing')

    # Filter out images where files are missing.
    len_before = len(df)
    # print(df[~np.array(map(os.path.exists, df['filepath']))]['filepath'].values)
    df = df.loc[map(os.path.exists, df['filepath'])]
    print('Filtered out', len_before - len(df), 'of', len_before, 'images because of missing files')

    # Filter out images with MCI.
    len_before = len(df)
    df = df[df['Group'] != 'MCI']
    print('Filtered out', len_before - len(df), 'of', len_before, 'images that were MCI')

    print('Final dataframe contains', len(df), 'images from', len(df['Subject'].unique()), 'patients')
    print()

    return df


def load_data_table_3T():
    """Load the data table for all 3 Tesla images."""
    return load_data_table(table_3T, image_dir_3T, corrupt_images_3T)


def load_data_table_15T():
    """Load the data table for all 1.5 Tesla images."""
    return load_data_table(table_15T, image_dir_15T, corrupt_images_15T)


def load_data_table_both():
    """Load the data tables for all 1.5 Tesla and 3 Tesla images and combine them."""
    df_15T = load_data_table(table_15T, image_dir_15T, corrupt_images_15T)
    df_15T_compl = load_data_table(table_15T_compl, image_dir_15T_compl, corrupt_images_15T)
    #     print(image_dir_3T)
    #     df_3T = load_data_table(table_3T, image_dir_3T, corrupt_images_3T)
    df = pd.concat([df_15T, df_15T_compl])
    return df


def get_image_filepath(df_row, mode, root_dir=''):
    """Return the filepath of the image that is described in the row of the data table."""
    descr = df_row["Description"].replace(';', '_')
    descr = descr.replace(' ', '_')
    date = pd.to_datetime(df_row["Acq Date"]).strftime('%Y/%m/%d').replace('/', '-')
    filename = glob.glob(f'{mode}/ADNI/{df_row["Subject"]}/{descr}/{date}*/{df_row["Image Data ID"]}/*')
    if filename:
        return filename[0]
    return ''
