"""
Author: Shadi Zabad
Date: April 2018

This script aims to serve as a module for loading, transforming, and augmenting DeepLoc images
(https://github.com/okraus/DeepLoc)
and prepare them for training with Keras and other deep learning packages.

"""

# -------------------------
# Import required libraries
# -------------------------

import os
import errno

import numpy as np
import h5py

from sklearn.neighbors import DistanceMetric
from skimage.util import crop
from skimage import exposure

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -------------------------
# Constants and global variables
# -------------------------

localization_terms = ['ACTIN', 'BUDNECK', 'BUDTIP', 'CELLPERIPHERY', 'CYTOPLASM',
                      'ENDOSOME', 'ER', 'GOLGI', 'MITOCHONDRIA', 'NUCLEARPERIPHERY',
                      'NUCLEI', 'NUCLEOLUS', 'PEROXISOME', 'SPINDLE', 'SPINDLEPOLE',
                      'VACUOLARMEMBRANE', 'VACUOLE', 'DEAD', 'GHOST']

wt_localization_terms = ['ACTIN', 'BUD', 'BUDNECK', 'BUDPERIPHERY', 'BUDSITE',
                         'CELLPERIPHERY', 'CYTOPLASM', 'CYTOPLASMICFOCI',
                         'EISOSOMES', 'ER', 'ENDOSOME', 'GOLGI',
                         'LIPID_PARTICLES', 'MITOCHONDRIA', 'NONE',
                         'NUCLEARPERIPHERY', 'NUCLEOLUS', 'NUCLEI',
                         'PEROXISOME', 'PUNCTATENUCLEAR', 'VACUOLE',
                         'VACUOLEPERIPHERY']

rfp_localization_terms = ['ER', 'BUD', 'BUDNECK', 'CELLPERIPHERY', 'CYTOSOL',
                          'MITOCHONDRIA', 'NUCLEARPERIPHERY', 'NUCLEI',
                          'PUNCTATE', 'VACUOLE', 'VACUOLARMEMBRANE']

# -------------------------
# Define auxiliary functions
# -------------------------

def transform_transfer_data(X, y, loc_terms):

    indices_to_remove = []
    y_new = np.zeros((y.shape[0], len(localization_terms)))

    for idx in range(len(y)):

        img_class = loc_terms[np.argmax(y[idx])]

        if img_class in localization_terms:
            y_new[idx, localization_terms.index(img_class)] = 1
        else:
            indices_to_remove.append(idx)

    X_new = np.delete(X, indices_to_remove, axis=0)
    y_new = np.delete(y_new, indices_to_remove, axis=0)

    return X_new, y_new


def read_deeploc_data(file_path,
                      img_dims=(64, 64, 2),
                      remove_channels=None,
                      contrast_enhancer='rescale',
                      perc_range=(0.1, 99.9)):

    with h5py.File(file_path, 'r') as f_h5data:
        X, y = np.array(f_h5data['data1']), np.array(f_h5data['Index1'])

    X = X.reshape((X.shape[0],) + img_dims, order='F')
    X = np.transpose(X, (0, 2, 1 , 3))
    X = X[:, :, :, ::-1]

    if remove_channels is not None:
        X = np.delete(X, remove_channels, axis=-1)

    if contrast_enhancer == 'rescale':
        for i in range(len(X)):
            low_, high_ = np.percentile(X[i], perc_range)
            X[i] = exposure.rescale_intensity(X[i], in_range=(low_, high_))
    elif contrast_enhancer == 'histogram':
        for i in range(len(X)):
            X[i] = exposure.equalize_hist(X[i])

    return X, y


def random_crop(img_data, crop_size=4):
    """
    Randomly crops images by <crop_size>
    from both horizontal and vertical axes.
    """

    # If it's a batch rather than a single image,
    # then recursively call this function to process
    # the batch.
    if len(img_data.shape) > 3:

        n_img_data = np.zeros((
            img_data.shape[0],
            img_data.shape[1] - crop_size,
            img_data.shape[2] - crop_size,
            img_data.shape[3]
        ), dtype=img_data.dtype)

        for i in range(len(img_data)):
            n_img_data[i, :, :, :] = random_crop(img_data[i])

        return n_img_data

    row_before, col_before = np.random.randint(0, crop_size, 2)
    row_after, col_after = crop_size - row_before, crop_size - col_before

    crop_dims = (
        (row_before, row_after),
        (col_before, col_after),
        (0, 0)
    )

    return crop(img_data, crop_dims)


def random_transform(img_data):

    # If it's a batch rather than a single image,
    # then recursively call this function to process
    # the batch.
    if len(img_data.shape) > 3:
        for i in range(len(img_data)):
            img_data[i] = random_transform(img_data[i])

        return img_data


    # Flip vertically (up/down):
    if np.random.rand() < .5:
        img_data = np.flipud(img_data)

    # Flip horizontally (left/right):
    if np.random.rand() < .5:
        img_data = np.fliplr(img_data)

    # Rotate image by N*90 degrees, where N between [0, 3]
    img_data = np.rot90(img_data, k=np.random.randint(0, 4))

    return img_data


def match_data_subset_sizes(small_data_index, large_data_index, batch_size):
    """
    This is used in the training phase,
    primarily to match the size of the labelled set with the size
    of the unlabelled set (see fit_vae_models() function).
    """

    # Find the larger subset:
    if len(small_data_index) >= len(large_data_index):
        raise Exception('The first data index must be strictly smaller than the second data index')


    size_diff = len(large_data_index) - len(small_data_index)
    idx_to_add = []

    while size_diff - (len(idx_to_add) * batch_size) > 0:
        idx_to_add.append(np.random.choice(small_data_index, replace=False,
                                           size=batch_size))

    small_data_index = np.append(small_data_index,
                                 np.concatenate(idx_to_add))

    np.random.shuffle(small_data_index)

    return small_data_index, large_data_index


def add_blue_channel(img_data):
    # DeepLoc images come with green and red channels only, so we need to add a
    # blue channel with value of 0.0:
    return np.concatenate((img_data, np.zeros(img_data.shape[:-1] + (1,))), axis=-1)


def plot_deeploc_image(img_data, output_file_path):

    if img_data.shape[-1] == 2:
        img_data = add_blue_channel(img_data)

    # Plot the image:
    plt.imshow(img_data)
    plt.axis('off')

    plt.savefig(output_file_path)
    plt.close()


def plot_random_image_sample(X, y,
                             ext=".png",
                             images_per_class=2,
                             output_dir="./plots/sample/"):

    try:
        os.makedirs(output_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    y = np.argmax(y, axis=-1)

    for cls_idx in range(len(localization_terms)):
        cls_imgs = np.argwhere(y == cls_idx)
        cls_imgs = cls_imgs.T[0]

        if len(cls_imgs) < 1:
            continue

        for s_idx in range(images_per_class):
            img_idx = np.random.choice(cls_imgs)
            print localization_terms[cls_idx] + "_" +  str(s_idx), img_idx
            file_path = os.path.join(output_dir, localization_terms[cls_idx] +
                                     "_" +  str(s_idx) + ext)
            plot_deeploc_image(X[img_idx], file_path)


def find_closest_image(X, img):

    min_dist = None
    closest_img = None

    for im in X:
        dist = np.sum((im - img) ** 2)
        if min_dist is None or dist < min_dist:
            min_dist = dist
            closest_img = im

    return closest_img
