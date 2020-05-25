"""
Author: Shadi Zabad
Date: May 2018

This script implements a function that takes a matrix of images
and generates a montage using matplotlib.

"""

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def generate_montage(file_name,
                     img_matrix,
                     v_div_size=0,
                     h_div_size=0,
                     title=None,
                     x_tick_labels=None,
                     y_tick_labels=None,
                     x_axis_label=None,
                     y_axis_label=None,
                     axis_off=False,
                     fig_size=(12, 12)):
    """
    Note: v_div_size and h_div_size can be either integers or lists.
    If you provide a list, then the list size has to equal the number
    of images in the rows or columns of the matrix, respectively.
    """

    if len(img_matrix.shape) != 5:
        raise Exception("This function expects a 5 dimensional tensor! Given: "
                       + str(img_matrix.shape))

    if img_matrix.shape[-1] != 3:
        raise Exception("There should be 3 color channels!")

    num_imgs_v = img_matrix.shape[0]  # Number of images along the vertical axis
    num_imgs_h = img_matrix.shape[1]  # Number of images along the horizontal axis

    img_rows, img_cols, chnls = img_matrix.shape[-3:]

    try:
        iter(v_div_size)
        v_div_size = np.array(v_div_size)
    except TypeError:
        v_div_size = np.array([0] + list(np.repeat(v_div_size,
                                                   num_imgs_v - 1)))

    try:
        iter(h_div_size)
        h_div_size = np.array(h_div_size)
    except TypeError:
        h_div_size = np.array([0] + list(np.repeat(h_div_size,
                                                   num_imgs_h - 1)))


    fig_rows = img_rows * num_imgs_v + np.sum(v_div_size)
    fig_cols = img_cols * num_imgs_h + np.sum(h_div_size)

    c_fig = np.zeros((fig_rows, fig_cols, 3))

    for h in range (num_imgs_h):

        dh = h * img_cols + np.sum(h_div_size[:h + 1])

        for v in range(num_imgs_v):
            dv = v * img_rows + np.sum(v_div_size[:v + 1])

            c_fig[dv: dv + img_rows,
                  dh: dh + img_cols] = img_matrix[v, h]

            c_fig[dv - v_div_size[v]: dv, :] = np.ones((v_div_size[v],
                                                        fig_cols,
                                                        3))

        c_fig[:, dh - h_div_size[h]: dh] = np.ones((fig_rows,
                                                    h_div_size[h],
                                                    3))

    plt.figure(figsize=fig_size)

    # Plot the combined matrix
    plt.imshow(c_fig)

    # Add tick and axis labels, if provided:
    if x_tick_labels is not None:
        x_ticks = np.arange(num_imgs_h) * img_cols + int(.5 * img_cols)
        x_ticks += np.cumsum(h_div_size)

        plt.xticks(x_ticks, x_tick_labels, rotation=90)
    else:
        plt.xticks([], [])

    if y_tick_labels is not None:
        y_ticks = np.arange(num_imgs_v) * img_rows + int(.5 * img_rows)
        y_ticks += np.cumsum(v_div_size)

        plt.yticks(y_ticks, y_tick_labels)
    else:
        plt.yticks([], [])

    if x_axis_label is not None:
        plt.xlabel(x_axis_label)

    if y_axis_label is not None:
        plt.ylabel(y_axis_label)

    if title is not None:
        plt.title(title)

    if axis_off:
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

