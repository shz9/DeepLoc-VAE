#!/valr/szabad/tensorflow/bin/python -u

"""
Author: Shadi Zabad
Date: April 2018

This file contains some auxiliary functions to generate the plots
for the project report.

"""


# ------------------------------------------
# Importing libraries and modules
# ------------------------------------------

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
import itertools

from sklearn import manifold

# ------------------------------------------
# Plotting functions
# ------------------------------------------

def generate_avg_precision_plot(class_prec_scores,
                                mean_prec_score,
                                loc_terms,
                                sample_size,
                                set_category):

    excluded_cl = ['SPINDLE', 'DEAD', 'GHOST']
    loc_terms = [lt for lt in loc_terms if lt not in excluded_cl]

    ind = np.arange(len(loc_terms))
    width = .4

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.bar(ind,
           [class_prec_scores[cl] for cl in loc_terms if cl not in excluded_cl],
           width,
           color='#91d4d8')

    ax.set_ylabel('Average Precision')
    ax.set_xlabel('Category')
    ax.set_title('Performance of VAE M2 Model on the ' + set_category +
                 ' Set (' + str(sample_size) + ' Samples)')
    ax.set_xticks(ind)
    ax.set_xticklabels(loc_terms)

    ax.xaxis.set_tick_params(rotation=90)

    plt.axhline(y=mean_prec_score, color='#ba4a4f', linestyle='--')

    plt.tight_layout()
    plt.savefig("./plots/" + set_category + "_report_prec_fig.pdf")
    plt.close()


def generate_tsne_plot(layer_name, x_data, true_labels, perplexity=50):

    tsne = manifold.TSNE(n_components=2,
                         init='pca',
                         perplexity=perplexity)

    x_embed = tsne.fit_transform(x_data)

    palette = np.array(sns.color_palette("hls", len(np.unique(true_labels))))

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(x_embed[:, 0],
               x_embed[:, 1],
               lw=0, s=40,
               c=palette[true_labels])

    ax.axis('off')
    ax.axis('tight')

    plt.tight_layout()
    plt.savefig("./plots/tsne/" + layer_name + str(perplexity) + "_tsne_fig.png")
    plt.close()


def plot_confusion_matrix(cm,
                          loc_terms,
                          sample_size,
                          set_category,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    Modified from:
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """

    excluded_cl = ['SPINDLE', 'DEAD', 'GHOST']

    excluded_ind = [loc_terms.index(lt) for lt in excluded_cl]

    cm = np.delete(cm, excluded_ind, axis=0)
    cm = np.delete(cm, excluded_ind, axis=1)

    loc_terms = [lt for lt in loc_terms if lt not in excluded_cl]

    title = "Confusion Matrix of the " + set_category + \
            " Set (" + str(sample_size) + " Samples)"

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = "Normalized " + title
    else:
        title = "Raw " + title

    plt.figure(figsize=(9, 9))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(loc_terms))
    plt.xticks(tick_marks, loc_terms, rotation=90)
    plt.yticks(tick_marks, loc_terms)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.tight_layout()

    plt.savefig("./plots/" + title.lower().replace(' ', '_') + ".pdf")

    plt.close()


