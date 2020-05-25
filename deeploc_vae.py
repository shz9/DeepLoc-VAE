#!/valr/szabad/tensorflow/bin/python -u

"""
Author: Shadi Zabad
Date: April 2018

This is an implementation of the M2 Variational Autoencoder
model outlined in Kingma et al. 2014.

I'd like to acknowledge Brian Keng's excellent tutorial
on the subject, which can be found here:

http://bjlkeng.github.io/posts/semi-supervised-learning-with-variational-autoencoders/

"""

# ------------------------------------------
# Importing libraries and modules
# ------------------------------------------

from keras.layers import (Input, Conv2D, Conv2DTranspose, MaxPooling2D,
                          BatchNormalization, Flatten, Reshape,
                          Dense, Activation, Dropout, Lambda)
from keras.layers.merge import concatenate
from keras import backend as K
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.models import Model, load_model, save_model
from keras import metrics

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             classification_report, confusion_matrix,
                             average_precision_score)

import time
import datetime
import argparse
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_generator import *
from report_plots import *
from image_montage import generate_montage

# ------------------------------------------
# Read and parse command line arguments
# ------------------------------------------

argp = argparse.ArgumentParser()

argp.add_argument("-b", "--batchsize", dest='batchsize', type=int,
                  help="Batch size for model training",
                  default=200)

argp.add_argument("-e", "--epochs", dest='epochs', type=int,
                  help="Epochs for model training",
                  default=10)

argp.add_argument("-r", "--learningrate", dest='learningrate', type=float,
                  help="Learning rate for model training",
                  default=1e-3)

argp.add_argument("-d", "--labelled", dest='labelled', type=float,
                  help="Fraction of labelled training data [0.0, 1.0]",
                  default=1./6)

argp.add_argument("-t", "--trained", dest='trained', type=str,
                  help="Pass the prefix to a trained VAE model to use",
                  default=None)

argp.add_argument("-a", "--augment", dest='augment', type=int,
                  help="Number of times to apply random transformations to the image batch",
                  default=0)

argp.add_argument("-l", "--latent", dest='latentdim', type=int,
                  help="The dimension of the latent space",
                  default=128)

cmd_args = argp.parse_args()

# ------------------------------------------
# Defining constants and global variables
# ------------------------------------------

# Input-related constants:

TRAIN_DATA_PATH = "./datasets/Chong_train_set.hdf5"
VALID_DATA_PATH = "./datasets/Chong_valid_set.hdf5"
TEST_DATA_PATH = "./datasets/Chong_test_set.hdf5"
WT_DATA_PATH = "./datasets/wt2017_test_set.hdf5"
RFP_DATA_PATH = "./datasets/Schuldiner_test_set.hdf5"

IMG_ROWS, IMG_COLS, IMG_CHNLS = 60, 60, 2
NUM_CLASSES = 19

# Model and training hyperparameters:

BATCH_SIZE = cmd_args.batchsize
LEARNING_RATE = cmd_args.learningrate
EPOCHS = cmd_args.epochs

LATENT_DIM = cmd_args.latentdim
EPSILON = 1.

TRAINED_MODEL = cmd_args.trained
AUGMENT_TIMES = cmd_args.augment

# For semi-supervised learning, we'll start with about 1/6th of the data

FRAC_LABELLED = cmd_args.labelled

# Create a prefix for output files:

f_prefix = "_".join(["deeploc",
                     datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                     str(BATCH_SIZE),
                     str(EPOCHS),
                     str(AUGMENT_TIMES),
                     str(FRAC_LABELLED).replace('.', 'p')])

print '------------------------------------------'
print 'Files generated from this run are using the following prefix:'
print f_prefix
print '------------------------------------------'

# ------------------------------------------
# Define input (including input layers...)
# ------------------------------------------

# Load the training dataset:
x_train, y_train = read_deeploc_data(TRAIN_DATA_PATH)
crop_x_train = random_crop(x_train)

# Load the testing dataset:
x_test, y_test = read_deeploc_data(TEST_DATA_PATH)
crop_x_test = random_crop(x_test)

# Load the validation dataset:
x_valid, y_valid = read_deeploc_data(VALID_DATA_PATH)


# plot_random_image_sample(x_test, y_test,

#                         output_dir="./plots/sample_test/")

"""

# Load the SAWT-RFP test set:
x_rfp, y_rfp = read_deeploc_data(RFP_DATA_PATH)

# Transform and retain only classes that are in common:
x_rfp, y_rfp = transform_transfer_data(x_rfp, y_rfp, rfp_localization_terms)

# Load the wildtype test set:
x_wt, y_wt = read_deeploc_data(WT_DATA_PATH,
                               img_dims=(64, 64, 3),
                               remove_channels=1)

# Transform and retain only classes that are in common:
x_wt, y_wt = transform_transfer_data(x_wt, y_wt, wt_localization_terms)

#plot_random_image_sample(x_wt, y_wt,
#                         output_dir="./plots/sample_wt/")

"""

print '-------------------------------------'
print '> x_train shape:', x_train.shape, 'y_train shape:', y_train.shape
print '> x_test shape:', x_test.shape, 'y_test shape:', y_test.shape
print '-------------------------------------'

# Split the training data into labelled vs. unlabelled:

train_split = StratifiedShuffleSplit(n_splits=2,
                                     test_size=FRAC_LABELLED,
                                     random_state=0)
_, (ul_idx, l_idx) = train_split.split(x_train, y_train)

# Create labelled and unlabelled subsets of the data:
l_x_train, l_y_train = x_train[l_idx], y_train[l_idx]
ul_x_train = x_train[ul_idx]

# Set the alpha parameter here (need l_x_train dims for this!):
ALPHA = .1 * len(l_x_train)

print '-------------------------------------'
print '> Labelled x shape:', l_x_train.shape, 'Labelled y shape:', l_y_train.shape
print '> Unlabelled x shape:', ul_x_train.shape
print '-------------------------------------'

# Define input layers:

x_input = Input(batch_shape=(BATCH_SIZE, IMG_ROWS, IMG_COLS, IMG_CHNLS))
y_input = Input(batch_shape=(BATCH_SIZE, NUM_CLASSES))

# ------------------------------------------
# Auxiliary functions
# ------------------------------------------

def join_network_layers(layer_list):
    if len(layer_list) < 1:
        return None
    else:
        subnet = layer_list[0]
        for layr in layer_list[1:]:
            subnet = layr(subnet)
        return subnet


# ------------------------------------------
# Encoder network
# ------------------------------------------

def create_encoder_network(input_layer=None, conv_filters=64,
                           activation='relu', kernel_size=3, padding='same'):

    en_layers = [
        # 1st Conv layer (with BatchNorm and RELU activation):
        Conv2D(name='e_conv_1', filters=conv_filters, kernel_size=kernel_size,
               padding=padding),
        BatchNormalization(name='e_conv_1_BN'),
        Activation(activation, name='e_conv_1_ACT'),
        # 2nd Conv layer (with BatchNorm and RELU activation):    
        Conv2D(name='e_conv_2', filters=conv_filters, kernel_size=kernel_size,
               padding=padding),
        BatchNormalization(name='e_conv_2_BN'),
        Activation(activation, name='e_conv_2_ACT'),
        # 3rd Conv layer (with BatchNorm and RELU activation):    
        Conv2D(name='e_conv_3', filters=conv_filters, kernel_size=kernel_size,
               strides=2, padding=padding),
        BatchNormalization(name='e_conv_3_BN'),
        Activation(activation, name='e_conv_3_ACT'),
        # Flatten the output of the 3rd Conv layer:
        Flatten(name='e_flatten'),
        # Create a fully connected layer (with BatchNorm, RELU activation, and
        # dropout):
        Dense(512, name='e_FC'),
        BatchNormalization(name='e_FC_BN'),
        Activation(activation, name='e_FC_ACT'),
        Dropout(0.5, name='e_dropout')
    ]

    if input_layer is None:
        return en_layers
    else:
        return join_network_layers([input_layer] + en_layers)


# Call the create_encoder_network function
encoder_layers = create_encoder_network()
encoder_network = join_network_layers([x_input] + encoder_layers)

# Create a Dense layer whose output is a vector of the means of the latent variables (Z):
z_means = Dense(LATENT_DIM)(encoder_network)

# Create a Dense layer whose output is a vector of the log variances of the latent variables (Z):
z_log_vars = Dense(LATENT_DIM)(encoder_network)


# ------------------------------------------
# Classifier network
# ------------------------------------------

def create_classifier_network(input_layer=None, conv_filters_1=32, conv_filters_2=64,
                              activation='relu', f_activation='softmax', kernel_size=3, padding='same'):

    cls_layers = [
        # 1st Conv layer (with RELU activation): 
        Conv2D(name='cls_conv_1', filters=conv_filters_1,
               kernel_size=kernel_size, padding=padding),
        Activation(activation, name='cls_conv_1_ACT'),
        # 2nd Conv layer (with RELU activation): 
        Conv2D(name='cls_conv_2', filters=conv_filters_1,
               kernel_size=kernel_size),
        Activation(activation, name='cls_conv_2_ACT'),
        # 1st MaxPool layer:
        MaxPooling2D(pool_size=2, name='cls_maxpool_1'),
        Dropout(0.25, name='cls_dropout_1'),
        # 3rd Conv layer (with RELU activation): 
        Conv2D(name='cls_conv_3', filters=conv_filters_2,
               kernel_size=kernel_size, padding=padding),
        Activation(activation, name='cls_conv_3_ACT'),
        # 4th Conv layer (with RELU activation): 
        Conv2D(name='cls_conv_4', filters=conv_filters_2,
               kernel_size=kernel_size),
        Activation(activation, name='cls_conv_4_ACT'),
        # 2nd MaxPool layer: 
        MaxPooling2D(pool_size=2, name='cls_maxpool_2'),
        Dropout(0.25, name='cls_dropout_2'),
        Flatten(name='cls_flatten'),
        # 1st Fully Connected layer (with RELU activation):
        Dense(512, name='cls_FC_1'),
        Activation(activation, name='cls_FC_1_ACT'),
        Dropout(0.5, name='cls_dropout_3'),
        # 2nd Fully Connected layer (with softmax activation):
        Dense(NUM_CLASSES, name='cls_FC_2'),
        Activation(f_activation, name='cls_FC_2_ACT')
    ]

    if input_layer is None:
        return cls_layers
    else:
        return join_network_layers([input_layer] + cls_layers)


classifier_layers = create_classifier_network()
y_pred = join_network_layers([x_input] + classifier_layers)

# ------------------------------------------
# Decoder network
# ------------------------------------------

def sample_latent_variables(latent_vecs):

    z_means, z_log_vars = latent_vecs

    rand_sample = K.random_normal(shape=(BATCH_SIZE, LATENT_DIM), mean=0., stddev=EPSILON)

    return z_means + (rand_sample * K.exp(z_log_vars))


def create_decoder_network(input_layer=None, conv_filters=64, activation='relu',
                           f_activation='sigmoid', kernel_size=3, padding='same'):

    de_layers = [
        # 1st Fully Connected layer (with BatchNorm and RELU activation):
        Dense(64 * 15 * 15, name='d_FC'),
        BatchNormalization(name='d_FC_BN'),
        Activation(activation, name='d_FC_ACT'),
        Dropout(0.5, name='d_dropout'),
        # Reshape output of FC for the Deconv layers:
        Reshape((15, 15, 64), name='d_reshape'),
        # 1st Deconv layer (with BatchNorm and RELU activation):
        Conv2DTranspose(name='d_conv_1', filters=conv_filters,
                        kernel_size=kernel_size, padding=padding),
        BatchNormalization(name='d_conv_1_BN'),
        Activation(activation, name='d_conv_1_ACT'),
        # 2nd Deconv layer (with BatchNorm and RELU activation):
        Conv2DTranspose(name='d_conv_2', filters=conv_filters,
                        kernel_size=kernel_size, strides=2, padding=padding),
        BatchNormalization(name='d_conv_2_BN'),
        Activation(activation, name='d_conv_2_ACT'),
        # 3rd Deconv layer (with BatchNorm and RELU activation):
        Conv2DTranspose(name='d_conv_3', filters=conv_filters,
                        kernel_size=kernel_size, strides=2, padding=padding),
        BatchNormalization(name='d_conv_3_BN'),
        Activation(activation, name='d_conv_3_ACT'),
        # Final Deconv layer (decoded image):
        Conv2DTranspose(name='d_conv_final', filters=IMG_CHNLS,
                        kernel_size=1, padding=padding,
                        activation=f_activation)
    ]

    if input_layer is None:
        return de_layers
    else:
        return join_network_layers([input_layer] + de_layers)


# We first sample from the latent distribution:
sampling_layer = Lambda(sample_latent_variables,
                        output_shape=(LATENT_DIM,),
                        name='sampling_layer')([z_means, z_log_vars])

# Retrieve decoder network layers (these are shared between labelled &
# unlabelled VAE, so we should create them only once!):

decoder_layers = create_decoder_network()

# When we have labelled examples in the training, we want to concatenate the true label
# with the sample from the latent distribution:

concat_y_z = concatenate([y_input, sampling_layer])
labelled_decoder_network = join_network_layers([concat_y_z] + decoder_layers)

# When we have unlabelled examples, we want to concatenate the *predicted* label
# with the sample from the latent distribution:

concat_yhat_z = concatenate([y_pred, sampling_layer])
unlabelled_decoder_network = join_network_layers([concat_yhat_z] + decoder_layers)

# ------------------------------------------
# Objective functions
# ------------------------------------------

def classification_loss(y, y_pred):
    return ALPHA * metrics.categorical_crossentropy(y, y_pred)


def labelled_vae_loss(x_input, x_decoded):

    # First we obtain the KL loss:
    kl_loss = K.mean(- 0.5 * K.sum(1. + z_log_vars - K.square(z_means) - K.exp(z_log_vars), axis=-1))

    # Then we obtain the crossentropy loss:
    xent_loss = IMG_ROWS * IMG_COLS * IMG_CHNLS * metrics.binary_crossentropy(K.flatten(x_input),
                                                                              K.flatten(x_decoded))
    xent_loss -= np.log(1. / NUM_CLASSES)

    return kl_loss + xent_loss


def unlabelled_vae_loss(x_input, x_decoded):

    pred_entropy = metrics.categorical_crossentropy(y_pred, y_pred)
    labelled_loss = labelled_vae_loss(x_input, x_decoded)

    return K.mean(K.sum(y_pred * labelled_loss, axis=-1)) + pred_entropy

# ------------------------------------------
# Model training
# ------------------------------------------

# Setup the Adam optimizer:

optimizer = Adam(lr=LEARNING_RATE)

# Compile the labelled VAE model:

l_vae = Model(inputs=[x_input, y_input], outputs=[labelled_decoder_network, y_pred])
l_vae.compile(optimizer=optimizer, loss=[labelled_vae_loss, classification_loss])
print l_vae.summary()

# Compile the unlabelled VAE model:

ul_vae = Model(inputs=x_input, outputs=unlabelled_decoder_network)
ul_vae.compile(optimizer=optimizer, loss=unlabelled_vae_loss)
print ul_vae.summary()

# -------------------------------
# Model training and fitting
# -------------------------------

def training_round(x_labeled_batch, y_labeled_batch, x_unlabeled_batch):

    # Labeled
    l_loss = l_vae.train_on_batch([x_labeled_batch, y_labeled_batch],
                                  [x_labeled_batch, y_labeled_batch])

    print "|| Labeled Loss || ->", l_loss

    # Unlabeled
    ul_loss = ul_vae.train_on_batch(x_unlabeled_batch, x_unlabeled_batch)
    print "||| Unlabeled Loss ||| ->", ul_loss

    print '----------'


def fit_vae_models(X_unlabeled, X_labeled, y_labeled, epochs, batch_size):

    start_time = time.time()

    labeled_index = np.arange(len(X_labeled))
    unlabeled_index = np.arange(len(X_unlabeled))

    # Match the size of the labeled and unlabeled indices
    if len(labeled_index) > len(unlabeled_index):
        unlabeled_index, labeled_index = match_data_subset_sizes(unlabeled_index,
                                                                 labeled_index,
                                                                 batch_size)
    elif len(unlabeled_index) > len(labeled_index):
        labeled_index, unlabeled_index = match_data_subset_sizes(labeled_index,
                                                                 unlabeled_index,
                                                                 batch_size)

    # Determine the number of batches:
    if len(unlabeled_index) < len(labeled_index):
        num_batches = len(unlabeled_index) // batch_size
    else:
        num_batches = len(labeled_index) // batch_size


    for epoch in range(epochs):

        print '-------------------------------------'
        print '> Epoch:', epoch

        # Shuffle the indices:
        np.random.shuffle(unlabeled_index)
        np.random.shuffle(labeled_index)

        for i in range(num_batches):

            print '>> Batch:', '{} / {}'.format(i + 1, num_batches)

            l_index_range = labeled_index[i * batch_size:(i+1) * batch_size]
            x_l_batch = random_crop(X_labeled[l_index_range])
            y_l_batch = y_labeled[l_index_range]

            ul_index_range = unlabeled_index[i * batch_size:(i+1) * batch_size]
            x_ul_batch = random_crop(X_unlabeled[ul_index_range])

            training_round(x_l_batch, y_l_batch, x_ul_batch)

            for ai in range(AUGMENT_TIMES):
                training_round(random_transform(x_l_batch),
                               y_l_batch,
                               random_transform(x_ul_batch))


    end_time = time.time()
    elapsed = end_time - start_time

    print "Training time:", elapsed


if TRAINED_MODEL is None:

    fit_vae_models(ul_x_train, l_x_train, l_y_train, EPOCHS, BATCH_SIZE)

    # Save models:

    save_model(l_vae, "./models/deepLoc_vae/" + f_prefix + "_labelled_vae.kmodel")
    save_model(ul_vae, "./models/deepLoc_vae/" + f_prefix + "_unlabelled_vae.kmodel")

else:

    sl_vae = load_model("./models/deepLoc_vae/" + TRAINED_MODEL + "_labelled_vae.kmodel",
                        custom_objects={'classification_loss': classification_loss,
                                        'labelled_vae_loss': labelled_vae_loss,
                                        'BATCH_SIZE': BATCH_SIZE,
                                        'LATENT_DIM': LATENT_DIM,
                                        'EPSILON': EPSILON})
    l_vae.set_weights(sl_vae.get_weights())

    sul_vae = load_model("./models/deepLoc_vae/" + TRAINED_MODEL + "_unlabelled_vae.kmodel",
                         custom_objects={'unlabelled_vae_loss': unlabelled_vae_loss,
                                         'BATCH_SIZE': BATCH_SIZE,
                                         'LATENT_DIM': LATENT_DIM,
                                         'EPSILON': EPSILON})
    ul_vae.set_weights(sul_vae.get_weights())

# ------------------------------------------
# Model testing
# ------------------------------------------

def evaluate_classifier(clsf, X, y, set_category):

    y_hat = []
    y_true = []

    y_hat_all = []
    y_true_all = []

    prec_dict = {}

    X = X[:, :, :, ::-1]

    data_index = np.arange(len(X))

    np.random.shuffle(data_index)

    for i in range(len(data_index) // BATCH_SIZE):

        idx_range = data_index[i * BATCH_SIZE: (i+1) * BATCH_SIZE]

        batch_pred = clsf.predict(random_crop(X[idx_range]),
                                  batch_size=BATCH_SIZE)

        y_hat_all.append(batch_pred)
        y_true_all.append(y[idx_range])

        y_hat.append(np.argmax(batch_pred, axis=-1))
        y_true.append(np.argmax(y[idx_range], axis=-1))


    y_hat_all = np.concatenate(y_hat_all)
    y_true_all = np.concatenate(y_true_all)

    y_hat = np.choose(np.concatenate(y_hat), localization_terms)
    y_true = np.choose(np.concatenate(y_true), localization_terms)

    """
    y_true_cls, y_true_counts = np.unique(y_true, return_counts=True)
    print dict(zip(y_true_cls, y_true_counts))

    del_indices = np.where(np.isin(y_true, excluded_classes))

    y_hat_all = np.delete(y_hat_all, del_indices, axis=0)
    y_true_all = np.delete(y_true_all, del_indices, axis=0)

    y_hat = np.delete(y_hat, del_indices)
    y_true = np.delete(y_true, del_indices)

    y_true_cls, y_true_counts = np.unique(y_true, return_counts=True)
    true_class_counts = dict(zip(y_true_cls, y_true_counts))

    print true_class_counts
    """

    print '----------------------------'
    print '>>> Accuracy score:', accuracy_score(y_true, y_hat)
    print '----------------------------'

    """
    print '>>> ROC AUC scores:'
    print 'Macro ROC-AUC:', roc_auc_score(y_true_all,
                                          y_hat_all,
                                          average="macro")
    print 'Micro ROC-AUC:', roc_auc_score(y_true_all,
                                          y_hat_all,
                                          average="micro")
    print 'Weighted ROC-AUC:', roc_auc_score(y_true_all,
                                             y_hat_all,
                                             average="weighted")
    for lidx in range(len(localization_terms)):
        print localization_terms[lidx], '| ROC-AUC Score:', roc_auc_score(y_true_all[:, lidx],
                                                                     y_hat_all[:, lidx])
    """

    print '----------------------------'
    print '>>> Average precision scores:'
    macro_prec = average_precision_score(y_true_all,
                                         y_hat_all,
                                         average="macro")
    print 'Macro AP:', macro_prec
    print 'Micro AP:', average_precision_score(y_true_all,
                                               y_hat_all,
                                               average="micro")
    print 'Weighted AP:', average_precision_score(y_true_all,
                                                  y_hat_all,
                                                  average="weighted")
    for lidx in range(len(localization_terms)):
        prec_dict[localization_terms[lidx]] = average_precision_score(y_true_all[:, lidx],
                                                                 y_hat_all[:, lidx])
    print prec_dict

    print '----------------------------'
    print '>> Classification report:'
    print classification_report(y_true, y_hat)
    print '----------------------------'

    # Plot confusion matrices (raw and normalized)
    cnf_matrix = confusion_matrix(y_true, y_hat,
                                  labels=localization_terms)

    plot_confusion_matrix(cnf_matrix,
                          localization_terms,
                          len(y_hat_all),
                          set_category)

    plot_confusion_matrix(cnf_matrix,
                          localization_terms,
                          len(y_hat_all),
                          set_category,
                          normalize=True)

    generate_avg_precision_plot(prec_dict,
                                macro_prec,
                                localization_terms,
                                len(y_hat_all),
                                set_category)



"""
img_classifier = Model(inputs=[x_input], outputs=[y_pred])

print '----------------------------------'
print ' **|*|** Model Evaluation **|*|** '
print '----------------------------------'

print '---> Evaluation on training data <---'
evaluate_classifier(img_classifier, x_train, y_train, 'Training')

print '---> Evaluation on the validation data <---'
evaluate_classifier(img_classifier, x_valid, y_valid, 'Validation')

print '---> Evaluation on testing data <---'
evaluate_classifier(img_classifier, x_test, y_test, 'Test')

print '---> Evaluation on the WT data <---'
evaluate_classifier(img_classifier, x_wt, y_wt, 'TransferWT')

print '---> Evaluation on the SWAT-RFP data <---'
evaluate_classifier(img_classifier, x_rfp, y_rfp, 'TransferRFP')

"""

def vis_layer_activtions(layer_name, X, y, conv_layer_pool=None):

    l_act = []
    y_true = []

    intermed_model = Model(inputs=ul_vae.input,
                          outputs=ul_vae.get_layer(layer_name).output)

    if conv_layer_pool == 'max':
        layer_name += '_max'
    elif conv_layer_pool == 'avg':
        layer_name += '_avg'

    data_index = np.arange(len(X))
    np.random.shuffle(data_index)

    for i in range(len(data_index) // BATCH_SIZE):

        idx_range = data_index[i * BATCH_SIZE: (i+1) * BATCH_SIZE]

        batch_pred = intermed_model.predict(random_crop(X[idx_range]),
                                            batch_size=BATCH_SIZE)

        if conv_layer_pool is not None:
            if conv_layer_pool == 'max':
                batch_pred = np.max(batch_pred, axis=-1)
            elif conv_layer_pool == 'avg':
                batch_pred = np.mean(batch_pred, axis=-1)

            batch_pred = batch_pred.reshape(batch_pred.shape[0], -1)


        l_act.append(batch_pred)
        y_true.append(np.argmax(y[idx_range], axis=-1))

    l_act = np.concatenate(l_act)
    y_true = np.concatenate(y_true)

    generate_tsne_plot(layer_name, l_act, y_true, perplexity=25)
    generate_tsne_plot(layer_name, l_act, y_true, perplexity=50)
    generate_tsne_plot(layer_name, l_act, y_true, perplexity=75)
    generate_tsne_plot(layer_name, l_act, y_true, perplexity=100)



# Encoder network layers:

vis_layer_activtions('e_conv_1_ACT', x_test, y_test,
                     conv_layer_pool='max')
vis_layer_activtions('e_conv_1_ACT', x_test, y_test,
                     conv_layer_pool='avg')

vis_layer_activtions('e_conv_2_ACT', x_test, y_test,
                     conv_layer_pool='max')
vis_layer_activtions('e_conv_2_ACT', x_test, y_test,
                     conv_layer_pool='avg')

vis_layer_activtions('e_conv_3_ACT', x_test, y_test,
                     conv_layer_pool='max')
vis_layer_activtions('e_conv_3_ACT', x_test, y_test,
                     conv_layer_pool='avg')

vis_layer_activtions('e_FC_ACT', x_test, y_test)

# Classifier network layers:
vis_layer_activtions('cls_conv_1_ACT', x_test, y_test,
                     conv_layer_pool='max')
vis_layer_activtions('cls_conv_1_ACT', x_test, y_test,
                     conv_layer_pool='avg')

vis_layer_activtions('cls_conv_2_ACT', x_test, y_test,
                     conv_layer_pool='max')
vis_layer_activtions('cls_conv_2_ACT', x_test, y_test,
                     conv_layer_pool='avg')

vis_layer_activtions('cls_conv_3_ACT', x_test, y_test,
                     conv_layer_pool='max')
vis_layer_activtions('cls_conv_3_ACT', x_test, y_test,
                     conv_layer_pool='avg')

vis_layer_activtions('cls_conv_4_ACT', x_test, y_test,
                     conv_layer_pool='max')
vis_layer_activtions('cls_conv_4_ACT', x_test, y_test,
                     conv_layer_pool='avg')

vis_layer_activtions('cls_FC_1_ACT', x_test, y_test)

# Decoder network layers:

vis_layer_activtions('d_FC_ACT', x_test, y_test)

vis_layer_activtions('d_conv_1_ACT', x_test, y_test,
                     conv_layer_pool='max')
vis_layer_activtions('d_conv_1_ACT', x_test, y_test,
                     conv_layer_pool='avg')

vis_layer_activtions('d_conv_2_ACT', x_test, y_test,
                     conv_layer_pool='max')
vis_layer_activtions('d_conv_2_ACT', x_test, y_test,
                     conv_layer_pool='avg')

vis_layer_activtions('d_conv_3_ACT', x_test, y_test,
                     conv_layer_pool='max')
vis_layer_activtions('d_conv_3_ACT', x_test, y_test,
                     conv_layer_pool='avg')


"""
samp_model = Model(x_input, sampling_layer)

vis_layer_activtions('samp', samp_model, x_test, y_test)

y_pred_model = Model(x_input, y_pred)

vis_layer_activtions('ypred', y_pred_model, x_test, y_test)
"""

# ------------------------------------------
# Labelled VAE
# ------------------------------------------

def vis_image_pair(v_model, X, y, indices_to_plot, prefix=""):

    rec_imgs = []
    y_true = []

    data_index = np.arange(len(X))

    for i in range(len(data_index) // BATCH_SIZE):

        idx_range = data_index[i * BATCH_SIZE: (i+1) * BATCH_SIZE]

        X_pred, y_pred = v_model.predict([random_crop(X[idx_range]),
                                          y[idx_range]],
                                          batch_size=BATCH_SIZE)

        rec_imgs.append(X_pred)
        y_true.append(y[idx_range])

    rec_imgs = np.concatenate(rec_imgs)
    y_true = np.concatenate(y_true)

    for idx in indices_to_plot:
        if idx < BATCH_SIZE * (len(data_index) // BATCH_SIZE):
            plot_deeploc_image(X[idx], "./plots/pairs/" + prefix + "_" + str(idx) + "_orig.png")
            plot_deeploc_image(rec_imgs[idx], "./plots/pairs/" + prefix + "_" +  str(idx) + "_rec.png")


def generate_img_pair(v_model, img_idx):

    img = np.repeat(crop_x_test[[img_idx]], 200, axis=0)
    y_label = np.repeat(y_test[[img_idx]], 200, axis=0)

    img_in = img[:, :, :, ::-1]

    X_pred, y_pred = v_model.predict([img_in, y_label],
                                     batch_size=BATCH_SIZE)

    img_out = X_pred[0][:, :, ::-1]

    plot_deeploc_image(img[0], "./plots/pairs/report_orig.png")
    plot_deeploc_image(img_out, "./plots/pairs/report_rec.png")


# generate_img_pair(l_vae, 4280)


"""
for yi, lt in enumerate(localization_terms):

    cls_y = np.zeros(len(localization_terms))
    cls_y[yi] = 1

    cls_y = np.repeat([cls_y], len(x_test), axis=0)

    vis_image_pair(l_vae, x_test, cls_y,
                   [3393, 2931, 4514, 3790, 39, 492, 2814,
                    3785, 4302, 3611, 4280, 2483, 4389],
                  prefix=lt)
"""

# ------------------------------------------
# Generator network
# ------------------------------------------

def show_random_generated_images(dec_model, samples=6, div_size=6):

    img_mat = np.zeros((NUM_CLASSES,
                        samples * 2,
                        IMG_ROWS, IMG_COLS, 3))

    for j in range(0, 2 * samples, 2):
        for i in range(NUM_CLASSES):

            y_sample = np.zeros(NUM_CLASSES).reshape(1, NUM_CLASSES)
            y_sample[0, i] = 1

            z_sample = np.random.normal(size=LATENT_DIM).reshape(1, LATENT_DIM)

            x_decoded = dec_model.predict([y_sample, z_sample], batch_size=1)
            x_decoded = x_decoded[:, :, :, ::-1]

            decoded_img = x_decoded.reshape(IMG_ROWS, IMG_COLS, IMG_CHNLS)

            closest_img = find_closest_image(crop_x_train, x_decoded)

            img_mat[i, j] = add_blue_channel(decoded_img)
            img_mat[i, j + 1] = add_blue_channel(closest_img)


    xlab = np.repeat([['Generated image', 'Training image']],
                     samples,
                     axis=0).ravel()

    h_divs = [0] + list(np.repeat([[0, 6]], samples, axis=0).ravel())[:-1]

    generate_montage("./plots/test_montage_hv.pdf",
                     img_mat,
                     h_div_size=h_divs,
                     x_tick_labels=xlab,
                     y_tick_labels=localization_terms,
                     x_axis_label="Images Generated Using Random Samples from the Latent Space vs. Closest Training Image",
                     y_axis_label="Localization Pattern")


def interpolate_img_pair(enc_model, dec_model,
                         img_pairs,
                         num_intermediates=8):

    img_mat = np.zeros((len(img_pairs),
                        num_intermediates + 2,
                        IMG_ROWS, IMG_COLS, 3))

    y_labels = []

    for pid in range(len(img_pairs)):

        img1_idx, img2_idx = img_pairs[pid]

        y_label_1 = localization_terms[np.argmax(y_test[[img1_idx]], -1)[0]]
        y_label_2 = localization_terms[np.argmax(y_test[[img2_idx]], -1)[0]]

        y_labels.append(y_label_2 + " to " + y_label_1)

        img1 = np.repeat(crop_x_test[[img1_idx]], 100, axis=0)
        img2 = np.repeat(crop_x_test[[img2_idx]], 100, axis=0)

        preds = enc_model.predict(np.append(img1, img2, axis=0),
                                  batch_size=BATCH_SIZE)

        img1_embed = preds[0]
        img2_embed = preds[-1]


        img_mat[pid, 0] = add_blue_channel(crop_x_test[[img2_idx]])

        for idx, alpha in enumerate(np.linspace(0.0, 1.0, num_intermediates)):

            if idx < num_intermediates // 2:
                y_sample = y_test[[img2_idx]]
            else:
                y_sample = y_test[[img1_idx]]

            int_vec = (alpha * img1_embed +
                       (1 - alpha) * img2_embed).reshape(1, LATENT_DIM)

            decoded_img = dec_model.predict([y_sample, int_vec], batch_size=1)
            decoded_img = decoded_img[:, :, :, ::-1]

            img_mat[pid, idx + 1] = add_blue_channel(decoded_img)

        img_mat[pid, -1] = add_blue_channel(crop_x_test[[img1_idx]])


    x_labels = ["Original Image", "Decoded Image"]
    x_labels += list(np.repeat("", num_intermediates - 2))
    x_labels += x_labels[:2][::-1]

    generate_montage("./plots/test_interp_" + str(num_intermediates) + ".pdf",
                     img_mat,
                     v_div_size=6,
                     x_tick_labels=x_labels,
                     y_tick_labels=y_labels,
                     fig_size=(14, 9))


"""
dec_y = Input(shape=(NUM_CLASSES,))
dec_z = Input(shape=(LATENT_DIM,))
dec_merged = concatenate([dec_y, dec_z])

dec_output = join_network_layers([dec_merged] + decoder_layers)
dec_model = Model([dec_y, dec_z], dec_output)

show_random_generated_images(dec_model)


enc_model = Model(x_input, z_means)

interpolate_img_pair(enc_model, dec_model,
                     [(492, 3790),
                      (3393, 492),
                      (3785, 3393),
                      (3611, 3785),
                      (39, 3611),
                      (2483, 39),
                      (4389, 2483),
                      (4280, 4389),
                      (4302, 4280),
                      (4514, 4302)],
                     num_intermediates=12)
"""

