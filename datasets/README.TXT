datasets for DeepLoc model described in:
"Expediting the automated analysis of high content microscopy data with deep learning"

Datasets described in detail in the publication.

Each hdf5 file contains 'data1' and 'Index1' elements.
'data1' elements are cropped images of single cells (64x64) x number_of_channels flattened into vectors.
'Index1' elements are the protein localization labels for each single cell using one hot encoding.

More information on the data and label columns is given in the hdf5 attribute sets.

Use the h5py library to access the datasets

Example:

import h5py

f = h5py.File('Chong_valid_set.hdf5','r')
for key in f.keys():
   print key,f[key]
for key in f.attrs.keys():
   print key,f.attrs[key]
f.close()
 
