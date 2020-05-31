# DeepLoc-VAE
Semi-supervised Classification of Protein Subcellular Localization Patterns in Microscopy Images with Variational Autoencoders

Author: Shadi Zabad, University of Toronto

This is a report on my course project for CSC2548 (Machine Learning for Computer Vision), taught by Sanja Fidler.

![Image description](https://github.com/shz9/DeepLoc-VAE/blob/master/model.png)

Recent years have witnessed an unprecedented growth in the number and scale ofmicroscopy images generated from a variety of experimental protocols. In response,numerous image analysis pipelines have been proposed to automate various aspectsof  the  process  of  scientific  discovery,  from  image  segmentation  to  high-levelannotations. One particular research area that has garnered some attention is theclassification of protein subceullar localization patterns from microscopy images.Previous methods applied in this setting relied on fully-supervised machine learningtechniques to automate the annotation process. Here, we improve on previous workby deploying deep generative models to perform semi-supervised classificationof protein subcellular localization patterns.  We show that, despite using only asmall fraction of the labels, the semi-supervised generative model achieves state-of-the-art results. In addition, we leverage the generative capabilities of the modelto explore some of the properties of the inferred latent representations.

## Qualitative Assessment of the Generative Capabilities:

![Image description](https://github.com/shz9/DeepLoc-VAE/blob/master/pair_fig.png)
