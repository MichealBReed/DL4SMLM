---
title: 'Deep Learning for Single Molecule Localization Microscopy'
tags:
    - Python
    - Machine Learning
    - Microscopy
    - Quantitative Bioimaging
    - Image Processing
    - Spatial Resolution
    - Super Resolution Microscopy
authors:
    - name: Micheal Brandon Reed
      affiliation: "1, 2"
    - name: Reza Zadegan
      affilitation: "1, 2"
affilifations:
    - name: North Carolina Agricultural and Technical State University, Greensboro, North Carolina, United States
      index: 1
    - name: Joint School of Nanoscience and Nanoengineering, Greensboro, North Carolina, United States
      index: 2

---

# Summary

Single Molecule Localization Microscopy (SMLM) enables researchers to interrogate nanoscale spatial details in a range of systems. Biological investigations benefit greatly from SMLM due to its ability to quantitatively
investigate details of great importance such as protein distribution in the cell membrane or protein-protein interactions[@Badley:2017]. However, SMLM set-ups can be quite expensive, imaging times can be lengthy and data
analysis can require expert knowledge. Recently, Deep Learning (DL) algorithms have been developed to reduce imaging time and automate data analysis. Naturally each subsequent model aims to address a different shortcoming
of the prior work and so there is a family of model architectures. However, there is not a singular location for researchers to have access to these models dedicated to SMLM. We developed the Deep Learning for Single
Molecule Localization Microscopy (DL4SMLM) in Python using the PyTorch framework to democratize access to these models and lower the barrier of entry to SMLM.

# Statement of Need
DL4SMLM is a Python package that uses the Pytorch DL framework to implement  Convolutional Neural Nets (CNN) dedicated to SMLM. We created it for researchers and engineers who wish to use DL for SMLM in whichever field they
practice granted they have sufficient data. Currently, there are six models:Super Resolution Convolutional Neural Network (SRCNN)[@Dong:2015], Deep-Stochastic Optical Reconstruction Microscopy (Deep-STORM)[@Nehme:2018],
Skip-STORM, Unet for STORM (U-STORM), Deep Residual STORM (DRL-STORM)[@Yao:2020], and Fast Dense Image Reconstruction based Deep Learning STORM (FID-STORM)[@Zhou:2023]. SRCNN, Skip-STORM and U-STORM are novel architectures
to the SMLM field. While SRCNN has been applied to perform the super resolution task in more traditional image processing tasks, its utility in SMLM is unknown. Skip-STORM’s architecture is similar to Deep-STORM but a skip
connection between the initial image and final layer is introduced to enable it to have a spatial context during its reconstruction phase which could assist it in emitter dense images. U-STORM is inspired by U-Net and
adopts a similar biphasic architecture where in the first half the initial image is max pooled while simultaneously increasing channel number and the second half involves the up sampling and reduction in channel. It also
contains skip connections that connect the features of the first half to the those of second half during image reconstruction. 
These models are implemented using the object-oriented programming paradigm enabling researchers to instantiate multiple models for a single data set to allow downstream comparison of inference performance. Additionally, we
have implemented the L1L2 Loss metric first introduced by Nehme et al 2018 and used for all subsequent architectures implemented in this library. Other features include functions that automate the Training and Validation
process for a user-set number of epochs in addition to a function that automates the inference procedure of a trained model on a test data set and reports the Structured Similarity Index Measure (SSIM) and the Normalized
Mean Square Error (NMSE). These metrics are often used to report the inference performance of models dedicated to SMLM. To assist researchers in visualizing the performance of their model, we have implemented a function
that visualizes the diffraction limited image of the emitters, its super-resolved version, and the predicted super-resolved image according to the trained model. This empowers the researcher to troubleshoot the performance
of the model and determine if the model should be retrained with different parameters or to proceed with the experiment. 

# Acknowledgements
This work is supported by NIH award 1R16GM145671 and NSF award MCB 2027738. This work was performed at the Joint School of Nanoscience and Nanoengineering (JSNN), a member of the National Nanotechnology Coordinated Infrastructure (NNCI), which is supported by the National Science Foundation (Grant ECCS-2025462).


# References