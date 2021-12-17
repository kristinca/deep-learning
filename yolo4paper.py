""" My notes from the paper 'YOLOv4: Optimal Speed and Accuracy of Object Detection' by
            Alexey Bochkovskiy, Chien-Yao Wang and Hong-Yuan Mark Liao """


# Some features, such as batch-normalization and residual-connections
# are applicable to the majority of models, tasks, and datasets:
# -> Weighted-Residual-Connections (WRC)
# -> Cross-Stage-Partial-connections (CSP)
# -> Cross mini-Batch
# -> Normalization (CmBN)
# -> Self-adversarial-training (SAT)
# -> Mish-activation

# The majority of CNN-based object detectors are applicable ONLY FOR RECOMENDATION SYSTEMS:
# -> searching for free parking spaces via urban video cameras -> SLOW ACCURATE MODELS
# -> car collision warning -> FAST INACCURATE MODELS

# The most accurate modern neural networks DO NOT OPERATE IN REAL TIME and require large number of GPUs
# for training with a large mini-batch-size.
# We address such problems through creating a CNN that OPERATES IN REAL TIME ON A CONVENTIONAL GPU
# and for which training requires ONLY ONE CONVENTIONAL GPU.

# YOLOv4 vs and other state-of-the-art object detectors:
# -> runs TWICE FASTER than EfficientDet with comparable performance
# -> improves YOLOv3’s AP by 10%
# -> improves FPSand 12%


# Object detection models

# A modern detector is usually composed of two parts:
# -> a backbone which is pre-trained on ImageNet
# -> a head which is used to predict classes and bounding boxes of objects

# It is also possible to make a two stage object detector an anchor-free object detector ex. RepPoints.
# One-stage object detector -> the most representative models are YOLO, SSD and RetinaNet.
# Anchor-free one-stage object detectors -> CenterNet, CornerNet, FCOS.

# The neck of an object detector:  layers inserted between backbone and head
# usually used to collect feature maps from different stag.
# A neck is composed of several bottom-up paths and several top down paths.
# Networks equipped with this mechanism:
# -> Feature Pyramid Network (FPN), Path Aggregation Network (PAN), BiFPN, NAS-FPN.

# Parts of an ordinary object detector:
# 1. Input: Image, Patches, Image Pyramid
# 2. Backbones: VGG16, ResNet-50, SpineNet, EfficientNet-B0/B7, CSPResNeXt50, CSPDarknet53
# 3.  Neck:
#  -> Additional blocks: SPP, ASPP, RFB, SAM
#  -> Path-aggregation blocks: FPN, PAN, NAS-FPN, Fully-connected FPN, BiFPN, ASFF, SFAM
# 4.  Heads:
#  -> Dense Prediction (one-stage):
#       -> Anchor based: RPN, SSD, YOLO, RetinaNet
#       -> Anchor free: CornerNet, CenterNet, MatrixNet, FCOS
#  -> Sparse Prediction (two-stage):
#         -> Anchor based: Faster R-CNN, R-FCN, Mask RCNN
#         -> Anchor free: RepPoints

# Usually, a conventional object detector is trained offline
# Commonly used data augmentation methods:
# ->pixel-wise adjustments; all original pixel information in the adjusted area is retained
#  -> photometric distortions : adjust the brightness, contrast, hue, saturation, and noise of an image
#  -> geometric distortions : add random scaling, cropping, flipping, and rotating.
# Random erase and CutOut can randomly select the rectangle region in an image
# and fill in a random or complementary value of zero.
# hide-and-seek and grid mask randomly or evenly select multiple rectangle regions in an image
# and replace them to all zeros.

# The MixUp method uses two images to multiply and superimpose with different coefficient ratios
# and then adjusts the label with these superimposed ratios.
# The CutMix method is used to cover the cropped image to rectangle region of other images
# and adjusts the label according to the size of the mix area.
# Data augmentation with style transfer GAN -> effectively reduces the texture bias learned by CNN.

# One-hot hard representation -> a representation scheme

# The traditional object detector usually uses Mean Square Error (MSE)
# to directly perform regression on the center point coordinates and height and width of the BBox:
# i.e., {xcenter, ycenter, w, h}, or the upper left point
# and the lower right point,i.e., {xtop lef t, ytop lef t, xbottom right, ybottom right}.
# Anchor-based method, it is to estimate the corresponding offset,
# i.e., {xcenter of f set, ycenter of f set, wof f set, hof f set}
# and {xtop lef t of f set, ytop lef t of f set, xbottom right of f set, ybottom right of f set}.

# To directly estimate the coordinate values of each point of the BBox
# is to treat these points as independent variables, it DOES NOT consider the integrity of the object itself.


# IoU -> a scale invariant representation:
# Calculation of the four coordinate points of the BBox by:
# 1. executing IoU with the ground truth
# 2. connecting the generated results into a whole code.

# GIoU loss:
# includes the shape and orientation of object in addition to the coverage area
# 1. find the smallest area BBox that can simultaneously cover the predicted BBox and ground truth BBox
# 2. use this BBox as the denominator to replace the denominator originally used in IoU loss

# DIoU loss:
# -> considers the distance of the center of an object

#  CIo loss:
# -> simultaneously considers the
# overlapping area, the distance between center points, and the aspect ratio
# can achieve BETTER CONVERGENCE SPEED AND ACCURACY on the BBox regression problem.

# Common modules that can be used to enhance receptive field:
# -> SPP: integrates SPM into CNN and use max-pooling operation instead of bag-of-word operation
# -> ASPP: max-pooling of stride equals to 1 to several 3 × 3 kernel size, dilated ratio = k,
#  stride = 1 in dilated convolution operation.
# -> RFB: uses several dilated convolutions of k×k kernel, dilated ratio = k,
# stride = 1 to obtain a more comprehensive spatial coverage than ASPP.

# The attention module that is often used in object detection:
# -> channel-wise attention
#   -> Squeeze-and-Excitation (SE) module -  more appropriate to be used in MOBILE DEVICES
# -> pointwise attention
#   -> Spaial Attention Module (SAM)) module -  DOES NOT AFFECT not affect the speed of inference on the GPU

# A good activation function can make the gradient more efficiently propagated,
# + it will not cause too much extra computational cost
#   -> ReLU -> substantially solve the gradient vanish problem (that is in tanh and sigmoid activation function)
#      -> LReLU and PReLU: solve the problem that the gradient of ReLU is zero when the output is less than zero
#      -> ReLU6 and hard-Swish: specially designed for quantization networks
#      -> SELU: self-normalizing a neural network
#      ->  Swish and Mish: continuously differentiable activation functions


# post processing -> NO LONGER REQUIRED in the subsequent development of an anchor-free method.
# NMS : post-processing method used in deep learning-based object detection
# -> filters those BBoxes that badly predict the same object
# -> only retains the candidate BBoxes with higher response.
# -> this method is e is consistent with the method of optimizing an objective function
#  DIoU NMS:
# -> adds the information of the center point distance to the BBox screening process on the basis of soft NMS.


# The basic aim is fast operating speed of neural network in production systems
# and optimization for parallel computations, rather than the low computation volume theoretical indicator (BFLOP)

# A reference model which is optimal for classification is NOT ALWAYS optimal for a detector
# The detector also needs :
# -> higher input network size (resolution) – for detecting multiple small-sized objects
# -> more layers – for a higher receptive field to cover the increased size of input network
# -> more parameters – for greater capacity of a model to detect multiple objects of different sizes in a single image


# For improving the object detection training, a CNN uses:

# 1. Activations: ReLU, leaky-ReLU, parametric-ReLU, ReLU6, SELU, Swish, Mish
# 2. Bounding box regression loss: MSE, IoU, GIoU, CIoU, DIoU
# 3. Data augmentation: CutOut, MixUp, CutMix
# 4. Regularization method: DropOut, DropPath, Spatial DropOut, DropBlock
# 5. Normalization of the network activations by their mean and variance:
#   -> Batch Normalization, Cross-GPU Batch Normalization (CGBN or SyncBN),
#      Filter Response Normalization (FRN), Cross-Iteration Batch Normalization (CBN)
# 6. Skip-connections: Residual connections, Weighted residual connections,
#    Multi-input weighted residual connections, Cross stage partial connections (CSP)

# training activation function:
# DO NOT USE PReLU, SELU or ReLU6!!!
#   -> PReLU and SELU are more difficult to train
#   -> ReLU6 is specifically designed for quantization network
#  the DropBlock is used in this paper as regularization method.

# Mosaic : new data augmentation method that mixes 4 training images,
# batch normalization calculates activation statistics from 4 different images on each layer
# CutMix mixes only 2 input images

# Self-Adversarial Training (SAT): a new data augmentation method that operates in 2 forward backward stages:
# 1. stage: the neural network alters the original image instead of the network weights
# 2. stage: the neural network is trained to detect an object on this modified image in the normal way.


# YOLOv4

# Parts:
# 1. Backbone: CSPDarknet53
# 2. Neck: SPP, PAN
# 3. Head: YOLOv3

# YOLO v4 uses:
# backbone: CutMix and Mosaic data augmentation, DropBlock regularization, Class label smoothing
# detector: CIoU-loss, CmBN, DropBlock regularization, Mosaic data augmentation, Self-Adversarial Training,
# Eliminate grid sensitivity, Using multiple anchors for a single ground truth,
# Cosine annealing scheduler, Optimal hyperparameters, Random training shap
# detector: Mish activation, SPP-block, SAM-block, PAN path-aggregation block, DIoU-NMS.
