# U-Net-for-Covid-19-CT-Binary-Segmentation-in-Keras
A machine learning model for the binary segmentation of Covid-19 in CT images.


## Disclaimer
  This project should be used for research purposes only. The purpose of the project is to show the potential of machine learning in medical support such as diagnosis systems. Although the network has achieved fair performance, it is not meant to be an alternative to professional medical diagnosis.

## Introduction
  Biomedical image processing is an active and growing discipline. Within the field, the segmentation of medical images is a crucial step for diagnosis, monitoring, and treatment planning in clinical studies. While manual segmentation of images is a tedious and lengthy task, advances in machine learning algorithms can be leveraged for fast, efficient, and automated segmentation. One such method is U-Net [1], consisting of symmetrical contracting and expanding paths that allow such a network to both capture context and localize features. This architecture is implemented in this short introductory project to python and ML, for the binary segmentation of Covid-19 symptoms in chest axial CT images.

## Data and Preprocessing 
  The Covid-19 segmentation dataset [2] from Medical Segmentation is utilized for training and testing. This dataset consists of 100 axial CT images from more than 40 patients diagnosed with Covid-19, as well as 100 segmentation masks produced by expert radiologists [3]. Although labelled for three symptoms: ground glass opacification, consolidations, and pleural effusions, the masks are treated as binary images for simplicity. Images and masks are converted from NII (or NIfTI) files to 4D NumPy arrays using NiBabel, and a train-test split of 60/40 is performed. Some images and masks are shown below in Image 1 [2]. Refer to Appendix 1 for the full dataset.
  
  Due to the relatively small dataset size, image augmentations must be applied to allow the network to generalize. As such, Keras’ ImageDataGenerator is employed to perform random rotation, zoom, x-axis shifts, and y-axis shifts. A data generator pipeline is created for training images and masks separately, and is then zipped together for implementation in the U-Net. 
 
### Image 1: Images and Masks from the Covid-19 segmentation dataset
(./utils/sample-image-mask-plots.PNG)  

## U-Net Architecture
  As mentioned above, U-Net is an effective method for image processing. Developed for biomedical image segmentation by Olaf Ronneberger et al. in 2015, U-Net has quickly become popular for its speed and effectiveness. The contracting path (or the encoder) uses traditional convolutional and pooling layers to capture an image’s context. The expansive path (or decoder) is symmetric with the encoder, using transposed convolutional layers for “upsampling”. On the decoder, “upsampled” layers are concatenated with the corresponding layers on the encoder—this is what gives a U-Net its signature ability to generate deep localizable features, allowing for effective segmentation. These two symmetrical paths result in the architecture’s U-shape, leading to the network’s name: U-Net. The final layer in a U-Net is a convolutional layer with 1 (1x1) filter, which outputs a binary value for every pixel in the input image. In essence, U-Net performs classification for every pixel in an image. 
 
## Training and Results
  The network was written in python using Keras and trained in Google Colab. Initially using binary cross-entropy as its loss function, the network achieved good accuracy but poor precision and recall. Due to the imbalance between the background and the segmentation masks in some cases, a model that returns a blank image would still achieve decent accuracy. As such, a loss function derived from Sørensen–Dice index is implemented, which is designed to measure the relative overlap between predictions and ground truths. This way, the loss function is independent of object sizes. Precision, recall, Jaccard distance, area under ROC curve, and F1 score are also used to monitor the network during training. 
  
  Within the project’s time frame of a week, the network was able to achieve a Dice index/score of 0.79 with the RMSprop optimizer at a learning rate of 0.0001.
  
## Discussion
  The results of this project demonstrate the effectiveness of U-Net and the viability of Keras as an introductory machine learning tool for beginners. In just one week, a high school student with beginner proficiency in python and no experience in ML was able to complete this project with fair results, while gaining a substantial understanding of the design, training, and evaluation of deep learning models. Taking at most an hour to converge using Google Colab GPUs, the architecture of this U-Net allowed for quick training cycles and rapid inference of chest CT images, illustrating its potential in biomedical applications and image segmentation as a whole. However, other implementations for U-Net might require a significant increase in computational power due to more layers, higher training image resolution, or other hyperparameter changes. Although only 512x512 images were employed in training and testing, the network is, in fact, input image size agnostic because it doesn’t incorporate any fully connected layers— this also aids in reducing the model’s size.

  Although the RT-PCR (Reverse Transcription Polymerase Chain Reaction) test [4] and antigen tests exist for the diagnosis of Covid-19, the search for alternative screening and segmentation methods with higher accuracy have led researchers to experiment with learning networks such as U-Net. Other automated methods involving lung CT scans such as semi-supervised shallow neural networks [5] and implicit reverse attention and explicit edge-attention [6] may be useful for more widespread testing, severity assessments, treatment planning. However, due to the high variation in infected regions, more radiologist-annotated CT images must be gathered to develop a highly reliable and accurate medical tool. This is challenging because high-quality pixel-specific segmentation masks are laborious and time-consuming.
  
  Moving forward, elastic augmentations to the dataset could be used to simulate variation in human tissue, while generative models such as Generative Adversarial Networks [7] could be used to synthesize more images and masks from the dataset’s distribution. In addition, Object Region Mining with Adversarial Erasing [8] could be used to “mine” infected regions using classification networks while finding new infected regions by adversarially erasing “mined” regions. 
 
## References
[1] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional Networks for Biomedical Image Segmentation,” arXiv, 2015.  
[2] “COVID-19 CT segmentation dataset,” https://medicalsegmentation.com/covid19/, 2020.  
[3] H. B. Jenssen, “Covid-19 radiology — data collection and preparation for Artificial Intelligence,” Medium, 2020.   
[4] T. Ai, Z. Yang et al., “Correlation of Chest CT and RT-PCR Testing in Corona Virus Disease 2019 (COVID-19) in China: A Report of 1014 Cases,” Radiology, vol. 2019, p. 200642, 2020.  
[5] D. Konar, B. K. Panigrahi, S. Bhattacharyya, and N. Dey, “Auto-diagnosis of COVID-19 using Lung CT Images with Semi-supervised Shallow Learning Network,” Research Square, 2020.  
[6] D. P. Fan, T. Zhou et al., “Inf-Net: Automatic COVID-19 Lung Infection Segmentation from CT Images,” arXiv, 2020.  
[7] I. Goodfellow, J. Pouget-Abadie et al., “Generative Adversarial Networks,” arXiv, 2014.  
[8] Y. Wei, J. Feng et al., “Object Region Mining with Adversarial Erasing: A Simple Classification to Semantic Segmentation Approach,” arXiv, 2018.  
[9] M. Ilyas, H. Rehman, A. Nait-ali, “Detection of Covid-19 From Chest X-ray Images Using Artificial Intelligence: An Early Review,” arXiv, 2020.  

