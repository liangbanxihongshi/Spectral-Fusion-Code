# Spectral-Fusion-Code
This repository contains the code implementation of the spectral fusion CNN architecture from the paper "Dual-Branch Convolutional Neural Network with Attention Modules for LIBS-NIRS Data Fusion in Cement Composition Quantification."
DBAM_CNN(Dual-Branch Convolutional Neural Network with an Attention Module)：This study proposes a DBAM-CNN structure for the fusion of LIBS and NIRS data.The Fig below illustrates the structure of the proposed method, whereby the pre-processed LIBS data, with a size of 496×1, and the pre-processed NIRS data, with a size of 441×1, are fed into two separate CNN branches. The spectral data in both branches is subjected to feature extraction through two CNN modules. Subsequently, the extracted spectral features are fed into a Spatial Attention Module. for further detailed feature processing. This module generates feature maps that enhance the model's focus on key predictive features across all channels, thereby increasing its attention to features that significantly impact prediction. The dimensions of the feature data are then unified to 32×16 via a linear layer, thus facilitating the subsequent feature concatenation step. Following this, the extracted features from the LIBS and NIRS data are concatenated along the channel dimension, resulting in a 32×32 feature dataset. And a Channel Attention Module is employed to assign distinct weights to different channels, thereby guiding the model towards a greater focus on the most impactful feature channels. Subsequently, the high-dimensional 32×32 feature is transformed into a 1024×1 vector through the application of a flatten layer. The resulting vector is then passed through two linear layers with the objective of predicting the concentration of target components.
![image](https://github.com/user-attachments/assets/e50ccae7-f927-402e-ac34-6f6029b27033)
DB_CNN(Dual-Branch Convolutional Neural Network):The two-branch DBAM-CNN with the attention module removed, and the rest of the structure is consistent with the proposed model.
DsLF_CNN(Dataset-Level Fusion CNN): As illustrated in Fig below, this study presents a dataset-level fusion method, data level fusion is achieved by directly concatenating the pre-processed LIBS and NIRS data, which results in an input concatenated spectral data size of 937×1. Subsequently, the CNN structure is employed to extract features, which are then predicted by a fully connected layer.
FLF_CNN(Feature-Level Fusion CNN): As illustrated in Fig below, at the outset of the process, the principal components are extracted from the pre-processed LIBS and NIRS data through the application of PCA. Subsequently, the extracted features are concatenated, and the subsequent steps mirror those of the dataset-level fusion process. The concatenated feature data, comprising 20×1, is subjected to further deep feature extraction through the CNN structure, followed by a fully connected layer for the prediction of quantitative results.
DLF_CNN(Decision-Level Fusion CNN): Firstly, the pre-processed LIBS and NIRS data are subjected to quantitative predictions utilising analogous deep learning structures as previously mentioned(LIBS_CNN and NIRS_CNN). Subsequently, the two predicted values are integrated through multivariable linear regression (MLR) to obtain the prediction results for the decision-level fusion.
![image](https://github.com/user-attachments/assets/f678c06a-544b-49d5-af2d-a74d43b73f56)
