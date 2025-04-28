# Breast-Cancer-Detection-Using-Multiscale-Cascaded-Convolutions

# Abstract:
Breast cancer is one of the leading causes of cancer-related deaths worldwide. Early and
accurate detection is crucial for effective treatment and patient survival. Breast cancer
detection has been extensively studied using various deep learning models. Traditional
methods such as CNN, Multi-Channel CNN (MCCNN), and U-Net have been widely used for
feature extraction, classification, and segmentation. CNN has been effective in identifying
patterns in medical images but struggles with fine-grained classification. MCCNN improves
feature extraction by utilizing multiple channels but still faces challenges with accuracy and
false positives. U-Net is a popular choice for segmentation, yet it lacks robustness in handling
complex tumor structures, leading to misclassification and segmentation errors. Despite their
success, these models suffer from high false positive rates, low sensitivity, and misclassification
of tumors. To overcome the limitations of these we propose RAD-D NET, an advanced deep
learning architecture designed to enhance breast cancer detection. RAD-D NET integrates
optimized feature extraction techniques, deep learning layers and improves segmentation
mechanisms to increase classification accuracy and reduce false positives. Our experimental
results demonstrates that RAD-D NET outperforms CNN, MCCNN, and U-Net by achieving
higher accuracy, better sensitivity, and improved tumor localization. The proposed model
provides more reliable and precise breast cancer detection, aiding in early diagnosis and
effective treatment planning.

# Problem Statement:
Breast cancer detection from ultrasound images is difficult due to noise, variability, and the
need for manual analysis by experts. There is no automated system that can accurately
segment tumor regions and classify cancer types. This project aims to develop deep learning
models to automate both segmentation and classification, providing faster and more accurate
diagnosis.

# Dataset Sources:
https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset

https://data.mendeley.com/datasets/7fvgj4jsp7/1

# Dataset Preprocessing:

restructured and merged

Directory Structure:
breast_cancer_dataset/
├── Normal/
│ ├── Images/
│ └── Masks/
├── Benign/
│ ├── Images/
│ └── Masks/
└── Malignant/
├── Images/
└── Masks/

# Dataset Enhancement Pipeline:
The breast ultrasound images undergo a series of enhancement techniques, where the output of
each step is used as the input for the next, improving image quality and tumor visibility.

Contrast Limited Adaptive Histogram Equalization (CLAHE): Enhances local contrast to make
hidden tumor areas clearer.

Variant-Enhanced Mechanism: Highlights important differences to emphasize tumor edges
and reduce noise.

Wavelet Features and Contrast Enhancement: Enhances fine tumor details by breaking the
image into multiple levels.

Fuzzy Logic Transformation: Handles unclear boundaries and improves image clarity.

![Screenshot 2025-04-28 210650](https://github.com/user-attachments/assets/955d373d-e39d-4652-baa9-27d0f8378d9e)

Resizing: After all enhancements, both images and masks are resized to 224×224×3 for
consistency in model training.

Augmentation:
To improve model performance and prevent overfitting, data augmentation is applied only
to the training dataset after splitting the dataset into train (60%), validation (20%), and
test (20%) sets. Various augmentation techniques are used, including random rotation
(rotating images by a random angle), horizontal shift (moving images left or right), vertical
shift (moving images up or down), and random zoom (zooming in or out). These
transformations increase the size of the training dataset and help the model recognize tumors
from different angles and positions. This process enhances the model's ability to generalize to
new data while reducing the risk of overfitting.

# Segmentation Model:

We utilize a Multiscale Cascaded Convolution with Residual Attention-based Double
Decoder Network (RAD-D Net) to accurately segment breast cancer regions from ultrasound
images. The model follows a U-shaped architecture comprising four encoder blocks, a
multiscale cascaded convolution bridge, and four decoder blocks.

Encoder Path: The encoder uses multi-scale cascaded convolution blocks with 3×3, 5×5, and
7×7 kernels to capture diverse spatial features. Each block performs down-sampling to
extract discriminative features from the input image.

Decoder Path: The decoder path employs a residual attention mechanism to focus on
important tumor regions. Skip connections are used to transfer learned features from the
encoder to the decoder, enhancing feature recovery and reducing information loss.

Output Layer: The final segmented tumor region is generated using a 1×1 convolution followed
by a sigmoid activation function. This outputs a binary mask that highlights the tumor areas in the
breast ultrasound image.

Model Training:

The segmentation model is trained using the Adam optimizer with a learning rate of 0.001 and a
combined loss function to enhance performance. Key metrics like accuracy, precision, and recall are
monitored during training. The training process uses a batch size of 8, with calculated steps per epoch
based on the dataset size. To improve model efficiency and prevent overfitting, the following callbacks
like ModelCheckpoint, EarlyStopping etc are used.

Model Evaluation:
The segmentation model is evaluated using accuracy, precision, and recall to measure segmentation
performance. ModelCheckpoint saves the best model based on validation loss, while EarlyStopping
prevents overfitting by restoring the best weights after 6 epochs without improvement.

Reference:

https://ieeexplore.ieee.org/document/10599427

# Classification Model:

We implement a novel RAD-D Net model, a convolutional neural network (CNN) designed to
classify breast cancer into three categories—Normal, Benign, and Malignant. The model is divided
into three key stages:

Stage 1: This stage consists of a convolutional layer, followed by batch normalization and
the Mish activation function to capture low-level features.

Stage 2: Seven convolutional blocks are used in this stage, each containing convolutional
layers, batch normalization, Mish activation, and max-pooling to extract complex patterns.
The outputs of blocks 5, 6, and 7 are concatenated for better feature fusion.

Stage 3: The third stage consists of four dense blocks, each with fully connected layers and
dropout to prevent overfitting. The output is processed using a softmax layer for classification.

Model Training:
The classification model is trained using the Adam optimizer with a learning rate of 0.0001 and
categorical cross-entropy loss function.The model is trained for 100 epochs with early stopping
to prevent overfitting, model checkpointing to save the best model, and learning rate reduction
to adjust the learning rate when validation loss stops improving.

Model Evaluation:
Performance is measured using accuracy. Metrics like accuracy, precision, and recall. The model
is compared with other CNN architectures like AlexNet, ResNet-18, and VGG16, where RAD-D
Net outperforms them in classification accuracy.
Finally, the result analysis of both the segmentation and classification models is presented
here.

Reference:

https://ieeexplore.ieee.org/abstract/document/10169383

# Conclusion:

The Breast Cancer Detection System using Multiscale Cascaded Convolutions with RAD-D Net significantly enhances the accuracy and efficiency of breast cancer diagnosis by leveraging deep learning for automated feature extraction and classification. Its multiscale cascaded convolutional approach enables precise identification of malignant tumors while reducing false positives and false negatives. The integration of a web-based interface and cloud deployment ensures accessibility for radiologists and healthcare professionals, streamlining the diagnostic process. Our model RAD-D net achieves 98% classification accuracy and 92% segmentation accuracy, outperforming existing methods.
