# Breast-Cancer-Detection-Using-Multiscale-Cascaded-Convolutions

# Abstract
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

# Problem Statement
Breast cancer detection from ultrasound images is difficult due to noise, variability, and the
need for manual analysis by experts. There is no automated system that can accurately
segment tumor regions and classify cancer types. This project aims to develop deep learning
models to automate both segmentation and classification, providing faster and more accurate
diagnosis.

# Dataset Sources:
https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset

https://data.mendeley.com/datasets/7fvgj4jsp7/1

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


